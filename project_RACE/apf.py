from dataclasses import dataclass, field
from pathlib import Path

import imageio.v3 as iio
from datetime import datetime 
import numpy as np
import os
import glfw
import jax

from crazyflow.control import Control
from crazyflow.sim import Physics, Sim
from crazyflow.sim.visualize import draw_line, draw_points

# ---------------------------------------------------------------------------
# World parameters
# ---------------------------------------------------------------------------
BORDER_MIN = np.array([0.0, 0.0, 0.0])
BORDER_MAX = np.array([3.0, 3.0, 2.0])

START = np.array([0.3, 0.3, 0.5])
GOAL  = np.array([2.7, 2.7, 1.5])


@dataclass
class MovingObstacle:
    radius: float
    motion: str
    center: np.ndarray | None = None
    anchor: np.ndarray | None = None
    angular_speed: float = 0.0
    orbit_radius: float = 0.0
    axis: tuple[int, int] = (0, 1)
    phase: float = 0.0
    velocity: np.ndarray | None = None
    max_speed: float = 0.25
    turn_interval: float = 1.5
    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)
    _time_to_turn: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self):
        self.center = np.array(
            self.center if self.center is not None else self.anchor,
            dtype=float,
        )
        self.anchor = np.array(
            self.anchor if self.anchor is not None else self.center,
            dtype=float,
        )
        self.velocity = np.array(
            self.velocity if self.velocity is not None else np.zeros(3),
            dtype=float,
        )
        self._rng = np.random.default_rng(self.seed)
        self._time_to_turn = self.turn_interval
        if self.motion == "random" and np.linalg.norm(self.velocity) < 1e-6:
            self.velocity = self._random_velocity()

    def _random_velocity(self) -> np.ndarray:
        direction = self._rng.normal(size=3)
        direction[2] *= 0.6
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction = np.array([1.0, 0.0, 0.0])
            norm = 1.0
        return direction / norm * self.max_speed

    def update(self, t: float, dt: float):
        if self.motion == "circle":
            theta = self.phase + self.angular_speed * t
            center = self.anchor.copy()
            a0, a1 = self.axis
            center[a0] += self.orbit_radius * np.cos(theta)
            center[a1] += self.orbit_radius * np.sin(theta)
            self.center = center
            return

        if self.motion == "random":
            self._time_to_turn -= dt
            if self._time_to_turn <= 0.0:
                self.velocity = self._random_velocity()
                self._time_to_turn = self.turn_interval

            self.center = self.center + self.velocity * dt
            for axis in range(3):
                lower = BORDER_MIN[axis] + self.radius
                upper = BORDER_MAX[axis] - self.radius
                if self.center[axis] < lower:
                    self.center[axis] = lower
                    self.velocity[axis] *= -1.0
                elif self.center[axis] > upper:
                    self.center[axis] = upper
                    self.velocity[axis] *= -1.0
            return

        if self.motion != "static":
            raise ValueError(f"Unsupported obstacle motion: {self.motion}")


# Planner-only sphere obstacles with motion models.
OBSTACLES = [
    MovingObstacle(
        center=np.array([1.0, 1.0, 0.8]),
        anchor=np.array([1.0, 1.0, 0.8]),
        radius=0.25,
        motion="circle",
        orbit_radius=0.22,
        angular_speed=0.9,
        axis=(0, 1),
    ),
    MovingObstacle(
        center=np.array([1.8, 1.5, 1.2]),
        anchor=np.array([1.8, 1.5, 1.2]),
        radius=0.25,
        motion="circle",
        orbit_radius=0.18,
        angular_speed=-1.2,
        axis=(1, 2),
        phase=np.pi / 3,
    ),
    MovingObstacle(
        center=np.array([1.3, 2.2, 1.0]),
        radius=0.25,
        motion="random",
        velocity=np.array([0.16, -0.12, 0.08]),
        max_speed=0.22,
        turn_interval=1.8,
        seed=7,
    ),
]

# APF tuning
K_ATT      = 1.5    # attractive gain
K_REP_WALL = 0.8    # repulsive gain - boundary walls
K_REP_OBS  = 1.2    # repulsive gain - sphere obstacles
D_REP      = 0.4    # influence distance from walls / obstacle surface (m)
STEP_LEN   = 0.1   # gradient-descent step size (m)
GOAL_TOL   = 0.08   # goal-reached tolerance (m)
PREVIEW_STEPS = 10

# Simulation
SIM_FREQ    = 500
STATE_FREQ  = 100
CTRL_FREQ   = 50
SIM_DURATION = 20.0

CAPTURE_DIR = Path(__file__).resolve().parent / "captures"
CAPTURE_FPS = 24
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURE_CAMERA = -1
SAVE_VIDEO = False
SHOW_WINDOW = True


# ---------------------------------------------------------------------------
# APF functions
# ---------------------------------------------------------------------------

def attractive_force(pos: np.ndarray, goal: np.ndarray) -> np.ndarray:
    return K_ATT * (goal - pos)


def wall_repulsive_force(pos: np.ndarray) -> np.ndarray:
    force = np.zeros(3)
    for i in range(3):
        d_min = pos[i] - BORDER_MIN[i]
        d_max = BORDER_MAX[i] - pos[i]
        if 1e-6 < d_min < D_REP:
            force[i] += K_REP_WALL * (1.0 / d_min - 1.0 / D_REP) / d_min ** 2
        if 1e-6 < d_max < D_REP:
            force[i] -= K_REP_WALL * (1.0 / d_max - 1.0 / D_REP) / d_max ** 2
    return force


def obstacle_repulsive_force(pos: np.ndarray, obstacles: list[MovingObstacle]) -> np.ndarray:
    force = np.zeros(3)
    for obstacle in obstacles:
        diff = pos - obstacle.center
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            diff = np.array([1e-6, 0.0, 0.0])
            dist = 1e-6
        d_surface = dist - obstacle.radius  # distance from obstacle surface
        if d_surface < 1e-6:
            d_surface = 1e-6
        if d_surface < D_REP:
            magnitude = K_REP_OBS * (1.0 / d_surface - 1.0 / D_REP) / d_surface ** 2
            force += magnitude * (diff / dist)
    return force


def apf_step(pos: np.ndarray, goal: np.ndarray, obstacles: list[MovingObstacle]) -> np.ndarray:
    total = (attractive_force(pos, goal)
             + wall_repulsive_force(pos)
             + obstacle_repulsive_force(pos, obstacles))
    norm = np.linalg.norm(total)
    if norm > 1e-6:
        total /= norm
    return np.clip(pos + STEP_LEN * total, BORDER_MIN + 0.01, BORDER_MAX - 0.01)


def update_obstacles(obstacles: list[MovingObstacle], t: float, dt: float):
    for obstacle in obstacles:
        obstacle.update(t, dt)


def rollout_preview(start: np.ndarray,
                    goal: np.ndarray,
                    obstacles: list[MovingObstacle],
                    max_iters: int = PREVIEW_STEPS) -> np.ndarray:
    path = [start.copy()]
    pos = start.copy()
    for _ in range(max_iters):
        if np.linalg.norm(pos - goal) < GOAL_TOL:
            break
        pos = apf_step(pos, goal, obstacles)
        path.append(pos.copy())
    path.append(goal.copy())
    return np.array(path)


def clone_obstacles(obstacles: list[MovingObstacle]) -> list[MovingObstacle]:
    return [
        MovingObstacle(
            center=obstacle.center.copy(),
            anchor=obstacle.anchor.copy(),
            radius=obstacle.radius,
            motion=obstacle.motion,
            angular_speed=obstacle.angular_speed,
            orbit_radius=obstacle.orbit_radius,
            axis=obstacle.axis,
            phase=obstacle.phase,
            velocity=obstacle.velocity.copy(),
            max_speed=obstacle.max_speed,
            turn_interval=obstacle.turn_interval,
            seed=obstacle.seed,
        )
        for obstacle in obstacles
    ]


# ---------------------------------------------------------------------------
# Visualization helpers (must be called every frame before sim.render())
# ---------------------------------------------------------------------------

def draw_scene(sim: Sim, path: np.ndarray, obstacles: list[MovingObstacle]):
    # Planned path - blue preview from the current state.
    if len(path) >= 2:
        idx = np.linspace(0, len(path) - 1, min(200, len(path)), dtype=int)
        draw_line(sim, path[idx], rgba=np.array([0.2, 0.4, 1.0, 0.8]), start_size=1.5, end_size=1.5)

    # Goal - large green sphere
    draw_points(sim, GOAL[None], rgba=np.array([0.0, 1.0, 0.2, 1.0]), size=0.08)

    # Start - small white sphere
    draw_points(sim, START[None], rgba=np.array([1.0, 1.0, 1.0, 0.8]), size=0.05)

    # Obstacles - red spheres (approximate radius with point size)
    for obstacle in obstacles:
        draw_points(
            sim,
            obstacle.center[None],
            rgba=np.array([1.0, 0.15, 0.1, 0.9]),
            size=obstacle.radius,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    obstacles = clone_obstacles(OBSTACLES)
    initial_preview = rollout_preview(START, GOAL, obstacles)
    print("Planning path with APF...")
    print(f"  Initial preview waypoints: {len(initial_preview)}")
    print("DISPLAY =", os.environ.get("DISPLAY"))
    print("WAYLAND_DISPLAY =", os.environ.get("WAYLAND_DISPLAY"))
    print("GLFW init =", glfw.init())
    print("JAX devices =", jax.devices())

    sim = Sim(
        n_worlds=1,
        n_drones=1,
        physics=Physics.first_principles,
        control=Control.state,
        freq=SIM_FREQ,
        state_freq=STATE_FREQ,
        attitude_freq=SIM_FREQ,
        device="gpu",
    )
    sim.reset()

    # Place drone at start
    sim.data = sim.data.replace(
        states=sim.data.states.replace(
            pos=sim.data.states.pos.at[0, 0].set(START),
            rotor_vel=sim.data.states.rotor_vel.at[0, 0].set(np.ones(4) * 20000),
        )
    )

    cmd = np.zeros((1, 1, 13))
    fps = CAPTURE_FPS
    steps_per_ctrl = sim.freq // CTRL_FREQ

    preview_path = initial_preview
    video_frames = []

    print("Simulating...")
    try:
        for i in range(int(SIM_DURATION * CTRL_FREQ)):
            t = i / CTRL_FREQ
            update_obstacles(obstacles, t, 1.0 / CTRL_FREQ)
            pos = np.array(sim.data.states.pos[0, 0])
            next_target = apf_step(pos, GOAL, obstacles)

            cmd[0, 0, :3] = next_target
            sim.state_control(cmd)
            sim.step(steps_per_ctrl)

            if (i * fps) % CTRL_FREQ < fps:
                preview_path = rollout_preview(pos, GOAL, obstacles)
                draw_scene(sim, preview_path, obstacles)
                if SAVE_VIDEO:
                    frame = sim.render(
                        mode="rgb_array",
                        camera=CAPTURE_CAMERA,
                        width=CAPTURE_WIDTH,
                        height=CAPTURE_HEIGHT,
                    )
                    if frame is not None:
                        video_frames.append(frame)
                if SHOW_WINDOW:
                    sim.render(
                        camera=CAPTURE_CAMERA,
                        width=CAPTURE_WIDTH,
                        height=CAPTURE_HEIGHT,
                    )

            if np.linalg.norm(pos - GOAL) < GOAL_TOL:
                print(f"Goal reached! t={i / CTRL_FREQ:.2f}s")
                break
        else:
            print("Time limit reached.")
    finally:
        if SAVE_VIDEO and video_frames:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
            output_path = CAPTURE_DIR / f"apf_capture_Cam{CAPTURE_CAMERA}_{timestamp}.mp4"
            iio.imwrite(output_path, video_frames, fps=CAPTURE_FPS)
            print(f"Saved capture to {output_path}")
        sim.close()


if __name__ == "__main__":
    main()
