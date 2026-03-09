import numpy as np

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

# Sphere obstacles: (center [x,y,z], radius)
OBSTACLES = [
    (np.array([1.0, 1.0, 0.8]), 0.25),
    (np.array([1.8, 1.5, 1.2]), 0.25),
    (np.array([1.3, 2.2, 1.0]), 0.25),
]

# APF tuning
K_ATT      = 1.5    # attractive gain
K_REP_WALL = 0.8    # repulsive gain – boundary walls
K_REP_OBS  = 1.2    # repulsive gain – sphere obstacles
D_REP      = 0.4    # influence distance from walls / obstacle surface (m)
STEP_LEN   = 0.03   # gradient-descent step size (m)
GOAL_TOL   = 0.08   # goal-reached tolerance (m)

# Simulation
SIM_FREQ    = 500
STATE_FREQ  = 100
CTRL_FREQ   = 100
SIM_DURATION = 20.0


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


def obstacle_repulsive_force(pos: np.ndarray) -> np.ndarray:
    force = np.zeros(3)
    for center, radius in OBSTACLES:
        diff = pos - center
        dist = np.linalg.norm(diff)
        d_surface = dist - radius  # distance from obstacle surface
        if d_surface < 1e-6:
            d_surface = 1e-6
        if d_surface < D_REP:
            magnitude = K_REP_OBS * (1.0 / d_surface - 1.0 / D_REP) / d_surface ** 2
            force += magnitude * (diff / dist)
    return force


def apf_step(pos: np.ndarray, goal: np.ndarray) -> np.ndarray:
    total = (attractive_force(pos, goal)
             + wall_repulsive_force(pos)
             + obstacle_repulsive_force(pos))
    norm = np.linalg.norm(total)
    if norm > 1e-6:
        total /= norm
    return np.clip(pos + STEP_LEN * total, BORDER_MIN + 0.01, BORDER_MAX - 0.01)


def plan_path(max_iters: int = 3000) -> np.ndarray:
    path = [START.copy()]
    pos = START.copy()
    for _ in range(max_iters):
        if np.linalg.norm(pos - GOAL) < GOAL_TOL:
            break
        pos = apf_step(pos, GOAL)
        path.append(pos.copy())
    path.append(GOAL.copy())
    return np.array(path)


# ---------------------------------------------------------------------------
# Visualization helpers (must be called every frame before sim.render())
# ---------------------------------------------------------------------------

def draw_scene(sim: Sim, path: np.ndarray):
    # Planned path – blue (downsample to at most 200 points for the marker budget)
    if len(path) >= 2:
        idx = np.linspace(0, len(path) - 1, min(200, len(path)), dtype=int)
        draw_line(sim, path[idx], rgba=np.array([0.2, 0.4, 1.0, 0.8]), start_size=1.5, end_size=1.5)

    # Goal – large green sphere
    draw_points(sim, GOAL[None], rgba=np.array([0.0, 1.0, 0.2, 1.0]), size=0.08)

    # Start – small white sphere
    draw_points(sim, START[None], rgba=np.array([1.0, 1.0, 1.0, 0.8]), size=0.05)

    # Obstacles – red spheres (approximate radius with point size)
    obs_centers = np.array([c for c, _ in OBSTACLES])
    obs_radii   = [r for _, r in OBSTACLES]
    for center, radius in zip(obs_centers, obs_radii):
        draw_points(sim, center[None], rgba=np.array([1.0, 0.15, 0.1, 0.9]), size=radius)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Planning path with APF...")
    path = plan_path()
    print(f"  Waypoints: {len(path)}")

    sim = Sim(
        n_worlds=1,
        n_drones=1,
        physics=Physics.first_principles,
        control=Control.state,
        freq=SIM_FREQ,
        state_freq=STATE_FREQ,
        attitude_freq=SIM_FREQ,
        device="cpu",
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
    fps = 60
    steps_per_ctrl = sim.freq // CTRL_FREQ
    waypoint_idx = 0

    print("Simulating...")
    try:
        for i in range(int(SIM_DURATION * CTRL_FREQ)):
            pos = np.array(sim.data.states.pos[0, 0])

            # Advance to next waypoint when close enough
            while waypoint_idx < len(path) - 1 and np.linalg.norm(pos - path[waypoint_idx]) < GOAL_TOL:
                waypoint_idx += 1

            cmd[0, 0, :3] = path[waypoint_idx]
            sim.state_control(cmd)
            sim.step(steps_per_ctrl)

            if (i * fps) % CTRL_FREQ < fps:
                draw_scene(sim, path)
                sim.render()

            if np.linalg.norm(pos - GOAL) < GOAL_TOL:
                print(f"Goal reached! t={i / CTRL_FREQ:.2f}s")
                break
        else:
            print("Time limit reached.")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
