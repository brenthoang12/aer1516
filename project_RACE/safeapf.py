from dataclasses import dataclass, field
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from crazyflow.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.sim import use_box_collision
from crazyflow.sim.visualize import draw_line, draw_points


GOAL = np.array([3.0, 3.0])
WORLD_MIN = np.array([0.0, 0.0, 0.0])
WORLD_MAX = np.array([6.0, 7.0, 3.0])
PREVIEW_STEPS = 12
CTRL_FREQ = 100
SIM_STEPS = 5000
Z_REF = 0.5
YAW_REF = 0.0
STEP_GAIN = 0.15
GOAL_TOL = 0.05

SPHERE_SPECS = [
    {
        "name": "sphere2",
        "center": np.array([3.0, 2.0]),
        "radius": 0.25,
        "z": 0.25,
        "rgba": (1.0, 0.0, 0.0, 1.0),
        "motion": "circle",
        "anchor": np.array([3.0, 2.0]),
        "orbit_radius": 0.45,
        "angular_speed": 0.8,
        "phase": 0.0,
    },
]

CAVE_WALLS = [
    {
        "name": "uwall_bottom",
        "center": np.array([1.81819805, 1.81819805]),
        "half_extents": np.array([0.5, 0.05]),
        "angle_deg": -45.0,
        "z": 0.0,
        "rgba": (0.8, 0.2, 0.2, 1.0),
    },
    {
        "name": "uwall_left",
        "center": np.array([1.18180195, 1.81819805]),
        "half_extents": np.array([0.05, 0.5]),
        "angle_deg": -45.0,
        "z": 0.0,
        "rgba": (0.8, 0.2, 0.2, 1.0),
    },
    {
        "name": "uwall_right",
        "center": np.array([1.81819805, 1.18180195]),
        "half_extents": np.array([0.05, 0.5]),
        "angle_deg": -45.0,
        "z": 0.0,
        "rgba": (0.8, 0.2, 0.2, 1.0),
    },
]


@dataclass
class MovingSphere:
    name: str
    center: np.ndarray
    radius: float
    z: float
    rgba: tuple[float, float, float, float]
    motion: str = "static"
    anchor: np.ndarray | None = None
    orbit_radius: float = 0.0
    angular_speed: float = 0.0
    phase: float = 0.0
    velocity: np.ndarray | None = None
    max_speed: float = 0.3
    turn_interval: float = 1.5
    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)
    _time_to_turn: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self):
        self.center = np.array(self.center, dtype=float)
        self.anchor = np.array(self.anchor if self.anchor is not None else self.center, dtype=float)
        self.velocity = np.array(self.velocity if self.velocity is not None else np.zeros(2), dtype=float)
        self._rng = np.random.default_rng(self.seed)
        self._time_to_turn = self.turn_interval
        if self.motion == "random" and np.linalg.norm(self.velocity) < 1e-6:
            self.velocity = self._random_velocity()

    def _random_velocity(self) -> np.ndarray:
        direction = self._rng.normal(size=2)
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction = np.array([1.0, 0.0])
            norm = 1.0
        return direction / norm * self.max_speed

    def update(self, t: float, dt: float):
        if self.motion == "circle":
            theta = self.phase + self.angular_speed * t
            self.center = self.anchor + self.orbit_radius * np.array([np.cos(theta), np.sin(theta)])
            return

        if self.motion == "random":
            self._time_to_turn -= dt
            if self._time_to_turn <= 0.0:
                self.velocity = self._random_velocity()
                self._time_to_turn = self.turn_interval

            self.center = self.center + self.velocity * dt
            for axis in range(2):
                lower = WORLD_MIN[axis] + self.radius
                upper = WORLD_MAX[axis] - self.radius
                if self.center[axis] < lower:
                    self.center[axis] = lower
                    self.velocity[axis] *= -1.0
                elif self.center[axis] > upper:
                    self.center[axis] = upper
                    self.velocity[axis] *= -1.0
            return

        if self.motion != "static":
            raise ValueError(f"Unsupported obstacle motion: {self.motion}")


def clone_sphere_obstacles(specs):
    obstacles = []
    for spec in specs:
        obstacles.append(
            MovingSphere(
                name=spec["name"],
                center=np.array(spec["center"], dtype=float),
                radius=float(spec["radius"]),
                z=float(spec["z"]),
                rgba=tuple(spec["rgba"]),
                motion=spec.get("motion", "static"),
                anchor=np.array(spec.get("anchor", spec["center"]), dtype=float),
                orbit_radius=float(spec.get("orbit_radius", 0.0)),
                angular_speed=float(spec.get("angular_speed", 0.0)),
                phase=float(spec.get("phase", 0.0)),
                velocity=np.array(spec.get("velocity", np.zeros(2)), dtype=float),
                max_speed=float(spec.get("max_speed", 0.3)),
                turn_interval=float(spec.get("turn_interval", 1.5)),
                seed=spec.get("seed"),
            )
        )
    return obstacles


def update_obstacles(obstacles, t, dt):
    for obstacle in obstacles:
        obstacle.update(t, dt)


def quat_to_yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def rotmat(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, -s], [s, c]])


def rgba_str(rgba):
    return " ".join(f"{v:g}" for v in rgba)


def sphere_nearest_point(p, sphere):
    diff = p - sphere.center
    dist = np.linalg.norm(diff)
    if dist < 1e-9:
        return sphere.center + np.array([sphere.radius, 0.0])
    return sphere.center + sphere.radius * diff / dist


def sphere_repulsion_2d(p, sphere, eta, Qstar):
    diff = p - sphere.center
    center_dist = np.linalg.norm(diff)
    surface_dist = center_dist - sphere.radius

    if surface_dist < 1e-6:
        if center_dist < 1e-9:
            direction = np.array([1.0, 0.0])
        else:
            direction = diff / center_dist
        return eta * (1.0 / 1e-6 - 1.0 / Qstar) * (1.0 / (1e-6 ** 2)) * direction

    if surface_dist > Qstar:
        return np.zeros(2)

    direction = diff / (center_dist + 1e-9)
    return eta * (1.0 / Qstar - 1.0 / surface_dist) * (1.0 / (surface_dist ** 2)) * direction


def wall_local_point(p, wall):
    angle = np.deg2rad(wall["angle_deg"])
    return rotmat(angle).T @ (p - wall["center"])


def wall_world_point(local_p, wall):
    angle = np.deg2rad(wall["angle_deg"])
    return wall["center"] + rotmat(angle) @ local_p


def wall_nearest_point(p, wall):
    local_p = wall_local_point(p, wall)
    local_closest = np.clip(local_p, -wall["half_extents"], wall["half_extents"])
    return wall_world_point(local_closest, wall)


def wall_repulsion_2d(p, wall, eta, Qstar):
    local_p = wall_local_point(p, wall)
    local_closest = np.clip(local_p, -wall["half_extents"], wall["half_extents"])
    local_diff = local_p - local_closest
    dist = np.linalg.norm(local_diff)

    if dist < 1e-6:
        pen = wall["half_extents"] - np.abs(local_p)
        axis = int(np.argmin(pen))
        local_direction = np.zeros(2)
        local_direction[axis] = 1.0 if local_p[axis] >= 0.0 else -1.0
        world_direction = rotmat(np.deg2rad(wall["angle_deg"])) @ local_direction
        world_direction /= np.linalg.norm(world_direction) + 1e-9
        return eta * (1.0 / 1e-6 - 1.0 / Qstar) * (1.0 / (1e-6 ** 2)) * world_direction

    if dist > Qstar:
        return np.zeros(2)

    local_grad = eta * (1.0 / Qstar - 1.0 / dist) * (1.0 / (dist ** 2)) * local_diff
    return rotmat(np.deg2rad(wall["angle_deg"])) @ local_grad


def wall_polygon(wall):
    hx, hy = wall["half_extents"]
    local_corners = np.array([
        [-hx, -hy],
        [hx, -hy],
        [hx, hy],
        [-hx, hy],
        [-hx, -hy],
    ])
    return np.array([wall_world_point(corner, wall) for corner in local_corners])


def build_scene_xml():
    center = 0.5 * (WORLD_MIN + WORLD_MAX)
    parts = [
        '<mujoco model="SAPF Scene">',
        '    <option timestep="0.001"/>',
        '    <asset>',
        '        <material name="boundary" rgba="0.5 0.7 0.9 0.12"/>',
        '    </asset>',
        '    <worldbody>',
        '',
        '        <!-- Floor -->',
        '        <geom type="plane" size="0 0 0.05" rgba="0.8 0.8 0.8 1"/>',
        '',
        '        <!-- Visual boundary walls -->',
        (
            f'        <geom name="ceiling" type="box" pos="{center[0]:g} {center[1]:g} {WORLD_MAX[2]:g}" '
            f'size="{0.5 * (WORLD_MAX[0] - WORLD_MIN[0]):g} {0.5 * (WORLD_MAX[1] - WORLD_MIN[1]):g} 0.02" '
            'material="boundary" contype="0" conaffinity="0"/>'
        ),
        (
            f'        <geom name="wall_xn" type="box" pos="{WORLD_MIN[0]:g} {center[1]:g} {center[2]:g}" '
            f'size="0.02 {0.5 * (WORLD_MAX[1] - WORLD_MIN[1]):g} {center[2]:g}" '
            'material="boundary" contype="0" conaffinity="0"/>'
        ),
        (
            f'        <geom name="wall_xp" type="box" pos="{WORLD_MAX[0]:g} {center[1]:g} {center[2]:g}" '
            f'size="0.02 {0.5 * (WORLD_MAX[1] - WORLD_MIN[1]):g} {center[2]:g}" '
            'material="boundary" contype="0" conaffinity="0"/>'
        ),
        (
            f'        <geom name="wall_yn" type="box" pos="{center[0]:g} {WORLD_MIN[1]:g} {center[2]:g}" '
            f'size="{0.5 * (WORLD_MAX[0] - WORLD_MIN[0]):g} 0.02 {center[2]:g}" '
            'material="boundary" contype="0" conaffinity="0"/>'
        ),
        (
            f'        <geom name="wall_yp" type="box" pos="{center[0]:g} {WORLD_MAX[1]:g} {center[2]:g}" '
            f'size="{0.5 * (WORLD_MAX[0] - WORLD_MIN[0]):g} 0.02 {center[2]:g}" '
            'material="boundary" contype="0" conaffinity="0"/>'
        ),
        '',
        '        <!-- U-shape walls with collision -->',
    ]

    for wall in CAVE_WALLS:
        cx, cy = wall["center"]
        hx, hy = wall["half_extents"]
        parts += [
            (
                f'        <body name="{wall["name"]}" '
                f'pos="{cx:g} {cy:g} {wall["z"]:g}" euler="0 0 {wall["angle_deg"]:g}">'
            ),
            f'            <geom type="box" size="{hx:g} {hy:g} 0.5"',
            f'                  rgba="{rgba_str(wall["rgba"])}"',
            '                  contype="1" conaffinity="1"',
            '                  density="1000"/>',
            '        </body>',
            '',
        ]

    gx, gy = GOAL
    parts += [
        '        <!-- Goal marker (visual only) -->',
        f'        <body name="goal" pos="{gx:g} {gy:g} 0.5">',
        '            <geom type="sphere" size="0.1"',
        '                  rgba="0 1 0 1"',
        '                  contype="0" conaffinity="0"/>',
        '        </body>',
        '',
        '    </worldbody>',
        '</mujoco>',
    ]
    return "\n".join(parts)


def apf_gradient(p, theta, sphere_obstacles, params):
    zeta = params["zeta"]
    eta = params["eta"]
    dstar = params["dstar"]
    Qstar = params["Qstar"]
    dsafe = params["dsafe"]
    dvort = params["dvort"]
    alpha_th = params["alpha_th"]

    diff_goal = p - GOAL
    d_goal = np.linalg.norm(diff_goal)
    if d_goal <= dstar:
        grad_att = zeta * diff_goal
    else:
        grad_att = zeta * dstar * diff_goal / (d_goal + 1e-9)

    grad_rep_total = np.zeros(2)

    for sphere in sphere_obstacles:
        nearest = sphere_nearest_point(p, sphere)
        dist = max(np.linalg.norm(p - nearest), 1e-6)
        if dist > Qstar:
            continue

        grad_rep = sphere_repulsion_2d(p, sphere, eta, Qstar)
        alpha = theta - np.arctan2(nearest[1] - p[1], nearest[0] - p[0])
        alpha = wrap(alpha)
        direction_sign = 1 if abs(alpha) <= alpha_th else -1

        if dist <= dsafe:
            drel = 0.0
        elif dist >= dvort:
            drel = 1.0
        else:
            drel = (dist - dsafe) / (dvort - dsafe)
        drel = np.clip(drel, 0.0, 1.0)

        gamma = np.pi * direction_sign * drel
        grad_rep_total += rotmat(gamma) @ grad_rep

    for wall in CAVE_WALLS:
        nearest = wall_nearest_point(p, wall)
        dist = max(np.linalg.norm(p - nearest), 1e-6)
        if dist > Qstar:
            continue

        grad_rep = wall_repulsion_2d(p, wall, eta, Qstar)
        if np.linalg.norm(grad_rep) < 1e-9:
            continue

        alpha = theta - np.arctan2(nearest[1] - p[1], nearest[0] - p[0])
        alpha = wrap(alpha)
        direction_sign = 1 if abs(alpha) <= alpha_th else -1

        if dist <= dsafe:
            drel = 0.0
        elif dist >= dvort:
            drel = 1.0
        else:
            drel = (dist - dsafe) / (dvort - dsafe)
        drel = np.clip(drel, 0.0, 1.0)

        gamma = np.pi * direction_sign * drel
        grad_rep_total += rotmat(gamma) @ grad_rep

    return grad_att + grad_rep_total


def rollout_preview(start, theta, sphere_obstacles, params, steps=PREVIEW_STEPS, step_gain=STEP_GAIN):
    path = [start.copy()]
    p = start.copy()
    heading = theta
    for _ in range(steps):
        if np.linalg.norm(p - GOAL) < GOAL_TOL:
            break
        g = apf_gradient(p, heading, sphere_obstacles, params)
        direction = -g
        norm_dir = np.linalg.norm(direction)
        if norm_dir < 1e-6:
            break
        direction /= norm_dir
        p = p + step_gain * direction
        p[0] = np.clip(p[0], WORLD_MIN[0] + 0.01, WORLD_MAX[0] - 0.01)
        p[1] = np.clip(p[1], WORLD_MIN[1] + 0.01, WORLD_MAX[1] - 0.01)
        heading = np.arctan2(direction[1], direction[0])
        path.append(p.copy())
    return np.array(path)


def draw_scene(sim, preview_path, sphere_obstacles, start_pos):
    if len(preview_path) >= 2:
        draw_line(
            sim,
            np.column_stack([preview_path, np.full(len(preview_path), Z_REF)]),
            rgba=np.array([0.2, 0.4, 1.0, 0.85]),
            start_size=1.5,
            end_size=1.5,
        )

    draw_points(sim, np.array([[GOAL[0], GOAL[1], Z_REF]]), rgba=np.array([0.0, 1.0, 0.2, 1.0]), size=0.08)
    draw_points(sim, np.array([[start_pos[0], start_pos[1], Z_REF]]), rgba=np.array([1.0, 1.0, 1.0, 0.8]), size=0.05)

    for obstacle in sphere_obstacles:
        draw_points(
            sim,
            np.array([[obstacle.center[0], obstacle.center[1], obstacle.z]]),
            rgba=np.array(obstacle.rgba),
            size=obstacle.radius,
        )


def main():
    sphere_obstacles = clone_sphere_obstacles(SPHERE_SPECS)

    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        tmp.write(build_scene_xml().encode())
        tmp.flush()
        xml_path = Path(tmp.name)

    sim = Sim(control=Control.state, xml_path=xml_path)
    use_box_collision(sim, enable=True)
    sim.reset()

    params = dict(
        zeta=1.1547,
        eta=0.0732,
        dstar=0.3,
        Qstar=0.5,
        dsafe=0.2,
        dvort=0.35,
        alpha_th=np.deg2rad(5),
    )

    fps = 60
    traj = []
    obstacle_traces = [[] for _ in sphere_obstacles]
    cmd = np.zeros((sim.n_worlds, sim.n_drones, 13))
    start_pos = np.array(sim.data.states.pos[0, 0, :2])

    print("Running full APF + Safe-APF with moving sphere obstacles...")

    for step in range(SIM_STEPS):
        t = step / CTRL_FREQ
        update_obstacles(sphere_obstacles, t, 1.0 / CTRL_FREQ)

        states = sim.data.states
        pos = states.pos[0, 0]
        p = np.array([pos[0], pos[1]])
        theta = quat_to_yaw(states.quat[0, 0])

        traj.append(p.copy())
        for idx, obstacle in enumerate(sphere_obstacles):
            obstacle_traces[idx].append(obstacle.center.copy())

        if np.linalg.norm(p - GOAL) < GOAL_TOL:
            print("Reached goal.")
            break

        g = apf_gradient(p, theta, sphere_obstacles, params)
        direction = -g
        norm_dir = np.linalg.norm(direction)
        if norm_dir > 1e-6:
            direction /= norm_dir
        else:
            direction = np.zeros(2)

        p_des = p + STEP_GAIN * direction
        p_des[0] = np.clip(p_des[0], WORLD_MIN[0] + 0.01, WORLD_MAX[0] - 0.01)
        p_des[1] = np.clip(p_des[1], WORLD_MIN[1] + 0.01, WORLD_MAX[1] - 0.01)

        cmd[..., 0] = p_des[0]
        cmd[..., 1] = p_des[1]
        cmd[..., 2] = Z_REF
        cmd[..., 3] = YAW_REF
        cmd[..., 6] = 0.0

        sim.state_control(cmd)
        sim.step(sim.freq // sim.control_freq)

        if (step * fps) % sim.control_freq < fps:
            preview_path = rollout_preview(p, theta, sphere_obstacles, params)
            draw_scene(sim, preview_path, sphere_obstacles, start_pos)
            sim.render()
    else:
        print("Time limit reached.")

    sim.close()

    traj = np.array(traj)
    plt.figure(figsize=(6, 6))
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    world = np.array([
        [WORLD_MIN[0], WORLD_MIN[1]],
        [WORLD_MAX[0], WORLD_MIN[1]],
        [WORLD_MAX[0], WORLD_MAX[1]],
        [WORLD_MIN[0], WORLD_MAX[1]],
        [WORLD_MIN[0], WORLD_MIN[1]],
    ])
    plt.plot(world[:, 0], world[:, 1], color="0.4", linewidth=1.5, alpha=0.8, label="Boundary")

    for trace, obstacle in zip(obstacle_traces, sphere_obstacles):
        trace = np.array(trace)
        if len(trace) > 1:
            plt.plot(trace[:, 0], trace[:, 1], "r--", alpha=0.35)
        circle = plt.Circle(obstacle.center, obstacle.radius, color="r", alpha=0.35)
        plt.gca().add_patch(circle)

    for wall in CAVE_WALLS:
        polygon = wall_polygon(wall)
        plt.plot(polygon[:, 0], polygon[:, 1], "r-", alpha=0.7)

    if len(traj) > 0:
        plt.plot(traj[:, 0], traj[:, 1], "b-", label="Trajectory")
        plt.scatter(traj[0, 0], traj[0, 1], c="w", edgecolors="k", s=50, label="Start")
    plt.scatter(GOAL[0], GOAL[1], c="g", marker="x", s=80, label="Goal")
    plt.legend()
    plt.title("Safe-APF with moving spheres and live preview")
    plt.show()


if __name__ == "__main__":
    main()
