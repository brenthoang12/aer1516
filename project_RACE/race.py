import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import shlex
import sys
import tempfile
import xml.etree.ElementTree as ET

import imageio.v3 as iio
import jax.numpy as jnp
import mujoco
import numpy as np

from crazyflow.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.sim import use_box_collision
from crazyflow.sim.visualize import draw_line, draw_points


PREVIEW_STEPS = 12
CTRL_FREQ = 100
Z_REF = 0.5
YAW_REF = 0.0
STEP_GAIN = 0.05
GOAL_TOL = 0.05

FREE_CAM = {
    "lookat": np.array([2.540425, 3.509272, 0.366132]),
    "distance": 7.599483,
    "azimuth": 95.094340,
    "elevation": -24.189189,
}

CAPTURE_DIR = Path(__file__).resolve().parent / "captures"
CAPTURE_FPS = 24
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURE_CAMERA = -1

DEFAULT_TIMEOUT = 50.0
DEFAULT_MOVING_SPHERES = 0
DEFAULT_MOTION = "circle"
DEFAULT_SPHERE_RADIUS = 0.25
DEFAULT_SPHERE_RGBA = (1.0, 0.0, 0.0, 1.0)
IDENTITY_QUAT = jnp.array([1.0, 0.0, 0.0, 0.0])

GOAL = np.array([3.0, 3.0])
WORLD_MIN = np.array([0.0, 0.0, 0.0])
WORLD_MAX = np.array([6.0, 7.0, 3.0])
STATIC_BOX_OBSTACLES = []
STATIC_SPHERE_OBSTACLES = []


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


def parse_args():
    parser = argparse.ArgumentParser(description="Run Safe-APF with a static XML map and optional moving spheres.")
    parser.add_argument("map_xml", type=Path, help="XML file containing the static environment and static obstacles.")
    parser.add_argument("--moving-spheres", type=int, default=DEFAULT_MOVING_SPHERES, help="Number of moving sphere obstacles to add at runtime.")
    parser.add_argument("--motion", choices=["static", "circle", "random"], default=DEFAULT_MOTION, help="Motion model used for all moving spheres.")
    parser.add_argument("--save-video", action="store_true", help="Save an MP4 capture of the simulation.")
    parser.add_argument("--no-vis", action="store_true", help="Disable live rendering and the final matplotlib plot.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Simulation timeout in seconds.")

    # NEW FLAG: enable Safe‑APF vortex rotation
    parser.add_argument("--safe-apf", action="store_true",
                        help="Enable Safe-APF vortex field. If not set, uses regular APF.")

    return parser.parse_args()


def parse_vec(text, expected_len):
    values = np.fromstring(text or "", sep=" ")
    if values.size != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {values.size} from {text!r}")
    return values


def rgba_or_default(text, default):
    if not text:
        return default
    rgba = parse_vec(text, 4)
    return tuple(float(v) for v in rgba)


def parse_map(map_xml: Path):
    root = ET.parse(map_xml).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{map_xml} is missing <worldbody>.")

    wall_xn = worldbody.find("./geom[@name='wall_xn']")
    wall_xp = worldbody.find("./geom[@name='wall_xp']")
    wall_yn = worldbody.find("./geom[@name='wall_yn']")
    wall_yp = worldbody.find("./geom[@name='wall_yp']")
    ceiling = worldbody.find("./geom[@name='ceiling']")
    goal_body = worldbody.find("./body[@name='goal']")

    if None in (wall_xn, wall_xp, wall_yn, wall_yp, ceiling, goal_body):
        raise ValueError("Map XML must define wall_xn, wall_xp, wall_yn, wall_yp, ceiling, and a goal body.")

    world_min = np.array([parse_vec(wall_xn.attrib["pos"], 3)[0], parse_vec(wall_yn.attrib["pos"], 3)[1], 0.0])
    world_max = np.array([
        parse_vec(wall_xp.attrib["pos"], 3)[0],
        parse_vec(wall_yp.attrib["pos"], 3)[1],
        parse_vec(ceiling.attrib["pos"], 3)[2],
    ])
    goal = parse_vec(goal_body.attrib["pos"], 3)[:2]

    static_boxes = []
    static_spheres = []
    for body in worldbody.findall("body"):
        name = body.attrib.get("name", "")
        if name == "goal":
            continue

        geom = body.find("geom")
        if geom is None:
            continue

        pos = parse_vec(body.attrib.get("pos", "0 0 0"), 3)
        euler = parse_vec(body.attrib.get("euler", "0 0 0"), 3)
        geom_type = geom.attrib.get("type")
        rgba = rgba_or_default(geom.attrib.get("rgba"), DEFAULT_SPHERE_RGBA)

        if geom_type == "box":
            size = parse_vec(geom.attrib["size"], 3)
            static_boxes.append({
                "name": name,
                "center": pos[:2],
                "half_extents": size[:2],
                "angle_deg": float(euler[2]),
                "z": float(pos[2]),
                "rgba": rgba,
            })
        elif geom_type == "sphere":
            static_spheres.append({
                "name": name,
                "center": pos[:2],
                "radius": float(parse_vec(geom.attrib["size"], 1)[0]),
                "z": float(pos[2]),
                "rgba": rgba,
            })

    return world_min, world_max, goal, static_boxes, static_spheres


def build_moving_sphere_specs(count: int, motion: str):
    if count <= 0:
        return []

    x_candidates = np.linspace(WORLD_MIN[0] + 1.0, WORLD_MAX[0] - 1.0, count)
    y_base = WORLD_MIN[1] + 0.35 * (WORLD_MAX[1] - WORLD_MIN[1])
    specs = []
    for idx, x in enumerate(x_candidates):
        center = np.array([x, y_base + 0.25 * np.sin(idx)])
        phase = idx * np.pi / max(count, 1)
        velocity = np.array([0.18 * np.cos(phase), 0.18 * np.sin(phase)])
        specs.append(
            MovingSphere(
                name=f"moving_sphere_{idx}",
                center=center,
                radius=DEFAULT_SPHERE_RADIUS,
                z=DEFAULT_SPHERE_RADIUS,
                rgba=DEFAULT_SPHERE_RGBA,
                motion=motion,
                anchor=center.copy(),
                orbit_radius=0.45,
                angular_speed=0.8 + 0.1 * idx,
                phase=phase,
                velocity=velocity,
                max_speed=0.22,
                turn_interval=1.5,
                seed=idx,
            )
        )
    return specs


def create_runtime_map(map_xml: Path, moving_spheres):
    if not moving_spheres:
        return map_xml, None

    tree = ET.parse(map_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{map_xml} is missing <worldbody>.")

    for obstacle in moving_spheres:
        body = ET.SubElement(
            worldbody,
            "body",
            {
                "name": obstacle.name,
                "mocap": "true",
                "pos": f"{obstacle.center[0]:g} {obstacle.center[1]:g} {obstacle.z:g}",
            },
        )
        ET.SubElement(
            body,
            "geom",
            {
                "type": "sphere",
                "size": f"{obstacle.radius:g}",
                "rgba": " ".join(f"{v:g}" for v in obstacle.rgba),
                "contype": "1",
                "conaffinity": "1",
                "density": "1000",
            },
        )

    tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False)
    tmp_path = Path(tmp.name)
    tree.write(tmp_path, encoding="unicode")
    tmp.close()
    return tmp_path, tmp_path


def update_obstacles(obstacles, t, dt):
    for obstacle in obstacles:
        obstacle.update(t, dt)


def sync_moving_spheres_to_sim(sim, moving_spheres, mocap_ids):
    if not moving_spheres:
        return

    mocap_pos = sim.mjx_data.mocap_pos
    mocap_quat = sim.mjx_data.mocap_quat
    for obstacle, mocap_id in zip(moving_spheres, mocap_ids):
        pos = jnp.array([obstacle.center[0], obstacle.center[1], obstacle.z])
        mocap_pos = mocap_pos.at[0, mocap_id].set(pos)
        mocap_quat = mocap_quat.at[0, mocap_id].set(IDENTITY_QUAT)
    sim.mjx_data = sim.mjx_data.replace(mocap_pos=mocap_pos, mocap_quat=mocap_quat)


def get_mocap_ids(sim, moving_spheres):
    mocap_ids = []
    for obstacle in moving_spheres:
        body_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, obstacle.name)
        if body_id < 0:
            raise ValueError(f"Moving sphere body {obstacle.name!r} not found in MuJoCo model.")
        mocap_id = sim.mj_model.body_mocapid[body_id]
        if mocap_id < 0:
            raise ValueError(f"Body {obstacle.name!r} is not a mocap body.")
        mocap_ids.append(int(mocap_id))
    return mocap_ids


def print_free_cam(sim):
    if sim.viewer is None:
        return
    human_viewer = sim.viewer._viewers.get("human")
    if human_viewer is None:
        return

    cam = human_viewer.cam
    print("FREE_CAM = {")
    print(f'    "lookat": np.array([{cam.lookat[0]:.6f}, {cam.lookat[1]:.6f}, {cam.lookat[2]:.6f}]),')
    print(f'    "distance": {float(cam.distance):.6f},')
    print(f'    "azimuth": {float(cam.azimuth):.6f},')
    print(f'    "elevation": {float(cam.elevation):.6f},')
    print("}")


def quat_to_yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def rotmat(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, -s], [s, c]])


def sphere_nearest_point(p, sphere):
    center = sphere.center if isinstance(sphere, MovingSphere) else sphere["center"]
    radius = sphere.radius if isinstance(sphere, MovingSphere) else sphere["radius"]
    diff = p - center
    dist = np.linalg.norm(diff)
    if dist < 1e-9:
        return center + np.array([radius, 0.0])
    return center + radius * diff / dist


def sphere_repulsion_2d(p, sphere, eta, Qstar):
    center = sphere.center if isinstance(sphere, MovingSphere) else sphere["center"]
    radius = sphere.radius if isinstance(sphere, MovingSphere) else sphere["radius"]
    diff = p - center
    center_dist = np.linalg.norm(diff)
    surface_dist = center_dist - radius

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
    local_corners = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy], [-hx, -hy]])
    return np.array([wall_world_point(corner, wall) for corner in local_corners])


def apf_gradient(p, theta, moving_spheres, params, safe_apf=True):
    zeta = params["zeta"]
    eta = params["eta"]
    dstar = params["dstar"]
    Qstar = params["Qstar"]
    dsafe = params["dsafe"]
    dvort = params["dvort"]
    alpha_th = params["alpha_th"]

    # Attractive potential
    diff_goal = p - GOAL
    d_goal = np.linalg.norm(diff_goal)
    if d_goal <= dstar:
        grad_att = zeta * diff_goal
    else:
        grad_att = zeta * dstar * diff_goal / (d_goal + 1e-9)

    grad_rep_total = np.zeros(2)

    # -----------------------------
    # SPHERES
    # -----------------------------
    for sphere in [*STATIC_SPHERE_OBSTACLES, *moving_spheres]:
        nearest = sphere_nearest_point(p, sphere)
        dist = max(np.linalg.norm(p - nearest), 1e-6)
        if dist > Qstar:
            continue

        grad_rep = sphere_repulsion_2d(p, sphere, eta, Qstar)

        # Regular APF (no vortex)
        if not safe_apf:
            grad_rep_total += grad_rep
            continue

        # Safe‑APF vortex rotation
        alpha = wrap(theta - np.arctan2(nearest[1] - p[1], nearest[0] - p[0]))
        direction_sign = 1 if abs(alpha) <= alpha_th else -1

        if dist <= dsafe:
            drel = 0.0
        elif dist >= dvort:
            drel = 1.0
        else:
            drel = (dist - dsafe) / (dvort - dsafe)

        gamma = 1.15 * np.pi * direction_sign * drel
        grad_rep_total += rotmat(gamma) @ grad_rep

    # -----------------------------
    # WALLS
    # -----------------------------
    for wall in STATIC_BOX_OBSTACLES:
        nearest = wall_nearest_point(p, wall)
        dist = max(np.linalg.norm(p - nearest), 1e-6)
        if dist > Qstar:
            continue

        grad_rep = wall_repulsion_2d(p, wall, eta, Qstar)

        # Regular APF
        if not safe_apf:
            grad_rep_total += grad_rep
            continue

        # Safe‑APF vortex
        alpha = wrap(theta - np.arctan2(nearest[1] - p[1], nearest[0] - p[0]))
        direction_sign = 1 if abs(alpha) <= alpha_th else -1

        if dist <= dsafe:
            drel = 0.0
        elif dist >= dvort:
            drel = 1.0
        else:
            drel = (dist - dsafe) / (dvort - dsafe)

        gamma = np.pi * direction_sign * drel
        grad_rep_total += rotmat(gamma) @ grad_rep

    return grad_att + grad_rep_total


def rollout_preview(start, theta, moving_spheres, params, safe_apf=True,
                    steps=PREVIEW_STEPS, step_gain=STEP_GAIN):
    path = [start.copy()]
    p = start.copy()
    heading = theta

    for _ in range(steps):
        if np.linalg.norm(p - GOAL) < GOAL_TOL:
            break

        # Pass safe_apf into APF
        g = apf_gradient(p, heading, moving_spheres, params, safe_apf=safe_apf)

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


def draw_scene(sim, preview_path, start_pos):
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


def plot_results(traj, obstacle_traces, moving_spheres, plot_path=None, title=None):
    import matplotlib.pyplot as plt

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

    for sphere in STATIC_SPHERE_OBSTACLES:
        circle = plt.Circle(sphere["center"], sphere["radius"], color="r", alpha=0.35)
        plt.gca().add_patch(circle)

    for trace, obstacle in zip(obstacle_traces, moving_spheres):
        trace = np.array(trace)
        if len(trace) > 1:
            plt.plot(trace[:, 0], trace[:, 1], "r--", alpha=0.35)
        circle = plt.Circle(obstacle.center, obstacle.radius, color="r", alpha=0.35)
        plt.gca().add_patch(circle)

    for wall in STATIC_BOX_OBSTACLES:
        polygon = wall_polygon(wall)
        plt.plot(polygon[:, 0], polygon[:, 1], "r-", alpha=0.7)

    if len(traj) > 0:
        plt.plot(traj[:, 0], traj[:, 1], "b-", label="Trajectory")
        plt.scatter(traj[0, 0], traj[0, 1], c="w", edgecolors="k", s=50, label="Start")
    plt.scatter(GOAL[0], GOAL[1], c="g", marker="x", s=80, label="Goal")
    plt.legend()
    plt.title(title or "RACE")
    if plot_path is not None:
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {plot_path}")
    plt.show()


def main():
    global GOAL, WORLD_MIN, WORLD_MAX, STATIC_BOX_OBSTACLES, STATIC_SPHERE_OBSTACLES

    args = parse_args()
    title_tokens = [tok for tok in sys.argv[1:] if tok not in {"--save-video", "--no-vis"}]
    plot_title = f"safeapf.py {' '.join(shlex.quote(tok) for tok in title_tokens)}".strip() 
    map_xml = args.map_xml.resolve()
    if not map_xml.exists():
        raise FileNotFoundError(f"Map XML not found: {map_xml}")
    if args.moving_spheres < 0:
        raise ValueError("--moving-spheres must be non-negative.")
    if args.timeout <= 0.0:
        raise ValueError("--timeout must be positive.")
    WORLD_MIN, WORLD_MAX, GOAL, STATIC_BOX_OBSTACLES, STATIC_SPHERE_OBSTACLES = parse_map(map_xml)
    moving_spheres = build_moving_sphere_specs(args.moving_spheres, args.motion)
    runtime_xml, temp_xml = create_runtime_map(map_xml, moving_spheres)
    sim = Sim(control=Control.state, xml_path=runtime_xml, device="cpu")
    use_box_collision(sim, enable=True)
    sim.reset()

    mocap_ids = get_mocap_ids(sim, moving_spheres)
    sync_moving_spheres_to_sim(sim, moving_spheres, mocap_ids)

    params = dict(zeta=1.1547, eta=0.09, dstar=0.3, Qstar=0.6, dsafe=0.15, dvort=0.4, alpha_th=np.deg2rad(12))

    fps = CAPTURE_FPS
    sim_steps = int(args.timeout * CTRL_FREQ)
    traj = []
    obstacle_traces = [[] for _ in moving_spheres]
    cmd = np.zeros((sim.n_worlds, sim.n_drones, 13))
    start_pos = np.array(sim.data.states.pos[0, 0, :2])
    video_frames = []
    output_stem = None
    show_window = not args.no_vis

    print(f"Running Safe-APF with map {map_xml.name}...")

    try:
        for step in range(sim_steps):
            t = step / CTRL_FREQ
            update_obstacles(moving_spheres, t, 1.0 / CTRL_FREQ)
            sync_moving_spheres_to_sim(sim, moving_spheres, mocap_ids)

            states = sim.data.states
            pos = states.pos[0, 0]
            p = np.array([pos[0], pos[1]])
            theta = quat_to_yaw(states.quat[0, 0])

            traj.append(p.copy())
            for idx, obstacle in enumerate(moving_spheres):
                obstacle_traces[idx].append(obstacle.center.copy())

            if np.linalg.norm(p - GOAL) < GOAL_TOL:
                print("Reached goal.")
                break

            g = apf_gradient(p, theta, moving_spheres, params, safe_apf=args.safe_apf)
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

            if (show_window or args.save_video) and (step * fps) % sim.control_freq < fps:
                preview_path = rollout_preview(p, theta, moving_spheres, params, safe_apf=args.safe_apf)
                if show_window:
                    # Draw overlays onto the currently active (window) viewer.
                    draw_scene(sim, preview_path, start_pos)
                    sim.render(camera=CAPTURE_CAMERA, cam_config=FREE_CAM, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT)

                if args.save_video:
                    # Offscreen and window viewers keep separate marker buffers.
                    # Prime/select the rgb viewer, then redraw overlays for capture.
                    sim.render(mode="rgb_array", camera=CAPTURE_CAMERA, cam_config=FREE_CAM, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT)
                    draw_scene(sim, preview_path, start_pos)
                    frame = sim.render(mode="rgb_array", camera=CAPTURE_CAMERA, cam_config=FREE_CAM, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT)
                    if frame is not None:
                        video_frames.append(frame)
        else:
            print("Time limit reached.")
    finally:
        if args.save_video and video_frames:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
            map_str = str(args.map_xml)
            map_str = map_str.replace(".", "")
            output_stem = CAPTURE_DIR / f"safeapf_{map_str}_NumMovObs-{args.moving_spheres}_{args.motion}_{timestamp}"
            output_path = output_stem.with_suffix(".mp4")
            iio.imwrite(output_path, video_frames, fps=CAPTURE_FPS)
            print(f"Saved capture to {output_path}")
        print_free_cam(sim)
        sim.close()
        if temp_xml is not None:
            temp_xml.unlink(missing_ok=True)

    plot_path = output_stem.with_suffix(".png") if output_stem is not None else None
    plot_results(traj, obstacle_traces, moving_spheres, plot_path=plot_path, title=plot_title)


if __name__ == "__main__":
    main()
