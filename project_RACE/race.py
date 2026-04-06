import argparse
from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path
import shlex
import sys
import tempfile
import xml.etree.ElementTree as ET

os.environ.setdefault("SCIPY_ARRAY_API", "1")

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

from crazyflow.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.sim import use_box_collision
from crazyflow.sim.visualize import draw_line, draw_points


PREVIEW_STEPS = 12  # Number of rollout steps used for the preview-path visualization.
CTRL_FREQ = 100  # Main APF control-loop frequency in Hz.
Z_REF = 0.5  # Fixed commanded flight altitude.
YAW_REF = 0.0  # Fixed commanded drone yaw.
STEP_GAIN = 0.5  # Gain mapping APF gradient magnitude into commanded position step.
MAX_APF_STEP = 0.08  # Hard cap on one APF position update for stability.
GOAL_TOL = 0.05  # Goal radius at which the run is considered successful.

FREE_CAM = {
    "lookat": np.array([2.540425, 3.509272, 0.366132]),
    "distance": 7.599483,
    "azimuth": 95.094340,
    "elevation": -24.189189,
}  # Default free-camera pose for live rendering and capture.

CAPTURE_DIR = Path(__file__).resolve().parent / "captures"  # Output folder for saved videos and plots.
CAPTURE_FPS = 24  # Video capture frame rate.
CAPTURE_WIDTH = 1280  # Render width for live view and capture.
CAPTURE_HEIGHT = 720  # Render height for live view and capture.
CAPTURE_CAMERA = -1  # Camera id used for rendering; -1 means free camera.

DEFAULT_TIMEOUT = 50.0  # Default simulation time limit in seconds.
DEFAULT_MOVING_SPHERES = 0  # Default number of runtime moving-sphere obstacles.
DEFAULT_MOTION = "circle"  # Default motion model for moving spheres.
DEFAULT_SPHERE_RADIUS = 0.25  # Radius of runtime moving-sphere obstacles.
DEFAULT_SPHERE_RGBA = (1.0, 0.0, 0.0, 1.0)  # Default color of moving-sphere obstacles.
DEFAULT_BOUNDARY_RGBA = (0.5, 0.7, 0.9, 0.12)  # Default translucent color for map boundary walls.
DRONE_SAFETY_RADIUS = 0.05  # Effective drone radius used to inflate obstacles/walls in the repulsive APF.
OBSTACLE_BODY_MASS = 0.15  # Mass assigned to runtime rigid-body obstacle spheres.
OBSTACLE_POS_KP = 10.0  # Position-tracking gain for rigid-body obstacle motion control.
OBSTACLE_VEL_KD = 6.0  # Velocity damping gain for rigid-body obstacle motion control.
OBSTACLE_MAX_FORCE = 1.5  # Maximum planar control force applied to moving rigid obstacles.
ARROW_REFRESH_PERIOD = 4  # Render-frame interval between arrow overlay refreshes.
FIELD_SAMPLE_SNAP = 0.08  # Spatial quantization used for local field-sample caching.
MOVING_SAMPLE_SNAP = 0.10  # Spatial quantization used for moving-obstacle cache keys.
ARROW_RENDER_STYLE = "line"  # Force-field glyph style: "cone" or "line".

GOAL = np.array([3.0, 3.0])  # Goal position in the xy plane; overwritten from the loaded map.
WORLD_MIN = np.array([0.0, 0.0, 0.0])  # World lower bounds; overwritten from the loaded map.
WORLD_MAX = np.array([6.0, 7.0, 3.0])  # World upper bounds; overwritten from the loaded map.
STATIC_BOX_OBSTACLES = []  # Parsed static box obstacles, including boundary walls.
STATIC_SPHERE_OBSTACLES = []  # Parsed static sphere obstacles from the map.


def select_sim_device() -> str:
    return "gpu" if any(device.platform == "gpu" for device in jax.devices()) else "cpu"


def draw_arrow(sim: Sim, p1, p2, radius=0.01, rgba=None, geom_type=mujoco.mjtGeom.mjGEOM_ARROW1):
    """Draw a single connector-style arrow geom between two points.

    This stays local to race.py so we do not need to modify the crazyflow repo.
    """
    if sim.viewer is None:
        return

    rgba = np.array([1.0, 0.0, 0.0, 1.0]) if rgba is None else rgba
    geom = mujoco.MjvGeom()
    mujoco.mjv_initGeom(
        geom,
        geom_type,
        size=np.zeros(3, dtype=np.float64),
        pos=np.zeros(3, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).reshape(-1),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        geom_type,
        radius,
        np.asarray(p1, dtype=float),
        np.asarray(p2, dtype=float),
    )
    sim.viewer.viewer.add_marker(
        type=geom.type,
        pos=np.asarray(geom.pos, dtype=np.float64).reshape(3),
        size=np.asarray(geom.size, dtype=np.float64).reshape(3),
        mat=np.asarray(geom.mat, dtype=np.float64).reshape(9),
        rgba=np.asarray(geom.rgba, dtype=np.float32).reshape(4),
    )


@dataclass(frozen=True)
class BodyHandle:
    body_id: int
    qpos_adr: int
    qpos_dim: int
    qvel_adr: int
    qvel_dim: int
    mass: float


@dataclass(frozen=True)
class ArrowPrimitive:
    tail: np.ndarray
    tip: np.ndarray
    rgba: np.ndarray
    width: float
    tip_radius: float


@dataclass
class ForceOverlayCache:
    render_counter: int = 0
    last_refresh_counter: int = -ARROW_REFRESH_PERIOD
    sample_key: tuple | None = None
    sample_points: list[np.ndarray] = field(default_factory=list)
    field_arrows: list[ArrowPrimitive] = field(default_factory=list)
    drone_force_arrows: list[ArrowPrimitive] = field(default_factory=list)


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
    base_center: np.ndarray | None = None
    _rng: np.random.Generator = field(init=False, repr=False)
    _time_to_turn: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self):
        self.center = np.array(self.center, dtype=float)
        self.base_center = np.array(self.base_center if self.base_center is not None else self.center, dtype=float)
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
            return

        if self.motion == "random":
            self._time_to_turn -= dt
            if self._time_to_turn <= 0.0:
                self.velocity = self._random_velocity()
                self._time_to_turn = self.turn_interval
            return

        if self.motion != "static":
            raise ValueError(f"Unsupported obstacle motion: {self.motion}")

    def target_position(self, t: float) -> np.ndarray:
        if self.motion == "circle":
            theta = self.phase + self.angular_speed * t
            return self.anchor + self.orbit_radius * np.array([np.cos(theta), np.sin(theta)])
        if self.motion == "static":
            return self.anchor.copy()
        return self.center.copy()

    def target_velocity(self, t: float) -> np.ndarray:
        if self.motion == "circle":
            theta = self.phase + self.angular_speed * t
            tangential = np.array([-np.sin(theta), np.cos(theta)])
            return self.orbit_radius * self.angular_speed * tangential
        if self.motion == "random":
            return self.velocity.copy()
        return np.zeros(2)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Safe-APF with a static XML map and optional moving spheres.")
    parser.add_argument("map_xml", type=Path, help="XML file containing the static environment and static obstacles.")
    parser.add_argument("--moving-spheres", type=int, default=DEFAULT_MOVING_SPHERES, help="Number of moving sphere obstacles to add at runtime.")
    parser.add_argument("--motion", choices=["static", "circle", "random"], default=DEFAULT_MOTION, help="Motion model used for all moving spheres.")
    parser.add_argument("--save-video", action="store_true", help="Save an MP4 capture of the simulation.")
    parser.add_argument("--no-vis", action="store_true", help="Disable live rendering and the final matplotlib plot.")
    parser.add_argument("--draw_arrows", action="store_true", help="Draw the APF force-field and drone-force arrows.")
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
    for wall_geom in (wall_xn, wall_xp, wall_yn, wall_yp):
        wall_pos = parse_vec(wall_geom.attrib["pos"], 3)
        wall_size = parse_vec(wall_geom.attrib["size"], 3)
        static_boxes.append({
            "name": wall_geom.attrib["name"],
            "center": wall_pos[:2],
            "half_extents": wall_size[:2],
            "angle_deg": 0.0,
            "z": float(wall_pos[2]),
            "rgba": rgba_or_default(wall_geom.attrib.get("rgba"), DEFAULT_BOUNDARY_RGBA),
        })

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

    for geom_name in ("ceiling", "wall_xn", "wall_xp", "wall_yn", "wall_yp"):
        geom = worldbody.find(f"./geom[@name='{geom_name}']")
        if geom is not None:
            geom.attrib["contype"] = "1"
            geom.attrib["conaffinity"] = "1"

    for obstacle in moving_spheres:
        body = ET.SubElement(
            worldbody,
            "body",
            {
                "name": obstacle.name,
                "pos": f"{obstacle.center[0]:g} {obstacle.center[1]:g} {obstacle.z:g}",
            },
        )
        ET.SubElement(
            body,
            "joint",
            {
                "name": f"{obstacle.name}_x",
                "type": "slide",
                "axis": "1 0 0",
                "damping": "0.8",
                "limited": "false",
            },
        )
        ET.SubElement(
            body,
            "joint",
            {
                "name": f"{obstacle.name}_y",
                "type": "slide",
                "axis": "0 1 0",
                "damping": "0.8",
                "limited": "false",
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
                "mass": f"{OBSTACLE_BODY_MASS:g}",
                "friction": "0.8 0.05 0.05",
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


def joint_dims(joint_type: int) -> tuple[int, int]:
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7, 6
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4, 3
    if joint_type in (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE):
        return 1, 1
    raise ValueError(f"Unsupported MuJoCo joint type: {joint_type}")


def get_body_handle(sim, body_name: str) -> BodyHandle:
    body_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body {body_name!r} not found in MuJoCo model.")

    joint_start = int(sim.mj_model.body_jntadr[body_id])
    joint_count = int(sim.mj_model.body_jntnum[body_id])
    if joint_count <= 0:
        raise ValueError(f"Body {body_name!r} has no joints.")

    qpos_adr = int(sim.mj_model.jnt_qposadr[joint_start])
    qvel_adr = int(sim.mj_model.jnt_dofadr[joint_start])
    qpos_dim = 0
    qvel_dim = 0
    for joint_id in range(joint_start, joint_start + joint_count):
        add_qpos, add_qvel = joint_dims(int(sim.mj_model.jnt_type[joint_id]))
        qpos_dim += add_qpos
        qvel_dim += add_qvel

    return BodyHandle(
        body_id=body_id,
        qpos_adr=qpos_adr,
        qpos_dim=qpos_dim,
        qvel_adr=qvel_adr,
        qvel_dim=qvel_dim,
        mass=float(sim.mj_model.body_mass[body_id]),
    )


def get_drone_handles(sim):
    return [get_body_handle(sim, f"drone:{idx}") for idx in range(sim.n_drones)]


def get_obstacle_handles(sim, moving_spheres):
    return {obstacle.name: get_body_handle(sim, obstacle.name) for obstacle in moving_spheres}


def sync_drones_to_mj_data(sim, drone_handles):
    if sim.n_worlds != 1:
        raise ValueError("race.py rigid-body obstacle sync currently supports n_worlds=1 only.")

    for drone_idx, handle in enumerate(drone_handles):
        pos = np.array(sim.data.states.pos[0, drone_idx], dtype=float)
        quat = np.roll(np.array(sim.data.states.quat[0, drone_idx], dtype=float), 1)
        vel = np.array(sim.data.states.vel[0, drone_idx], dtype=float)
        ang_vel = np.array(sim.data.states.ang_vel[0, drone_idx], dtype=float)

        sim.mj_data.qpos[handle.qpos_adr : handle.qpos_adr + handle.qpos_dim] = np.concatenate([pos, quat])
        sim.mj_data.qvel[handle.qvel_adr : handle.qvel_adr + handle.qvel_dim] = np.concatenate([vel, ang_vel])


def enforce_world_bounds(sim, obstacle, handle):
    qpos = sim.mj_data.qpos[handle.qpos_adr : handle.qpos_adr + 2].copy()
    qvel = sim.mj_data.qvel[handle.qvel_adr : handle.qvel_adr + 2].copy()
    world_pos = obstacle.base_center + qpos

    for axis in range(2):
        lower = WORLD_MIN[axis] + obstacle.radius
        upper = WORLD_MAX[axis] - obstacle.radius
        if world_pos[axis] < lower:
            world_pos[axis] = lower
            qpos[axis] = lower - obstacle.base_center[axis]
            qvel[axis] = abs(qvel[axis])
            if obstacle.motion == "random":
                obstacle.velocity[axis] = abs(obstacle.velocity[axis])
        elif world_pos[axis] > upper:
            world_pos[axis] = upper
            qpos[axis] = upper - obstacle.base_center[axis]
            qvel[axis] = -abs(qvel[axis])
            if obstacle.motion == "random":
                obstacle.velocity[axis] = -abs(obstacle.velocity[axis])

    sim.mj_data.qpos[handle.qpos_adr : handle.qpos_adr + 2] = qpos
    sim.mj_data.qvel[handle.qvel_adr : handle.qvel_adr + 2] = qvel


def step_rigid_body_obstacles(sim, moving_spheres, obstacle_handles, drone_handles, t, dt):
    if not moving_spheres:
        return

    substeps = max(1, sim.freq // CTRL_FREQ)
    sync_drones_to_mj_data(sim, drone_handles)
    sim.mj_data.qfrc_applied[:] = 0.0

    target_t = t + dt
    for obstacle in moving_spheres:
        handle = obstacle_handles[obstacle.name]
        current_offset = sim.mj_data.qpos[handle.qpos_adr : handle.qpos_adr + 2]
        current_pos = obstacle.base_center + current_offset
        current_vel = sim.mj_data.qvel[handle.qvel_adr : handle.qvel_adr + 2]
        target_pos = obstacle.target_position(target_t)
        target_vel = obstacle.target_velocity(target_t)

        pos_err = target_pos - current_pos
        vel_err = target_vel - current_vel
        force = handle.mass * (OBSTACLE_POS_KP * pos_err + OBSTACLE_VEL_KD * vel_err)
        force = np.clip(force, -OBSTACLE_MAX_FORCE, OBSTACLE_MAX_FORCE)
        sim.mj_data.qfrc_applied[handle.qvel_adr : handle.qvel_adr + 2] = force

    mujoco.mj_step(sim.mj_model, sim.mj_data, nstep=substeps)
    for obstacle in moving_spheres:
        enforce_world_bounds(sim, obstacle, obstacle_handles[obstacle.name])
    mujoco.mj_forward(sim.mj_model, sim.mj_data)

    sim.mj_data.qfrc_applied[:] = 0.0


def refresh_moving_spheres_from_mj_data(sim, moving_spheres, obstacle_handles):
    for obstacle in moving_spheres:
        handle = obstacle_handles[obstacle.name]
        body_pos = np.array(sim.mj_data.xpos[handle.body_id], dtype=float)
        obstacle.center = body_pos[:2]


def ensure_render_target(sim, mode="human", camera=-1, cam_config=None, width=1920, height=1080):
    if sim.viewer is None:
        if isinstance(camera, str):
            cam_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            assert cam_id > -1, f"Camera name '{camera}' not found in the model."
        elif isinstance(camera, int):
            cam_id = camera
            assert cam_id >= -1, f"camera id must be >=-1, was {cam_id}"
        else:
            raise TypeError("camera argument must be integer or string")

        sim.mj_model.vis.global_.offwidth = width
        sim.mj_model.vis.global_.offheight = height
        sim.viewer = MujocoRenderer(
            sim.mj_model,
            sim.mj_data,
            max_geom=sim.max_visual_geom,
            default_cam_config=cam_config,
            height=height,
            width=width,
            camera_id=cam_id,
        )
    else:
        cam_id = sim.viewer.camera_id if isinstance(camera, int) else camera

    viewer = sim.viewer._get_viewer(mode)
    if mode == "human" and isinstance(camera, int) and camera > -1:
        viewer.cam.fixedcamid = camera
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    return viewer


def render_live_scene(sim, mode="human", camera=-1, cam_config=None, width=1920, height=1080):
    ensure_render_target(sim, mode=mode, camera=camera, cam_config=cam_config, width=width, height=height)
    mujoco.mj_forward(sim.mj_model, sim.mj_data)
    return sim.viewer.render(mode)


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
    surface_dist = center_dist - radius - DRONE_SAFETY_RADIUS

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
    raw_dist = np.linalg.norm(local_diff)
    dist = raw_dist - DRONE_SAFETY_RADIUS

    if dist < 1e-6:
        if raw_dist < 1e-6:
            pen = wall["half_extents"] - np.abs(local_p)
            axis = int(np.argmin(pen))
            local_direction = np.zeros(2)
            local_direction[axis] = 1.0 if local_p[axis] >= 0.0 else -1.0
        else:
            local_direction = local_diff / (raw_dist + 1e-9)
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


def project_point_out_of_sphere(p, sphere, clearance):
    center = sphere.center if isinstance(sphere, MovingSphere) else sphere["center"]
    radius = sphere.radius if isinstance(sphere, MovingSphere) else sphere["radius"]
    diff = p - center
    dist = np.linalg.norm(diff)
    min_dist = radius + clearance
    if dist >= min_dist:
        return p, False
    if dist < 1e-9:
        direction = np.array([1.0, 0.0])
    else:
        direction = diff / dist
    return center + direction * min_dist, True


def project_point_out_of_wall(p, wall, clearance):
    local_p = wall_local_point(p, wall)
    local_closest = np.clip(local_p, -wall["half_extents"], wall["half_extents"])
    local_diff = local_p - local_closest
    raw_dist = np.linalg.norm(local_diff)

    if raw_dist >= clearance:
        return p, False

    if raw_dist < 1e-9:
        pen = wall["half_extents"] - np.abs(local_p)
        axis = int(np.argmin(pen))
        local_direction = np.zeros(2)
        local_direction[axis] = 1.0 if local_p[axis] >= 0.0 else -1.0
    else:
        local_direction = local_diff / raw_dist

    local_target = local_closest + local_direction * clearance
    return wall_world_point(local_target, wall), True


def project_point_to_free_space(p, moving_spheres, clearance=DRONE_SAFETY_RADIUS):
    projected = np.array(p, dtype=float)
    projected[0] = np.clip(projected[0], WORLD_MIN[0] + clearance, WORLD_MAX[0] - clearance)
    projected[1] = np.clip(projected[1], WORLD_MIN[1] + clearance, WORLD_MAX[1] - clearance)

    for _ in range(3):
        moved = False
        for sphere in [*STATIC_SPHERE_OBSTACLES, *moving_spheres]:
            projected, hit = project_point_out_of_sphere(projected, sphere, clearance)
            moved = moved or hit
        for wall in STATIC_BOX_OBSTACLES:
            projected, hit = project_point_out_of_wall(projected, wall, clearance)
            moved = moved or hit
        projected[0] = np.clip(projected[0], WORLD_MIN[0] + clearance, WORLD_MAX[0] - clearance)
        projected[1] = np.clip(projected[1], WORLD_MIN[1] + clearance, WORLD_MAX[1] - clearance)
        if not moved:
            break

    return projected


def enforce_drone_clearance(sim, moving_spheres, drone_idx=0):
    pos = np.array(sim.data.states.pos[0, drone_idx], dtype=float)
    projected_xy = project_point_to_free_space(pos[:2], moving_spheres)
    if np.linalg.norm(projected_xy - pos[:2]) < 1e-9:
        return

    pos_array = sim.data.states.pos.at[0, drone_idx, :2].set(jnp.array(projected_xy))
    current_xy_vel = np.array(sim.data.states.vel[0, drone_idx, :2], dtype=float)
    vel_array = sim.data.states.vel.at[0, drone_idx, :2].set(jnp.array(-current_xy_vel))
    sim.data = sim.data.replace(states=sim.data.states.replace(pos=pos_array, vel=vel_array))


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
        dist = max(np.linalg.norm(p - nearest) - DRONE_SAFETY_RADIUS, 1e-6)
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
        dist = max(np.linalg.norm(p - nearest) - DRONE_SAFETY_RADIUS, 1e-6)
        if dist > Qstar:
            continue

        grad_rep = wall_repulsion_2d(p, wall, eta, Qstar)

        # Keep the named environment boundary walls as pure push-away barriers
        # even in Safe-APF mode. Interior box obstacles can still use vortex
        # flow to encourage navigation around them.
        if (not safe_apf) or str(wall.get("name", "")).startswith("wall_"):
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


def apf_step_delta(grad, step_gain=STEP_GAIN, max_step=MAX_APF_STEP):
    """Convert APF gradient into a bounded position step without normalizing it.

    This preserves force magnitude information while still capping the commanded
    step for stability near strong repulsive fields.
    """
    delta = -step_gain * np.array(grad, dtype=float)
    delta_norm = np.linalg.norm(delta)
    if delta_norm < 1e-9:
        return np.zeros(2)
    if delta_norm > max_step:
        delta *= max_step / delta_norm
    return delta


def rollout_preview(start, theta, moving_spheres, params, safe_apf=True,
                    steps=PREVIEW_STEPS, step_gain=STEP_GAIN):
    path = [start.copy()]
    p = start.copy()
    preview_theta = float(theta)

    for _ in range(steps):
        if np.linalg.norm(p - GOAL) < GOAL_TOL:
            break

        # Mirror the actual control loop: use the current drone yaw for Safe-APF,
        # then step using the capped APF magnitude and project to free space.
        g = apf_gradient(p, preview_theta, moving_spheres, params, safe_apf=safe_apf)
        delta = apf_step_delta(g, step_gain=step_gain)
        if np.linalg.norm(delta) < 1e-9:
            break

        p = p + delta
        p = project_point_to_free_space(p, moving_spheres)
        path.append(p.copy())

    return np.array(path)


def quantize_xy(point, step):
    return np.round(np.array(point, dtype=float) / step) * step


def make_local_field_sample_key(drone_pos_2d, moving_spheres, focus_radius=1.45):
    drone_anchor = tuple(np.round(np.array(drone_pos_2d) / FIELD_SAMPLE_SNAP).astype(int))
    nearby_spheres = []
    for idx, sphere in enumerate(moving_spheres):
        if np.linalg.norm(sphere.center - drone_pos_2d) <= focus_radius + sphere.radius:
            sphere_anchor = tuple(np.round(sphere.center / MOVING_SAMPLE_SNAP).astype(int))
            nearby_spheres.append((idx, *sphere_anchor))
    return (drone_anchor, tuple(nearby_spheres))


def build_local_field_samples(drone_pos_2d, moving_spheres):
    samples = []
    sample_keys = set()
    drone_anchor = quantize_xy(drone_pos_2d, FIELD_SAMPLE_SNAP)
    windows = [
        (0.26, 11),  # very dense near the drone
        (0.48, 7),   # medium ring
        (0.68, 5),   # sparse outer ring
    ]
    previous_radius = -np.inf
    boundary_margin = 0.12

    def maybe_add_sample(p):
        if not (WORLD_MIN[0] + boundary_margin <= p[0] <= WORLD_MAX[0] - boundary_margin):
            return
        if not (WORLD_MIN[1] + boundary_margin <= p[1] <= WORLD_MAX[1] - boundary_margin):
            return
        key = tuple(np.round(p, 3))
        if key in sample_keys:
            return
        sample_keys.add(key)
        samples.append(np.array(p, dtype=float))

    for radius, grid_n in windows:
        xs = np.linspace(drone_anchor[0] - radius, drone_anchor[0] + radius, grid_n)
        ys = np.linspace(drone_anchor[1] - radius, drone_anchor[1] + radius, grid_n)
        for x in xs:
            for y in ys:
                cheb_dist = max(abs(x - drone_anchor[0]), abs(y - drone_anchor[1]))
                if cheb_dist <= previous_radius + 1e-9:
                    continue
                maybe_add_sample((x, y))
        previous_radius = radius

    # Add small dense patches around nearby obstacle boundaries so the field is
    # more informative where the APF changes quickly.
    focus_radius = 0.82
    patch_offsets = np.array([-0.09, -0.045, 0.0, 0.045, 0.09])
    for sphere in [*STATIC_SPHERE_OBSTACLES, *moving_spheres]:
        anchor = quantize_xy(sphere_nearest_point(drone_pos_2d, sphere), FIELD_SAMPLE_SNAP)
        if np.linalg.norm(anchor - drone_pos_2d) > focus_radius:
            continue
        for dx in patch_offsets:
            for dy in patch_offsets:
                maybe_add_sample(anchor + np.array([dx, dy]))

    for wall in STATIC_BOX_OBSTACLES:
        anchor = quantize_xy(wall_nearest_point(drone_pos_2d, wall), FIELD_SAMPLE_SNAP)
        if np.linalg.norm(anchor - drone_pos_2d) > focus_radius:
            continue
        tangent = rotmat(np.deg2rad(wall["angle_deg"])) @ np.array([1.0, 0.0])
        normal = rotmat(np.deg2rad(wall["angle_deg"])) @ np.array([0.0, 1.0])
        for dt in patch_offsets:
            for dn in patch_offsets:
                maybe_add_sample(anchor + 0.9 * dt * tangent + 0.45 * dn * normal)

    return samples


def get_local_field_samples(drone_pos_2d, moving_spheres, overlay_cache=None):
    if overlay_cache is None:
        return build_local_field_samples(drone_pos_2d, moving_spheres)

    sample_key = make_local_field_sample_key(drone_pos_2d, moving_spheres)
    if overlay_cache.sample_key != sample_key or not overlay_cache.sample_points:
        overlay_cache.sample_key = sample_key
        overlay_cache.sample_points = build_local_field_samples(drone_pos_2d, moving_spheres)
    return overlay_cache.sample_points


def force_field_rgba(force_norm, params):
    scale = params["zeta"] * 3.0
    intensity = float(np.clip(np.log1p(force_norm) / np.log1p(scale), 0.0, 1.0))
    return np.array([
        0.15 + 0.85 * intensity,
        0.35 + 0.45 * intensity,
        1.00 - 0.85 * intensity,
        0.30 + 0.40 * intensity,
    ])


def build_force_field_arrows(drone_pos_2d, theta, moving_spheres, params, drone_z=None, safe_apf=True, overlay_cache=None):
    ARROW_LEN = 0.05
    ARROW_W = 0.40
    TIP_R = 0.0035
    z = float(drone_z) if drone_z is not None else Z_REF
    arrows = []

    for p in get_local_field_samples(drone_pos_2d, moving_spheres, overlay_cache=overlay_cache):
        g = apf_gradient(p, theta, moving_spheres, params, safe_apf=safe_apf)
        force_norm = np.linalg.norm(g)
        if force_norm < 1e-6:
            continue

        direction = -g / force_norm
        tail = np.array([p[0], p[1], z])
        tip = np.array([
            p[0] + direction[0] * ARROW_LEN,
            p[1] + direction[1] * ARROW_LEN,
            z,
        ])
        rgba = force_field_rgba(force_norm, params)
        arrows.append(ArrowPrimitive(tail=tail, tip=tip, rgba=rgba, width=ARROW_W, tip_radius=TIP_R))

    return arrows


def build_drone_force_arrows(drone_pos_2d, theta, moving_spheres, params, safe_apf=True):
    """Draw the three APF force components anchored at the drone's live position.

    ● Green  — attractive force  (pulls toward goal)
    ● Red    — total repulsive contribution
    ● White  — resultant / net force
    All arrows are direction-normalised and scaled to SCALE metres.
    """
    SCALE = 0.30   # visual length (metres)
    WIDTH = 2.5    # line width in pixels
    TIP_R = 0.020  # tip sphere radius (metres)

    zeta  = params["zeta"]
    dstar = params["dstar"]
    eta   = params["eta"]
    Qstar = params["Qstar"]

    p = drone_pos_2d

    # Attractive gradient 
    diff_goal = p - GOAL
    d_goal = np.linalg.norm(diff_goal)
    grad_att = zeta * diff_goal if d_goal <= dstar else zeta * dstar * diff_goal / (d_goal + 1e-9)

    # Match the controller: the total field comes from apf_gradient, and the
    # displayed repulsive contribution is the remainder after subtracting attraction.
    grad_total = apf_gradient(p, theta, moving_spheres, params, safe_apf=safe_apf)
    grad_rep = grad_total - grad_att

    origin = np.array([p[0], p[1], Z_REF])
    vectors = [
        (-grad_att,   np.array([0.15, 1.00, 0.25, 1.0])),   # green  – attractive
        (-grad_rep,   np.array([1.00, 0.20, 0.15, 1.0])),   # red    – repulsive
        (-grad_total, np.array([1.00, 1.00, 1.00, 1.0])),   # white  – resultant
    ]
    arrows = []
    for force_2d, rgba in vectors:
        norm = np.linalg.norm(force_2d)
        if norm < 1e-6:
            continue
        direction = force_2d / norm
        tip = np.array([origin[0] + direction[0] * SCALE,
                        origin[1] + direction[1] * SCALE,
                        Z_REF])
        arrows.append(ArrowPrimitive(tail=origin.copy(), tip=tip, rgba=rgba, width=WIDTH, tip_radius=TIP_R))

    return arrows


def draw_arrow_primitives(sim, arrows):
    if sim.viewer is None:
        return
    for arrow in arrows:
        if ARROW_RENDER_STYLE == "line":
            draw_line(
                sim,
                np.array([arrow.tail, arrow.tip]),
                rgba=arrow.rgba,
                start_size=arrow.width,
                end_size=arrow.width,
            )
        else:
            draw_arrow(sim, arrow.tail, arrow.tip, radius=arrow.tip_radius, rgba=arrow.rgba)


def refresh_force_overlay_cache(overlay_cache, drone_pos_2d, theta, moving_spheres, params, drone_z=None, safe_apf=True):
    if overlay_cache is None:
        return

    refresh_needed = not overlay_cache.field_arrows
    refresh_needed = refresh_needed or (overlay_cache.render_counter - overlay_cache.last_refresh_counter >= ARROW_REFRESH_PERIOD)
    if not refresh_needed:
        return

    overlay_cache.field_arrows = build_force_field_arrows(
        drone_pos_2d,
        theta,
        moving_spheres,
        params,
        drone_z=drone_z,
        safe_apf=safe_apf,
        overlay_cache=overlay_cache,
    )
    overlay_cache.drone_force_arrows = build_drone_force_arrows(
        drone_pos_2d,
        theta,
        moving_spheres,
        params,
        safe_apf=safe_apf,
    )
    overlay_cache.last_refresh_counter = overlay_cache.render_counter


def draw_force_field(sim, drone_pos_2d, theta, moving_spheres, params, drone_z=None, safe_apf=True, overlay_cache=None):
    """Draw a local adaptive vector-field diagnostic around the drone."""
    if overlay_cache is None:
        arrows = build_force_field_arrows(
            drone_pos_2d,
            theta,
            moving_spheres,
            params,
            drone_z=drone_z,
            safe_apf=safe_apf,
            overlay_cache=None,
        )
        draw_arrow_primitives(sim, arrows)
        return

    refresh_force_overlay_cache(
        overlay_cache,
        drone_pos_2d,
        theta,
        moving_spheres,
        params,
        drone_z=drone_z,
        safe_apf=safe_apf,
    )
    draw_arrow_primitives(sim, overlay_cache.field_arrows)


def draw_drone_forces(sim, drone_pos_2d, theta, moving_spheres, params, safe_apf=True, overlay_cache=None):
    if overlay_cache is None:
        arrows = build_drone_force_arrows(drone_pos_2d, theta, moving_spheres, params, safe_apf=safe_apf)
        draw_arrow_primitives(sim, arrows)
        return

    refresh_force_overlay_cache(
        overlay_cache,
        drone_pos_2d,
        theta,
        moving_spheres,
        params,
        drone_z=Z_REF,
        safe_apf=safe_apf,
    )
    draw_arrow_primitives(sim, overlay_cache.drone_force_arrows)

# Added draw_force_field and draw_drone_forces functions
# Added additional arguments to draw_scene


def draw_scene(sim, preview_path, start_pos, drone_pos_2d, drone_z, theta, moving_spheres, params, draw_arrows=False, safe_apf=True, overlay_cache=None):
    if (not draw_arrows) and len(preview_path) >= 2:
        draw_line(
            sim,
            np.column_stack([preview_path, np.full(len(preview_path), Z_REF)]),
            rgba=np.array([0.2, 0.4, 1.0, 0.85]),
            start_size=1.5,
            end_size=1.5,
        )

    draw_points(sim, np.array([[GOAL[0], GOAL[1], Z_REF]]), rgba=np.array([0.0, 1.0, 0.2, 1.0]), size=0.08)
    draw_points(sim, np.array([[start_pos[0], start_pos[1], Z_REF]]), rgba=np.array([1.0, 1.0, 1.0, 0.8]), size=0.05)

    if draw_arrows:
        draw_force_field(sim, drone_pos_2d, theta, moving_spheres, params, drone_z=drone_z, safe_apf=safe_apf, overlay_cache=overlay_cache)
        # draw_drone_forces(sim, drone_pos_2d, theta, moving_spheres, params, safe_apf=safe_apf, overlay_cache=overlay_cache)

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
    sim_device = select_sim_device()
    sim = Sim(control=Control.state, xml_path=runtime_xml, device=sim_device)
    use_box_collision(sim, enable=True)
    sim.reset()

    drone_handles = get_drone_handles(sim)
    obstacle_handles = get_obstacle_handles(sim, moving_spheres)
    sync_drones_to_mj_data(sim, drone_handles)
    mujoco.mj_forward(sim.mj_model, sim.mj_data)
    refresh_moving_spheres_from_mj_data(sim, moving_spheres, obstacle_handles)

    params = dict(
        zeta=6.0,                 # Attractive gain toward the goal.
        eta=0.09,                  # Repulsive gain for walls and obstacles.
        dstar=0.1,                # Goal distance where attraction saturates.
        Qstar=0.1,                # Obstacle influence distance for repulsion.
        dsafe=0.05,               # Inner Safe-APF radius where vortexing is disabled.
        dvort=0.4,                # Outer Safe-APF radius where vortexing fades out.
        alpha_th=np.deg2rad(12),  # Heading threshold used to choose vortex direction.
    )

    fps = CAPTURE_FPS
    sim_steps = int(args.timeout * CTRL_FREQ)
    traj = []
    obstacle_traces = [[] for _ in moving_spheres]
    cmd = np.zeros((sim.n_worlds, sim.n_drones, 13))
    start_pos = np.array(sim.data.states.pos[0, 0, :2])
    video_frames = []
    output_stem = None
    show_window = not args.no_vis
    overlay_cache = ForceOverlayCache()

    print(f"Running Safe-APF with map {map_xml.name} on {sim_device}...")

    try:
        for step in range(sim_steps):
            t = step / CTRL_FREQ
            update_obstacles(moving_spheres, t, 1.0 / CTRL_FREQ)
            step_rigid_body_obstacles(sim, moving_spheres, obstacle_handles, drone_handles, t, 1.0 / CTRL_FREQ)
            refresh_moving_spheres_from_mj_data(sim, moving_spheres, obstacle_handles)

            states = sim.data.states
            pos = states.pos[0, 0]
            p = np.array([pos[0], pos[1]])
            # Added drone z-component
            drone_z = float(pos[2])
            theta = quat_to_yaw(states.quat[0, 0])

            traj.append(p.copy())
            for idx, obstacle in enumerate(moving_spheres):
                obstacle_traces[idx].append(obstacle.center.copy())

            if np.linalg.norm(p - GOAL) < GOAL_TOL:
                print("Reached goal.")
                break

            g = apf_gradient(p, theta, moving_spheres, params, safe_apf=args.safe_apf)
            p_des = p + apf_step_delta(g)
            p_des = project_point_to_free_space(p_des, moving_spheres)

            cmd[..., 0] = p_des[0]
            cmd[..., 1] = p_des[1]
            cmd[..., 2] = Z_REF
            cmd[..., 3] = YAW_REF
            cmd[..., 6] = 0.0

            sim.state_control(cmd)
            sim.step(sim.freq // sim.control_freq)
            enforce_drone_clearance(sim, moving_spheres)

            if (show_window or args.save_video) and (step * fps) % sim.control_freq < fps:
                sync_drones_to_mj_data(sim, drone_handles)
                if args.draw_arrows:
                    overlay_cache.render_counter += 1
                preview_path = np.empty((0, 2)) if args.draw_arrows else rollout_preview(
                    p, theta, moving_spheres, params, safe_apf=args.safe_apf
                )
                if show_window:
                    ensure_render_target(sim, mode="human", camera=CAPTURE_CAMERA, cam_config=FREE_CAM, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT)
                    draw_scene(sim, preview_path, start_pos, p, drone_z, theta, moving_spheres, params, draw_arrows=args.draw_arrows, safe_apf=args.safe_apf, overlay_cache=overlay_cache)
                    render_live_scene(sim, mode="human", camera=CAPTURE_CAMERA, cam_config=FREE_CAM, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT)

                if args.save_video:
                    ensure_render_target(sim, mode="rgb_array", camera=CAPTURE_CAMERA, cam_config=FREE_CAM, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT)
                    draw_scene(sim, preview_path, start_pos, p, drone_z, theta, moving_spheres, params, draw_arrows=args.draw_arrows, safe_apf=args.safe_apf, overlay_cache=overlay_cache)
                    frame = render_live_scene(sim, mode="rgb_array", camera=CAPTURE_CAMERA, cam_config=FREE_CAM, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT)
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
