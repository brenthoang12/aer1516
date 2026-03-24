import numpy as np
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt

from crazyflow.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.sim import use_box_collision


GOAL = np.array([3.0, 3.0])
WORLD_MIN = np.array([0.0, 0.0, 0.0])
WORLD_MAX = np.array([6.0, 7.0, 3.0])
SPHERE_OBSTACLES = [
    # {
    #     "name": "sphere1",
    #     "center": np.array([1.0, 1.0]),
    #     "radius": 0.25,
    #     "z": 0.25,
    #     "rgba": (1.0, 0.0, 0.0, 1.0),
    # },
    {
        "name": "sphere2",
        "center": np.array([3.0, 2.0]),
        "radius": 0.25,
        "z": 0.25,
        "rgba": (1.0, 0.0, 0.0, 1.0),
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
    diff = p - sphere["center"]
    dist = np.linalg.norm(diff)
    if dist < 1e-9:
        return sphere["center"] + np.array([sphere["radius"], 0.0])
    return sphere["center"] + sphere["radius"] * diff / dist


def sphere_repulsion_2d(p, sphere, eta, Qstar):
    diff = p - sphere["center"]
    center_dist = np.linalg.norm(diff)
    surface_dist = center_dist - sphere["radius"]

    if surface_dist < 1e-6:
        if center_dist < 1e-9:
            direction = np.array([1.0, 0.0])
        else:
            direction = diff / center_dist
        return eta * (1.0 / 1e-6 - 1.0 / Qstar) * (1.0 / (1e-6**2)) * direction

    if surface_dist > Qstar:
        return np.zeros(2)

    direction = diff / (center_dist + 1e-9)
    return eta * (1.0 / Qstar - 1.0 / surface_dist) * (1.0 / (surface_dist**2)) * direction


def wall_local_point(p, wall):
    angle = np.deg2rad(wall["angle_deg"])
    R = rotmat(angle)
    return R.T @ (p - wall["center"])


def wall_world_point(local_p, wall):
    angle = np.deg2rad(wall["angle_deg"])
    R = rotmat(angle)
    return wall["center"] + R @ local_p


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
        world_direction = wall_world_point(local_direction, {**wall, "center": np.zeros(2)})
        world_direction /= np.linalg.norm(world_direction) + 1e-9
        return eta * (1.0 / 1e-6 - 1.0 / Qstar) * (1.0 / (1e-6**2)) * world_direction

    if dist > Qstar:
        return np.zeros(2)

    local_grad = eta * (1.0 / Qstar - 1.0 / dist) * (1.0 / (dist**2)) * local_diff
    angle = np.deg2rad(wall["angle_deg"])
    return rotmat(angle) @ local_grad


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
        '        <!-- Sphere obstacles with collision -->',
    ]

    for sphere in SPHERE_OBSTACLES:
        cx, cy = sphere["center"]
        parts += [
            f'        <body name="{sphere["name"]}" pos="{cx:g} {cy:g} {sphere["z"]:g}">',
            f'            <geom type="sphere" size="{sphere["radius"]:g}"',
            f'                  rgba="{rgba_str(sphere["rgba"])}"',
            '                  contype="1" conaffinity="1"',
            '                  density="1000"/>',
            '        </body>',
            '',
        ]

    parts += ['        <!-- U-shape walls with collision -->']
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


def apf_gradient(p, theta, params):
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

    for sphere in SPHERE_OBSTACLES:
        nearest = sphere_nearest_point(p, sphere)
        dist = max(np.linalg.norm(p - nearest), 1e-6)
        if dist > Qstar:
            continue

        grad_rep = sphere_repulsion_2d(p, sphere, eta, Qstar)
        alpha = theta - np.arctan2(nearest[1] - p[1], nearest[0] - p[0])
        alpha = wrap(alpha)
        D_alpha = 1 if abs(alpha) <= alpha_th else -1

        if dist <= dsafe:
            drel = 0.0
        elif dist >= dvort:
            drel = 1.0
        else:
            drel = (dist - dsafe) / (dvort - dsafe)
        drel = np.clip(drel, 0.0, 1.0)

        gamma = np.pi * D_alpha * drel
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
        D_alpha = 1 if abs(alpha) <= alpha_th else -1

        if dist <= dsafe:
            drel = 0.0
        elif dist >= dvort:
            drel = 1.0
        else:
            drel = (dist - dsafe) / (dvort - dsafe)
        drel = np.clip(drel, 0.0, 1.0)

        gamma = np.pi * D_alpha * drel
        grad_rep_total += rotmat(gamma) @ grad_rep

    return grad_att + grad_rep_total


def main():
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        tmp.write(build_scene_xml().encode())
        tmp.flush()
        xml_path = Path(tmp.name)

    sim = Sim(control=Control.state, xml_path=xml_path)
    use_box_collision(sim, enable=True)
    sim.reset()

    fps = 60
    z_ref = 0.5
    yaw_ref = 0.0
    params = dict(
        zeta=1.1547,
        eta=0.0732,
        dstar=0.3,
        Qstar=0.5,
        dsafe=0.2,
        dvort=0.35,
        alpha_th=np.deg2rad(5),
    )

    step_gain = 0.15
    traj = []

    print("Running full APF + Safe-APF with shared obstacle definitions...")

    for step in range(5000):
        states = sim.data.states
        pos = states.pos[0, 0]
        p = np.array([pos[0], pos[1]])
        theta = quat_to_yaw(states.quat[0, 0])

        traj.append(p.copy())

        if np.linalg.norm(p - GOAL) < 0.05:
            print("Reached goal.")
            break

        g = apf_gradient(p, theta, params)

        direction = -g
        norm_dir = np.linalg.norm(direction)
        if norm_dir > 1e-6:
            direction /= norm_dir
        else:
            direction = np.zeros(2)

        p_des = p + step_gain * direction

        cmd = np.zeros((sim.n_worlds, sim.n_drones, 13))
        cmd[..., 0] = p_des[0]
        cmd[..., 1] = p_des[1]
        cmd[..., 2] = z_ref
        cmd[..., 3] = yaw_ref
        cmd[..., 6] = 0.0

        sim.state_control(cmd)
        sim.step(sim.freq // sim.control_freq)

        if (step * fps) % sim.control_freq < fps:
            sim.render()

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

    for sphere in SPHERE_OBSTACLES:
        circle = plt.Circle(sphere["center"], sphere["radius"], color="r", alpha=0.35)
        plt.gca().add_patch(circle)

    for wall in CAVE_WALLS:
        polygon = wall_polygon(wall)
        plt.plot(polygon[:, 0], polygon[:, 1], "r-", alpha=0.7)

    plt.plot(traj[:, 0], traj[:, 1], "b-", label="Trajectory")
    plt.scatter(GOAL[0], GOAL[1], c="g", marker="x", s=80, label="Goal")
    plt.legend()
    plt.title("Full APF + Safe-APF with shared obstacle definitions")
    plt.show()


if __name__ == "__main__":
    main()
