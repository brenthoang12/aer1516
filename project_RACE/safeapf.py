import numpy as np
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt

from crazyflow.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.sim import use_box_collision


def quat_to_yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def dsearch(points, query):
    d = np.linalg.norm(points - query[None, :], axis=1)
    idx = np.argmin(d)
    return idx, d[idx]


def build_scene_xml():
    # All obstacles are boxes → box–box collisions supported by MJX
    return """
<mujoco model="SAPF Scene">
    <option timestep="0.001"/>
    <worldbody>

        <!-- Floor -->
        <geom type="plane" size="0 0 0.05" rgba="0.8 0.8 0.8 1"/>

        <!-- Box obstacles with collision -->
        <body name="box1" pos="1 1 0">
            <geom type="box" size="0.25 0.25 0.5"
                  rgba="1 0 0 1"
                  contype="1" conaffinity="1"
                  density="1000"/>
        </body>

        <body name="box2" pos="2 1 0">
            <geom type="box" size="0.25 0.25 0.5"
                  rgba="1 0 0 1"
                  contype="1" conaffinity="1"
                  density="1000"/>
        </body>

        <!-- U-shape walls with collision -->
        <body name="uwall_bottom" pos="1.5 2.05 0">
            <geom type="box" size="0.5 0.05 0.5"
                  rgba="0.8 0.2 0.2 1"
                  contype="1" conaffinity="1"
                  density="1000"/>
        </body>

        <body name="uwall_left" pos="1.05 2.5 0">
            <geom type="box" size="0.05 0.5 0.5"
                  rgba="0.8 0.2 0.2 1"
                  contype="1" conaffinity="1"
                  density="1000"/>
        </body>

        <body name="uwall_right" pos="1.95 2.5 0">
            <geom type="box" size="0.05 0.5 0.5"
                  rgba="0.8 0.2 0.2 1"
                  contype="1" conaffinity="1"
                  density="1000"/>
        </body>

        <!-- Goal marker (visual only) -->
        <body name="goal" pos="3 3 0.5">
            <geom type="sphere" size="0.1"
                  rgba="0 1 0 1"
                  contype="0" conaffinity="0"/>
        </body>

    </worldbody>
</mujoco>
"""


def apf_gradient(p, theta, obstacles, params):
    # Full APF (Sec. II A,B,C) + Safe-APF (Sec. III A,B)
    zeta = params["zeta"]
    eta = params["eta"]
    dstar = params["dstar"]
    Qstar = params["Qstar"]
    dsafe = params["dsafe"]
    dvort = params["dvort"]
    alpha_th = params["alpha_th"]
    goal = params["goal"]

    # Attractive (II-A)
    diff_goal = p - goal
    d_goal = np.linalg.norm(diff_goal)
    if d_goal <= dstar:
        grad_att = zeta * diff_goal
    else:
        grad_att = zeta * dstar * diff_goal / (d_goal + 1e-9)

    grad_rep_total = np.zeros(2)

    for obs in obstacles:
        idx, dist = dsearch(obs, p)
        p_obs = obs[idx]
        dist = max(dist, 1e-6)

        if dist > Qstar:
            continue

        diff_o = p - p_obs

        # Repulsive (II-B): ∇U_rep = η (1/Q* - 1/d) (1/d^2) (p - p_obs)
        grad_rep = eta * (1.0 / Qstar - 1.0 / dist) * (1.0 / (dist**2)) * diff_o

        # Vortex + Safe-APF (II-C, III)
        alpha = theta - np.arctan2(p_obs[1] - p[1], p_obs[0] - p[0])
        alpha = wrap(alpha)
        D_alpha = +1 if abs(alpha) <= alpha_th else -1

        if dist <= dsafe:
            drel = 0.0
        elif dist >= dvort:
            drel = 1.0
        else:
            drel = (dist - dsafe) / (dvort - dsafe)
        drel = np.clip(drel, 0.0, 1.0)

        gamma = np.pi * D_alpha * drel
        R = np.array([[np.cos(gamma), -np.sin(gamma)],
                      [np.sin(gamma),  np.cos(gamma)]])
        grad_rep_total += R @ grad_rep

    return grad_att + grad_rep_total


def main():
    # Build scene
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
    goal = np.array([3.0, 3.0])

    # Obstacles for APF math (box centers approximated as discs)
    angles = np.linspace(0, 2 * np.pi, 360)
    obstacles = []

    for c in [(1.0, 1.0), (2.0, 1.0)]:
        xs = 0.25 * np.cos(angles) + c[0]
        ys = 0.25 * np.sin(angles) + c[1]
        obstacles.append(np.stack([xs, ys], axis=1))

    def rect(x, y, w, h, n=400):
        xs = np.random.uniform(x, x + w, n)
        ys = np.random.uniform(y, y + h, n)
        return np.stack([xs, ys], axis=1)

    obstacles += [
        rect(1.0, 2.0, 1.0, 0.1),
        rect(1.0, 2.0, 0.1, 1.0),
        rect(1.9, 2.0, 0.1, 1.0),
    ]

    params = dict(
        zeta=1.1547,
        eta=0.0732,
        dstar=0.3,
        Qstar=0.5,
        dsafe=0.2,
        dvort=0.35,
        alpha_th=np.deg2rad(5),
        goal=goal,
    )

    step_gain = 0.15
    traj = []

    print("Running full APF + Safe-APF with box obstacles...")

    for step in range(5000):
        states = sim.data.states
        pos = states.pos[0, 0]
        p = np.array([pos[0], pos[1]])
        theta = quat_to_yaw(states.quat[0, 0])

        traj.append(p.copy())

        if np.linalg.norm(p - goal) < 0.05:
            print("Reached goal.")
            break

        g = apf_gradient(p, theta, obstacles, params)

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
        cmd[..., 6] = 0.0  # vz_des = 0, stabilize altitude

        sim.state_control(cmd)
        sim.step(sim.freq // sim.control_freq)

        if (step * fps) % sim.control_freq < fps:
            sim.render()

    sim.close()

    # Trail (2D)
    traj = np.array(traj)
    plt.figure(figsize=(6, 6))
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    for obs in obstacles:
        plt.scatter(obs[:, 0], obs[:, 1], s=2, c="r", alpha=0.4)

    plt.plot(traj[:, 0], traj[:, 1], "b-", label="Trajectory")
    plt.scatter(goal[0], goal[1], c="g", marker="x", s=80, label="Goal")
    plt.legend()
    plt.title("Full APF + Safe-APF with box obstacles (trail)")
    plt.show()


if __name__ == "__main__":
    main()
