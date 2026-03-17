"""
cave_apf.py — APF path planning in a U-shape (cave) environment.

Environment layout (top-down view, x horizontal, y forward):

  y=7  +──────────────────────────────+
       │           GOAL (★)           │
  y=4.5│   [←── back wall ──────────]│
       │   [left│              │right]│
  y=2.7│   [arm │   (TRAP)     │arm  ]│
       │        ← opening 3 m →       │
  y=0  │           START (●)          │
       +──────────────────────────────+
      x=0      x=1.5  x=4.5         x=6

Box obstacles are 1.0 m tall (z=[0,1.0]).  The drone starts at z=0.7 inside
that height range, so it must navigate laterally (or climb above z=1.0).
Basic APF gets trapped in the concave pocket — the goal of this scene.
"""

from pathlib import Path

import numpy as np

from crazyflow.control import Control
from crazyflow.sim import Physics, Sim
from crazyflow.sim.visualize import draw_line, draw_points

# ---------------------------------------------------------------------------
# World parameters
# ---------------------------------------------------------------------------
BORDER_MIN = np.array([0.0, 0.0, 0.0])
BORDER_MAX = np.array([6.0, 7.0, 3.0])

START = np.array([3.0, 0.5, 0.7])
GOAL  = np.array([3.0, 6.5, 0.7])

# ---------------------------------------------------------------------------
# Box obstacles  (min corner, max corner)
# Each tuple: (np.array([xmin, ymin, zmin]), np.array([xmax, ymax, zmax]))
# ---------------------------------------------------------------------------
BOX_OBSTACLES = [
    # Back wall  — closes the U
    (np.array([1.5,  4.35, 0.0]), np.array([4.5,  4.65, 1.0])),
    # Left arm   — extends from back wall toward start
    (np.array([1.35, 2.70, 0.0]), np.array([1.65, 4.50, 1.0])),
    # Right arm  — mirror of left
    (np.array([4.35, 2.70, 0.0]), np.array([4.65, 4.50, 1.0])),
]

# ---------------------------------------------------------------------------
# APF tuning
# ---------------------------------------------------------------------------
K_ATT      = 1.5   # attractive gain
K_REP_WALL = 0.8   # repulsive gain – bounding-box walls
K_REP_OBS  = 1.2   # repulsive gain – box obstacles
D_REP      = 0.5   # influence distance from walls / obstacle surface (m)
STEP_LEN   = 0.03  # gradient-descent step size (m)
GOAL_TOL   = 0.08  # goal-reached tolerance (m)

# ---------------------------------------------------------------------------
# Force-field visualisation settings
# ---------------------------------------------------------------------------
ARROW_LEN  = 0.28  # fixed display length for every force arrow (m)
GRID_STEP  = 0.5   # spacing between force-field sample points (m)
FIELD_Z    = 0.7   # height of the 2D force-field slice (drone flight height)

# ---------------------------------------------------------------------------
# Simulation timing
# ---------------------------------------------------------------------------
SIM_FREQ     = 500
STATE_FREQ   = 100
CTRL_FREQ    = 100
SIM_DURATION = 25.0


# ---------------------------------------------------------------------------
# APF force functions
# ---------------------------------------------------------------------------

def attractive_force(pos: np.ndarray) -> np.ndarray:
    return K_ATT * (GOAL - pos)


def wall_repulsive_force(pos: np.ndarray) -> np.ndarray:
    """Repulsion from each face of the bounding box."""
    force = np.zeros(3)
    for i in range(3):
        d_min = pos[i] - BORDER_MIN[i]
        d_max = BORDER_MAX[i] - pos[i]
        if 1e-6 < d_min < D_REP:
            force[i] += K_REP_WALL * (1.0 / d_min - 1.0 / D_REP) / d_min ** 2
        if 1e-6 < d_max < D_REP:
            force[i] -= K_REP_WALL * (1.0 / d_max - 1.0 / D_REP) / d_max ** 2
    return force


def box_repulsive_force(pos: np.ndarray) -> np.ndarray:
    """Proper AABB repulsion: distance is measured to the nearest surface point."""
    force = np.zeros(3)
    for box_min, box_max in BOX_OBSTACLES:
        # Closest point on the box surface to pos
        closest = np.clip(pos, box_min, box_max)
        diff = pos - closest
        dist = np.linalg.norm(diff)

        if dist < 1e-6:
            # pos is inside the box — push out along the axis of least penetration
            pen = np.minimum(pos - box_min, box_max - pos)
            axis = int(np.argmin(pen))
            direction = np.zeros(3)
            direction[axis] = 1.0 if pos[axis] > (box_min[axis] + box_max[axis]) / 2 else -1.0
            force += K_REP_OBS * (1.0 / 1e-6 - 1.0 / D_REP) / (1e-6 ** 2) * direction
            continue

        if dist < D_REP:
            magnitude = K_REP_OBS * (1.0 / dist - 1.0 / D_REP) / dist ** 2
            force += magnitude * (diff / dist)

    return force


def apf_step(pos: np.ndarray) -> np.ndarray:
    total = attractive_force(pos) + wall_repulsive_force(pos) + box_repulsive_force(pos)
    norm = np.linalg.norm(total)
    if norm > 1e-6:
        total /= norm
    return np.clip(pos + STEP_LEN * total, BORDER_MIN + 0.01, BORDER_MAX - 0.01)


def plan_path(max_iters: int = 5000) -> np.ndarray:
    path = [START.copy()]
    pos  = START.copy()
    for _ in range(max_iters):
        if np.linalg.norm(pos - GOAL) < GOAL_TOL:
            path.append(GOAL.copy())  # only append goal when actually reached
            break
        new_pos = apf_step(pos)
        if np.linalg.norm(new_pos - pos) < 1e-4:  # forces cancelled out — local minima
            break
        pos = new_pos
        path.append(pos.copy())
    # path ends at wherever APF stopped — drone will hover there
    return np.array(path)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _norm(v: np.ndarray) -> np.ndarray:
    """Return unit vector; zero vector if magnitude is negligible."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.zeros(3)


def _inside_any_box(pos: np.ndarray) -> bool:
    return any(np.all(pos >= bmin) and np.all(pos <= bmax)
               for bmin, bmax in BOX_OBSTACLES)


def _box_corners(bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    xs, ys, zs = [bmin[0], bmax[0]], [bmin[1], bmax[1]], [bmin[2], bmax[2]]
    return np.array([[x, y, z] for x in xs for y in ys for z in zs])


def _box_edges(bmin: np.ndarray, bmax: np.ndarray):
    """Yield (p0, p1) pairs for the 12 edges of an AABB."""
    c = _box_corners(bmin, bmax)
    for i in range(8):
        for bit in range(3):
            j = i ^ (1 << bit)
            if j > i:
                yield c[i], c[j]


def draw_box(sim: Sim, bmin: np.ndarray, bmax: np.ndarray, rgba: np.ndarray):
    """Draw a wireframe box — 1 marker per edge (12 edges total)."""
    for p0, p1 in _box_edges(bmin, bmax):
        draw_line(sim, np.array([p0, p1]), rgba=rgba, start_size=1.5, end_size=1.5)


def draw_arrow(sim: Sim, origin: np.ndarray, direction: np.ndarray, rgba: np.ndarray):
    """Draw a force arrow with a fixed length of ARROW_LEN.
    Direction is normalized so every arrow has the same visual length.
    Tip is a sphere to mark the end point clearly.
    """
    tip = origin + _norm(direction) * ARROW_LEN
    draw_line(sim, np.array([origin, tip]), rgba=rgba, start_size=1.5, end_size=1.5)
    draw_points(sim, tip[None], rgba=rgba, size=0.04)


def draw_force_field(sim: Sim):
    """Draw the net APF force (attractive - repulsive) at a 2D grid of points.
    One arrow per grid point, all normalized to ARROW_LEN.
    Color: green (attraction dominates) → orange (repulsion dominates).
    """
    xs = np.arange(BORDER_MIN[0] + GRID_STEP, BORDER_MAX[0], GRID_STEP)
    ys = np.arange(BORDER_MIN[1] + GRID_STEP, BORDER_MAX[1], GRID_STEP)

    for x in xs:
        for y in ys:
            pt = np.array([x, y, FIELD_Z])

            if _inside_any_box(pt):
                continue

            f_att = attractive_force(pt)
            f_rep = box_repulsive_force(pt)
            net   = f_att - f_rep

            if np.linalg.norm(net) < 1e-6:
                continue

            # Blend green → orange based on how much repulsion dominates
            rep_ratio = np.clip(
                np.linalg.norm(f_rep) / (np.linalg.norm(f_att) + 1e-6), 0.0, 1.0
            )
            color = np.array([rep_ratio, 1.0 - rep_ratio * 0.6, 0.1, 0.95])
            draw_arrow(sim, pt, net, rgba=color)


def draw_scene(sim: Sim, path: np.ndarray):
    # Planned path — blue (150 pts → 149 markers)
    if len(path) >= 2:
        idx = np.linspace(0, len(path) - 1, min(150, len(path)), dtype=int)
        draw_line(sim, path[idx], rgba=np.array([0.2, 0.4, 1.0, 0.9]),
                  start_size=1.5, end_size=1.5)

    # Box obstacles — wireframe (12 edges × 3 boxes = 36 markers)
    obs_colors = [
        np.array([1.0, 0.2, 0.1, 1.0]),  # back wall
        np.array([1.0, 0.5, 0.1, 1.0]),  # left arm
        np.array([1.0, 0.5, 0.1, 1.0]),  # right arm
    ]
    for (bmin, bmax), color in zip(BOX_OBSTACLES, obs_colors):
        draw_box(sim, bmin, bmax, rgba=color)

    # Force field — net APF force arrows
    draw_force_field(sim)

    # Goal — bright green sphere
    draw_points(sim, GOAL[None],  rgba=np.array([0.0, 1.0, 0.2, 1.0]), size=0.10)
    # Start — white sphere
    draw_points(sim, START[None], rgba=np.array([1.0, 1.0, 1.0, 0.9]), size=0.06)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Planning path with APF (box obstacles)...")
    path = plan_path()
    final_dist = np.linalg.norm(path[-1] - GOAL)
    if final_dist < GOAL_TOL:
        print(f"  Path found — {len(path)} waypoints")
    else:
        print(f"  WARNING: APF did not reach goal (dist={final_dist:.3f} m) — local minima detected!")
    print(f"  Final position: {path[-2].round(3)}")

    xml_path = Path(__file__).parent / "cave_env.xml"
    sim = Sim(
        n_worlds=1,
        n_drones=1,
        physics=Physics.first_principles,
        control=Control.state,
        freq=SIM_FREQ,
        state_freq=STATE_FREQ,
        attitude_freq=SIM_FREQ,
        device="cpu",
        xml_path=xml_path,
    )
    sim.reset()

    sim.data = sim.data.replace(
        states=sim.data.states.replace(
            pos=sim.data.states.pos.at[0, 0].set(START),
            rotor_vel=sim.data.states.rotor_vel.at[0, 0].set(np.ones(4) * 20000),
        )
    )

    cmd             = np.zeros((1, 1, 13))
    fps             = 60
    steps_per_ctrl  = sim.freq // CTRL_FREQ
    waypoint_idx    = 0

    print("Simulating... (press ESC to quit)")
    try:
        for i in range(int(SIM_DURATION * CTRL_FREQ)):
            pos = np.array(sim.data.states.pos[0, 0])

            while (waypoint_idx < len(path) - 1
                   and np.linalg.norm(pos - path[waypoint_idx]) < GOAL_TOL):
                waypoint_idx += 1

            cmd[0, 0, :3] = path[waypoint_idx]
            sim.state_control(cmd)
            sim.step(steps_per_ctrl)

            if (i * fps) % CTRL_FREQ < fps:
                draw_scene(sim, path)
                sim.render()

            if np.linalg.norm(pos - GOAL) < GOAL_TOL:
                print(f"  Goal reached at t={i / CTRL_FREQ:.2f}s")
                break
        else:
            print("  Time limit reached — drone likely stuck in local minima.")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
