"""
Artificial Potential Field (APF) for Mobile Robots
Based on: Khatib (1986) - "Real-Time Obstacle Avoidance for Manipulators and Mobile Robots"
          The International Journal of Robotics Research, Vol. 5, No. 1.

Core idea (Section 3):
  The robot moves in a field of forces.
  The goal is an attractive pole; obstacles are repulsive surfaces.

  Total potential:  U_art(x) = U_att(x) + U_rep(x)         [Eq. 7]
  Total force:      F_art     = F_att    + F_rep             [Eq. 9]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class APF:
    """
    Implements the Artificial Potential Field from Khatib 1986.
    
    Parameters
    ----------
    k_p  : float  Attractive gain (position stiffness)
    eta  : float  Repulsive gain  (η in the paper)
    rho_0: float  Influence radius (ρ_0) — repulsion is zero beyond this distance
    """

    def __init__(self, k_p: float = 1.0, eta: float = 1.0, rho_0: float = 2.0):
        self.k_p   = k_p
        self.eta   = eta
        self.rho_0 = rho_0

    # ------------------------------------------------------------------
    # Attractive field  (Eq. 11)
    # ------------------------------------------------------------------

    def attractive_potential(self, x: np.ndarray, x_d: np.ndarray) -> float:
        """
        U_att(x) = 0.5 * k_p * ||x - x_d||^2        (Eq. 11)

        A simple parabolic well centred at the goal x_d.
        """
        return 0.5 * self.k_p * np.dot(x - x_d, x - x_d)

    def attractive_force(self, x: np.ndarray, x_d: np.ndarray) -> np.ndarray:
        """
        F_att = -∇U_att = k_p * (x_d - x)           (Eq. 10)

        Points directly toward the goal; magnitude grows with distance.
        """
        return self.k_p * (x_d - x)

    # ------------------------------------------------------------------
    # Repulsive field / FIRAS  (Eq. 17, 18)
    # ------------------------------------------------------------------

    def repulsive_potential(self, x: np.ndarray,
                             obstacles: list[np.ndarray]) -> float:
        """
        U_rep(x) = Σ_O  { 0.5*η*(1/ρ - 1/ρ_0)^2   if ρ ≤ ρ_0   (Eq. 17)
                         { 0                          if ρ > ρ_0

        ρ is the shortest distance from x to obstacle O.
        Obstacles are treated as point obstacles (ρ = ||x - x_obs||).
        """
        U = 0.0
        for obs in obstacles:
            rho = np.linalg.norm(x - obs)
            if rho <= self.rho_0 and rho > 1e-6:
                U += 0.5 * self.eta * (1.0 / rho - 1.0 / self.rho_0) ** 2
        return U

    def repulsive_force(self, x: np.ndarray,
                         obstacles: list[np.ndarray]) -> np.ndarray:
        """
        FIRAS — Force Inducing an Artificial Repulsion from the Surface (Eq. 18):

        F_rep = Σ_O  { η*(1/ρ - 1/ρ_0) * (1/ρ^2) * (∂ρ/∂x)   if ρ ≤ ρ_0
                      { 0                                         if ρ > ρ_0

        For a point obstacle: ∂ρ/∂x = (x - x_obs) / ρ           (Eq. 19)
        The force is directed away from each obstacle and
        vanishes outside the influence radius ρ_0.
        """
        F = np.zeros_like(x, dtype=float)
        for obs in obstacles:
            diff = x - obs
            rho  = np.linalg.norm(diff)
            if rho <= self.rho_0 and rho > 1e-6:
                grad_rho = diff / rho                          # ∂ρ/∂x  (unit vector)
                F += self.eta * (1.0 / rho - 1.0 / self.rho_0) * (1.0 / rho**2) * grad_rho
        return F

    # ------------------------------------------------------------------
    # Combined field
    # ------------------------------------------------------------------

    def total_potential(self, x: np.ndarray, x_d: np.ndarray,
                         obstacles: list[np.ndarray]) -> float:
        """U_art = U_att + U_rep   (Eq. 7)"""
        return self.attractive_potential(x, x_d) + self.repulsive_potential(x, obstacles)

    def total_force(self, x: np.ndarray, x_d: np.ndarray,
                     obstacles: list[np.ndarray]) -> np.ndarray:
        """F_art = F_att + F_rep   (Eq. 9)"""
        return self.attractive_force(x, x_d) + self.repulsive_force(x, obstacles)


# ---------------------------------------------------------------------------
# Robot — first-order velocity dynamics with speed limiting (Eq. 13–15)
# ---------------------------------------------------------------------------

class Robot:
    """
    Point robot with velocity control and speed saturation.

    The paper (Eq. 12–15) reformulates the force command as a
    desired velocity direction with an upper speed limit V_max:

        ẋ_d = (k_p / k_v) * (x_d - x)                 (Eq. 13)
        ν   = min(1,  V_max / ||ẋ_d||)                 (Eq. 15)
        F_m = -k_v * (ẋ - ν * ẋ_d)                    (Eq. 14)

    Outside obstacle regions the robot follows a straight line
    to the goal at speed V_max.

    Parameters
    ----------
    position : array-like   Initial (x, y) position
    k_v      : float        Velocity damping gain
    V_max    : float        Maximum speed  (m/s)
    """

    def __init__(self, position, k_v: float = 2.0, V_max: float = 1.5):
        self.x     = np.array(position, dtype=float)
        self.xdot  = np.zeros(2)
        self.k_v   = k_v
        self.V_max = V_max

        self.trajectory = [self.x.copy()]

    def step(self, dt: float, F: np.ndarray):
        """
        Integrate one time step using Euler integration.
        F is the total APF force vector.
        """
        # Treat force as acceleration (unit mass — Eq. 6 decoupled end-effector)
        self.xdot += F * dt

        # Clamp speed to V_max
        speed = np.linalg.norm(self.xdot)
        if speed > self.V_max:
            self.xdot = self.xdot / speed * self.V_max

        self.x += self.xdot * dt
        self.trajectory.append(self.x.copy())


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

def run_simulation(start, goal, obstacles, apf: APF, robot: Robot,
                   steps: int = 2000, dt: float = 0.02,
                   goal_tol: float = 0.15) -> dict:
    """
    Run the APF simulation.

    Returns a dict with 'trajectory' and 'potentials' lists.
    """
    x_d  = np.array(goal, dtype=float)
    traj = [robot.x.copy()]
    pots = []

    for _ in range(steps):
        F = apf.total_force(robot.x, x_d, obstacles)
        pots.append(apf.total_potential(robot.x, x_d, obstacles))
        robot.step(dt, F)
        traj.append(robot.x.copy())

        if np.linalg.norm(robot.x - x_d) < goal_tol:
            break

    return {"trajectory": np.array(traj), "potentials": np.array(pots)}


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize(start, goal, obstacles, apf: APF,
              result: dict, world_size=(10, 10)):
    """
    Two-panel figure:
      Left  — potential field (contour) + force field (quiver) + trajectory
      Right — total potential energy along the robot's trajectory
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Khatib 1986 — Artificial Potential Field (APF)", fontsize=14)

    # ------------------------------------------------------------------
    # Left panel: field + trajectory
    # ------------------------------------------------------------------
    W, H   = world_size
    res    = 80                    # grid resolution
    xs     = np.linspace(0, W, res)
    ys     = np.linspace(0, H, res)
    XX, YY = np.meshgrid(xs, ys)

    # Evaluate potential on grid  (capped for readability)
    ZZ = np.zeros_like(XX)
    UU = np.zeros_like(XX)
    VV = np.zeros_like(YY)
    x_d = np.array(goal, dtype=float)

    for i in range(res):
        for j in range(res):
            pt = np.array([XX[i, j], YY[i, j]])
            ZZ[i, j] = apf.total_potential(pt, x_d, obstacles)
            F = apf.total_force(pt, x_d, obstacles)
            UU[i, j] = F[0]
            VV[i, j] = F[1]

    ZZ = np.clip(ZZ, 0, np.percentile(ZZ, 97))   # cap peaks for visibility

    # Potential contour
    cf = ax1.contourf(XX, YY, ZZ, levels=40, cmap="RdYlGn_r", alpha=0.75)
    fig.colorbar(cf, ax=ax1, label="U_art (potential)")

    # Force quiver (sub-sampled)
    step = 6
    ax1.quiver(XX[::step, ::step], YY[::step, ::step],
               UU[::step, ::step], VV[::step, ::step],
               color="steelblue", alpha=0.6, scale=80, width=0.003,
               label="Force field")

    # Obstacles
    for obs in obstacles:
        circle = plt.Circle(obs, apf.rho_0, color="gray", alpha=0.25, zorder=2)
        ax1.add_patch(circle)
        ax1.plot(*obs, "ks", markersize=8, zorder=3)

    # Trajectory
    traj = result["trajectory"]
    ax1.plot(traj[:, 0], traj[:, 1], "w-", linewidth=2, zorder=4, label="Robot path")
    ax1.plot(*start, "go", markersize=12, zorder=5, label="Start")
    ax1.plot(*goal,  "r*", markersize=16, zorder=5, label="Goal")

    ax1.set_xlim(0, W)
    ax1.set_ylim(0, H)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("Potential field & robot trajectory")
    ax1.legend(loc="upper left", fontsize=9)

    # ------------------------------------------------------------------
    # Right panel: potential energy vs step
    # ------------------------------------------------------------------
    pots = result["potentials"]
    ax2.plot(pots, color="royalblue", linewidth=1.8)
    ax2.set_xlabel("Simulation step")
    ax2.set_ylabel("U_art  (total potential)")
    ax2.set_title("Potential energy along trajectory")
    ax2.grid(True, alpha=0.35)

    plt.tight_layout()
    plt.savefig("apf_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Figure saved to apf_results.png")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Khatib 1986 — Artificial Potential Field Demo")
    print("=" * 60)

    # World
    START = [0.5, 0.5]
    GOAL  = [9.0, 8.5]

    # Three point obstacles in the path
    OBSTACLES = [
        np.array([3.5, 3.0]),
        np.array([6.0, 6.5]),
        np.array([4.8, 7.0]),
    ]

    # APF parameters
    apf = APF(
        k_p   = 1.0,    # attractive gain
        eta   = 3.0,    # repulsive gain  (η)
        rho_0 = 2.0,    # influence radius (ρ_0)
    )

    # Robot
    robot = Robot(
        position = START,
        k_v      = 2.0,   # velocity damping
        V_max    = 1.5,   # speed limit (m/s)
    )

    # Run
    print("Running simulation...")
    result = run_simulation(
        start     = START,
        goal      = GOAL,
        obstacles = OBSTACLES,
        apf       = apf,
        robot     = robot,
        steps     = 3000,
        dt        = 0.02,
        goal_tol  = 0.15,
    )

    traj = result["trajectory"]
    final_dist = np.linalg.norm(traj[-1] - np.array(GOAL))
    print(f"Steps taken      : {len(traj) - 1}")
    print(f"Final position   : ({traj[-1, 0]:.3f}, {traj[-1, 1]:.3f})")
    print(f"Distance to goal : {final_dist:.3f} m")
    print(f"Goal reached     : {final_dist < 0.5}")
    print()

    # Visualise
    visualize(
        start     = START,
        goal      = GOAL,
        obstacles = OBSTACLES,
        apf       = apf,
        result    = result,
        world_size= (10, 10),
    )
