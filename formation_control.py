"""
Fixed-Time Formation-Containment Control for Multi-Agent Systems
Based on: Su et al. (2024) - IEEE Transactions on Vehicular Technology

2D Top-Down Simulation of Quadcopter Formation Control
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from scipy.spatial import ConvexHull


class Agent:
    """
    Represents a single agent (quadcopter) in the system.

    Dynamics: Double integrator
    - ṗᵢ = vᵢ  (position derivative = velocity)
    - v̇ᵢ = uᵢ  (velocity derivative = control input/acceleration)
    """

    def __init__(self, agent_id, position, is_leader=False):
        self.id = agent_id
        self.position = np.array(position, dtype=float)  # [x, y]
        self.velocity = np.zeros(2)  # [vx, vy]
        self.acceleration = np.zeros(2)  # [ax, ay]
        self.is_leader = is_leader
        self.control_input = np.zeros(2)

        # History for visualization
        self.trajectory = [self.position.copy()]

    def update(self, dt):
        """Update agent state using Euler integration"""
        # v̇ = u
        self.velocity += self.acceleration * dt
        # ṗ = v
        self.position += self.velocity * dt

        # Store trajectory
        self.trajectory.append(self.position.copy())


class VirtualLeader:
    """
    Virtual leader agent that generates reference trajectory.
    Implements motion planning with APF.
    """

    def __init__(self, position, goal):
        self.position = np.array(position, dtype=float)
        self.velocity = np.zeros(2)
        self.goal = np.array(goal, dtype=float)
        self.trajectory = [self.position.copy()]

    def update(self, dt):
        """Update virtual leader position"""
        self.position += self.velocity * dt
        self.trajectory.append(self.position.copy())


class FormationContainmentController:
    """
    Implements the fixed-time formation-containment control scheme.

    Key equations from paper:
    - Formation error: ξᵢ = Σⱼ∈Nᵢ aᵢⱼ(pᵢ - pⱼ - δᵢⱼ)
    - Containment error: ξₖ = Σⱼ∈Nₖ aₖⱼ(pₖ - pⱼ)
    - Control law (Theorem 2, Eq. 23):
      uᵢ = -1/γᵢ[(α + β*μ̇/μ)ξᵢ + (α + β*μ̇/μ)ζᵢ - Σⱼ aᵢⱼv̇ⱼ]
    """

    def __init__(self, agents, virtual_leader, adjacency_matrix,
                 formation_offsets, T=20.0, alpha=0.05, beta=0.4, h=5.0):
        """
        Parameters:
        - agents: List of Agent objects
        - virtual_leader: VirtualLeader object
        - adjacency_matrix: Communication graph (NxN matrix)
        - formation_offsets: Dictionary {(i,j): [δx, δy]} for leader offsets
        - T: Fixed convergence time (seconds)
        - alpha, beta: Control gains
        - h: Time-varying function parameter
        """
        self.agents = agents
        self.virtual_leader = virtual_leader
        self.adjacency_matrix = adjacency_matrix
        self.formation_offsets = formation_offsets

        # Fixed-time parameters (from Theorem 2)
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.h = h

        self.time = 0.0

        # Separate leaders and followers
        self.leaders = [a for a in agents if a.is_leader]
        self.followers = [a for a in agents if not a.is_leader]

    def mu(self, t):
        """
        Time-varying function from Eq. (20):
        μ(t) = (T/(T-t))^h  for t < T
        μ(t) = 0            for t ≥ T
        """
        if t < self.T:
            return (self.T / (self.T - t)) ** self.h
        else:
            return 0.0

    def mu_dot(self, t):
        """
        [Probable error]
        Derivative of μ(t):
        μ̇(t) = h*T^h / (T-t)^(h+1)  for t < T
        """
        if t < self.T - 0.01:  # Small epsilon for numerical stability
            return self.h * (self.T ** self.h) / ((self.T - t) ** (self.h + 1))
        else:
            return 0.0

    def compute_formation_error(self, agent):
        """
        Compute formation tracking error for a leader agent.
        ξᵢ = Σⱼ∈Nᵢ aᵢⱼ(pᵢ - pⱼ - δᵢⱼ)
        """
        error = np.zeros(2)
        i = agent.id

        for j, a_ij in enumerate(self.adjacency_matrix[i]):
            if a_ij > 0:
                if j < len(self.agents):
                    # Connection to another agent
                    neighbor = self.agents[j]
                    delta_ij = self.formation_offsets.get((i, j), np.zeros(2))
                    error += a_ij * (agent.position - neighbor.position - delta_ij)
                else:
                    # Connection to virtual leader
                    delta_ij = self.formation_offsets.get((i, j), np.zeros(2))
                    error += a_ij * (agent.position - self.virtual_leader.position - delta_ij)

        return error

    def compute_containment_error(self, agent):
        """
        Compute containment error for a follower agent.
        ξₖ = Σⱼ∈Nₖ aₖⱼ(pₖ - pⱼ)
        """
        error = np.zeros(2)
        k = agent.id

        for j, a_kj in enumerate(self.adjacency_matrix[k]):
            if a_kj > 0 and j < len(self.agents):
                neighbor = self.agents[j]
                error += a_kj * (agent.position - neighbor.position)

        return error

    def compute_velocity_error(self, agent):
        """
        Compute velocity consensus error.
        ζᵢ = Σⱼ∈Nᵢ aᵢⱼ(vᵢ - vⱼ)
        """
        error = np.zeros(2)
        i = agent.id

        for j, a_ij in enumerate(self.adjacency_matrix[i]):
            if a_ij > 0:
                if j < len(self.agents):
                    neighbor = self.agents[j]
                    error += a_ij * (agent.velocity - neighbor.velocity)
                else:
                    # Virtual leader
                    error += a_ij * (agent.velocity - self.virtual_leader.velocity)

        return error

    def compute_control(self, agent):
        """
        Compute fixed-time control input for an agent (Theorem 2, Eq. 23).

        For leaders:
        uᵢ = -1/γᵢ[(α + β*μ̇/μ)ξᵢ + (α + β*μ̇/μ)ζᵢ - Σⱼ aᵢⱼv̇ⱼ]

        For followers:
        uₖ = -1/γₖ[(α + β*μ̇/μ)ξₖ + (α + β*μ̇/μ)ζₖ - Σⱼ aₖⱼv̇ⱼ]
        """
        i = agent.id

        # Compute γᵢ (sum of edge weights)
        gamma_i = np.sum(self.adjacency_matrix[i])

        if gamma_i == 0:
            return np.zeros(2)

        # Compute time-varying gains
        mu_t = self.mu(self.time)
        mu_dot_t = self.mu_dot(self.time)

        if mu_t > 1e-6:
            time_gain = self.alpha + self.beta * mu_dot_t / mu_t
        else:
            time_gain = self.alpha

        # Compute position and velocity errors
        if agent.is_leader:
            xi = self.compute_formation_error(agent)
        else:
            xi = self.compute_containment_error(agent)

        zeta = self.compute_velocity_error(agent)

        # Acceleration feedback term: Σⱼ aᵢⱼv̇ⱼ
        accel_feedback = np.zeros(2)
        for j, a_ij in enumerate(self.adjacency_matrix[i]):
            if a_ij > 0 and j < len(self.agents):
                accel_feedback += a_ij * self.agents[j].acceleration

        # Control law
        control = -(1.0 / gamma_i) * (
            time_gain * xi +
            time_gain * zeta -
            accel_feedback
        )

        return control

    def step(self, dt):
        """Execute one control step"""
        # Store previous accelerations (needed for feedback)
        for agent in self.agents:
            agent.acceleration = self.compute_control(agent)

        # Update all agents
        for agent in self.agents:
            agent.update(dt)

        self.time += dt


class MotionPlanner:
    """
    High-level motion planning for virtual leader using APF.
    Implements Algorithm 1 from the paper.
    """

    def __init__(self, virtual_leader, goal, obstacles=None,
                 ka=1.0, kb=2.0, Q_star=2.0, d_star_g=0.5, rs=1.0, P_star=0.1):
        """
        Parameters:
        - ka: Attractive gain
        - kb: Repulsive gain
        - Q_star: Threshold distance for repulsive field
        - d_star_g: Threshold distance near goal
        - rs: Safe region radius
        - P_star: Threshold for curl-free vector field activation
        """
        self.virtual_leader = virtual_leader
        self.goal = goal
        self.obstacles = obstacles if obstacles is not None else []

        # APF parameters
        self.ka = ka
        self.kb = kb
        self.Q_star = Q_star
        self.d_star_g = d_star_g
        self.rs = rs
        self.P_star = P_star

    def attractive_force(self):
        """
        Compute attractive force toward goal (Eq. 31).
        ∇Ua = ka*(p - pg)           if d ≤ d*g
        ∇Ua = d*g*ka*(p - pg)/d     if d > d*g
        """
        d = np.linalg.norm(self.virtual_leader.position - self.goal)

        if d <= self.d_star_g:
            grad = self.ka * (self.virtual_leader.position - self.goal)
        else:
            direction = (self.virtual_leader.position - self.goal) / (d + 1e-10)
            grad = self.d_star_g * self.ka * direction

        return -grad  # Negative gradient for attractive force

    def repulsive_force(self):
        """
        Compute repulsive force from obstacles (Eq. 34).
        ∇Ub = f(p)*kb*∇D(p)/D²(p)  if D ≤ Q*
        ∇Ub = 0                     if D > Q*

        where f(p) = 1/Q* - 1/D(p)
        """
        total_force = np.zeros(2)

        for obstacle in self.obstacles:
            D = np.linalg.norm(self.virtual_leader.position - obstacle)

            if D <= self.Q_star and D > 0.1:  # Avoid singularity
                f_p = (1.0 / self.Q_star) - (1.0 / D)
                direction = (self.virtual_leader.position - obstacle) / D
                grad = f_p * self.kb * direction / (D ** 2)
                total_force -= grad  # Negative gradient pushes away

        return total_force

    def curl_free_vector_field(self):
        """
        [TODO: need counter clockwise]
        
        Compute curl-free vector field to avoid local minima (Eq. 36-38).
        ∇Uc = R * ∇Ub

        where R is a rotation matrix (90° clockwise or counterclockwise)
        """
        repulsive = self.repulsive_force()

        # Determine rotation direction based on relative angle
        # Simplified: rotate 90° counterclockwise
        R = np.array([[0, -1],
                      [1, 0]])

        return R @ (-repulsive)  # Apply to repulsive gradient

    def compute_velocity_reference(self):
        """
        Compute motion reference for virtual leader (Algorithm 1).
        vref = -∇Ua - ∇Ub  (normal)
        vref = -∇Ua - ∇Uc  (when trapped in local minimum)
        """
        attractive = self.attractive_force()
        repulsive = self.repulsive_force()

        # Check if trapped in local minimum
        vref_normal = attractive + repulsive

        if np.linalg.norm(vref_normal) < self.P_star and len(self.obstacles) > 0:
            # Use curl-free vector field
            curl_free = self.curl_free_vector_field()
            vref = attractive + curl_free
        else:
            vref = vref_normal

        # Limit maximum velocity
        max_vel = 2.0
        if np.linalg.norm(vref) > max_vel:
            vref = vref / np.linalg.norm(vref) * max_vel

        return vref

    def step(self, dt):
        """Update virtual leader velocity and position"""
        self.virtual_leader.velocity = self.compute_velocity_reference()
        self.virtual_leader.update(dt)


class Simulation:
    """
    Main simulation environment.
    """

    def __init__(self, agents, virtual_leader, controller, motion_planner,
                 obstacles=None, arena_size=(10, 10)):
        self.agents = agents
        self.virtual_leader = virtual_leader
        self.controller = controller
        self.motion_planner = motion_planner
        self.obstacles = obstacles if obstacles is not None else []
        self.arena_size = arena_size

        # Simulation parameters
        self.dt = 0.05  # 50ms time step
        self.time = 0.0
        self.max_time = 60.0  # 60 seconds max

        # Data recording
        self.formation_errors = []
        self.containment_errors = []
        self.times = []

    def step(self):
        """Execute one simulation step"""
        # Motion planning for virtual leader
        self.motion_planner.step(self.dt)

        # Formation-containment control
        self.controller.step(self.dt)

        # Record metrics
        self.time += self.dt
        self.times.append(self.time)

        # Compute errors
        form_err = 0
        for agent in self.controller.leaders:
            err = self.controller.compute_formation_error(agent)
            form_err += np.linalg.norm(err)

        cont_err = 0
        for agent in self.controller.followers:
            err = self.controller.compute_containment_error(agent)
            cont_err += np.linalg.norm(err)

        self.formation_errors.append(form_err / max(len(self.controller.leaders), 1))
        self.containment_errors.append(cont_err / max(len(self.controller.followers), 1))

    def run(self, steps=None):
        """Run simulation for specified steps or until max_time"""
        if steps is None:
            steps = int(self.max_time / self.dt)

        for _ in range(steps):
            self.step()
            if self.time >= self.max_time:
                break

    def plot_results(self):
        """Plot simulation results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Final formation
        ax1 = axes[0, 0]
        self._plot_formation(ax1)

        # Plot 2: Formation tracking error
        ax2 = axes[0, 1]
        ax2.plot(self.times, self.formation_errors, 'b-', linewidth=2)
        ax2.axvline(x=self.controller.T, color='r', linestyle='--',
                    label=f'Fixed Time T={self.controller.T}s')
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Formation Error', fontsize=12)
        ax2.set_title('Formation Tracking Error', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Plot 3: Containment error
        ax3 = axes[1, 0]
        ax3.plot(self.times, self.containment_errors, 'g-', linewidth=2)
        ax3.axvline(x=self.controller.T, color='r', linestyle='--',
                    label=f'Fixed Time T={self.controller.T}s')
        ax3.set_xlabel('Time (s)', fontsize=12)
        ax3.set_ylabel('Containment Error', fontsize=12)
        ax3.set_title('Containment Error', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Plot 4: Trajectories
        ax4 = axes[1, 1]
        self._plot_trajectories(ax4)

        plt.tight_layout()
        plt.savefig('formation_control_results.png', dpi=150, bbox_inches='tight')
        plt.show()

    def _plot_formation(self, ax):
        """Plot current formation configuration"""
        ax.set_xlim(-1, self.arena_size[0] + 1)
        ax.set_ylim(-1, self.arena_size[1] + 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title('Final Formation Configuration', fontsize=14)

        # Plot obstacles
        for obs in self.obstacles:
            circle = plt.Circle(obs, 0.3, color='gray', alpha=0.5)
            ax.add_patch(circle)

        # Plot goal
        ax.plot(self.motion_planner.goal[0], self.motion_planner.goal[1],
                'r*', markersize=20, label='Goal')

        # Plot virtual leader
        ax.plot(self.virtual_leader.position[0], self.virtual_leader.position[1],
                'g*', markersize=15, label='Virtual Leader')

        # Plot leaders
        leader_positions = np.array([a.position for a in self.controller.leaders])
        if len(leader_positions) > 0:
            ax.scatter(leader_positions[:, 0], leader_positions[:, 1],
                      c='blue', s=150, marker='^', edgecolors='black',
                      linewidths=2, label='Leaders', zorder=5)

            # Draw formation shape (convex hull of leaders)
            if len(leader_positions) >= 3:
                hull = ConvexHull(leader_positions)
                for simplex in hull.simplices:
                    ax.plot(leader_positions[simplex, 0],
                           leader_positions[simplex, 1], 'b--', alpha=0.5)

        # Plot followers
        follower_positions = np.array([a.position for a in self.controller.followers])
        if len(follower_positions) > 0:
            ax.scatter(follower_positions[:, 0], follower_positions[:, 1],
                      c='orange', s=100, marker='o', edgecolors='black',
                      linewidths=2, label='Followers', zorder=5)

        ax.legend(fontsize=10)

    def _plot_trajectories(self, ax):
        """Plot agent trajectories"""
        ax.set_xlim(-1, self.arena_size[0] + 1)
        ax.set_ylim(-1, self.arena_size[1] + 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title('Agent Trajectories', fontsize=14)

        # Plot obstacles
        for obs in self.obstacles:
            circle = plt.Circle(obs, 0.3, color='gray', alpha=0.5)
            ax.add_patch(circle)

        # Plot goal
        ax.plot(self.motion_planner.goal[0], self.motion_planner.goal[1],
                'r*', markersize=20, label='Goal')

        # Plot virtual leader trajectory
        vl_traj = np.array(self.virtual_leader.trajectory)
        ax.plot(vl_traj[:, 0], vl_traj[:, 1], 'g-', linewidth=2,
                alpha=0.7, label='Virtual Leader')

        # Plot leader trajectories
        for agent in self.controller.leaders:
            traj = np.array(agent.trajectory)
            ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.5, linewidth=1.5)
            ax.scatter(traj[0, 0], traj[0, 1], c='blue', s=50, marker='x')
            ax.scatter(traj[-1, 0], traj[-1, 1], c='blue', s=150, marker='^',
                      edgecolors='black', linewidths=2)

        # Plot follower trajectories
        for agent in self.controller.followers:
            traj = np.array(agent.trajectory)
            ax.plot(traj[:, 0], traj[:, 1], 'orange', alpha=0.5, linewidth=1.5)
            ax.scatter(traj[0, 0], traj[0, 1], c='orange', s=50, marker='x')
            ax.scatter(traj[-1, 0], traj[-1, 1], c='orange', s=100, marker='o',
                      edgecolors='black', linewidths=2)

        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='g', lw=2, label='Virtual Leader'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='b',
                   markersize=10, label='Leaders', markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                   markersize=8, label='Followers', markeredgecolor='black'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='r',
                   markersize=15, label='Goal')
        ]
        ax.legend(handles=legend_elements, fontsize=10)


def create_example_scenario():
    """
    Create an example scenario similar to Experiment 1 from the paper.
    4 leaders in diamond formation, 2 followers, with obstacles.
    """

    # Create agents
    # Leaders (with obstacle detection)
    leader_positions = [
        [1.0, 3.0],  # L1
        [3.0, 2.0],  # L2
        [1.0, 1.0],  # L3
        [2.0, 0.0],  # L4
    ]

    # Followers (without obstacle detection)
    follower_positions = [
        [1.5, 2.0],  # F1
        [2.0, 1.5],  # F2
    ]

    agents = []
    agent_id = 0

    # Create leader agents
    for pos in leader_positions:
        agents.append(Agent(agent_id, pos, is_leader=True))
        agent_id += 1

    # Create follower agents
    for pos in follower_positions:
        agents.append(Agent(agent_id, pos, is_leader=False))
        agent_id += 1

    # Virtual leader
    virtual_leader = VirtualLeader(position=[2.0, 2.0], goal=[8.0, 5.0])

    # Adjacency matrix (communication graph)
    # Structure: Leaders connect to virtual leader and each other
    # Followers connect to leaders
    N = len(agents) + 1  # +1 for virtual leader
    adjacency_matrix = np.zeros((N, N))

    # Leaders form a connected graph
    # L1 ↔ L2, L2 ↔ L3, L3 ↔ L4, L4 ↔ L1
    adjacency_matrix[0, 1] = adjacency_matrix[1, 0] = 1.0  # L1-L2
    adjacency_matrix[1, 2] = adjacency_matrix[2, 1] = 1.0  # L2-L3
    adjacency_matrix[2, 3] = adjacency_matrix[3, 2] = 1.0  # L3-L4
    adjacency_matrix[3, 0] = adjacency_matrix[0, 3] = 1.0  # L4-L1

    # All leaders connect to virtual leader (index 6)
    for i in range(4):
        adjacency_matrix[i, 6] = 1.0

    # Followers connect to leaders
    adjacency_matrix[4, 0] = adjacency_matrix[4, 1] = 1.0  # F1 to L1, L2
    adjacency_matrix[5, 2] = adjacency_matrix[5, 3] = 1.0  # F2 to L3, L4

    # Formation offsets (diamond shape, centered at virtual leader)
    # Diamond with 1m sides
    formation_offsets = {
        (0, 6): np.array([0.0, 0.5]),   # L1 above virtual leader
        (1, 6): np.array([0.5, 0.0]),   # L2 right
        (2, 6): np.array([0.0, -0.5]),  # L3 below
        (3, 6): np.array([-0.5, 0.0]),  # L4 left
        (0, 1): np.array([-0.5, 0.5]),
        (1, 2): np.array([0.5, 0.5]),
        (2, 3): np.array([0.5, -0.5]),
        (3, 0): np.array([-0.5, -0.5]),
    }

    # Create controller
    controller = FormationContainmentController(
        agents=agents,
        virtual_leader=virtual_leader,
        adjacency_matrix=adjacency_matrix,
        formation_offsets=formation_offsets,
        T=20.0,  # Fixed convergence time
        alpha=0.05,
        beta=0.4,
        h=5.0
    )

    # Obstacles (choke point and obstacle)
    obstacles = [
        np.array([4.0, 2.0]),
        np.array([4.0, 3.5]),
        np.array([6.0, 4.0]),
    ]

    # Motion planner
    motion_planner = MotionPlanner(
        virtual_leader=virtual_leader,
        goal=[8.0, 5.0],
        obstacles=obstacles,
        ka=1.0,
        kb=2.0,
        Q_star=1.5,
        d_star_g=0.3,
        rs=1.0,
        P_star=0.1
    )

    # Create simulation
    simulation = Simulation(
        agents=agents,
        virtual_leader=virtual_leader,
        controller=controller,
        motion_planner=motion_planner,
        obstacles=obstacles,
        arena_size=(10, 6)
    )

    return simulation


if __name__ == "__main__":
    print("=" * 60)
    print("Fixed-Time Formation-Containment Control Simulation")
    print("Based on: Su et al. (2024) - IEEE TVT")
    print("=" * 60)
    print()

    # Create simulation scenario
    print("Creating simulation scenario...")
    sim = create_example_scenario()

    print(f"Number of agents: {len(sim.agents)}")
    print(f"  Leaders: {len(sim.controller.leaders)}")
    print(f"  Followers: {len(sim.controller.followers)}")
    print(f"Fixed convergence time: {sim.controller.T} seconds")
    print(f"Simulation time step: {sim.dt} seconds")
    print()

    # Run simulation
    print("Running simulation...")
    total_steps = int(50.0 / sim.dt)  # 50 seconds

    for step in range(total_steps):
        sim.step()

        # Print progress
        if step % 200 == 0:
            print(f"  Time: {sim.time:.1f}s / 50.0s", end='\r')

    print(f"  Time: {sim.time:.1f}s / 50.0s")
    print("Simulation completed!")
    print()

    # Display results
    print("Final Metrics:")
    print(f"  Final formation error: {sim.formation_errors[-1]:.4f}")
    print(f"  Final containment error: {sim.containment_errors[-1]:.4f}")
    print(f"  Virtual leader reached goal: {np.linalg.norm(sim.virtual_leader.position - sim.motion_planner.goal) < 0.5}")
    print()

    # Plot results
    print("Generating plots...")
    sim.plot_results()
    print("Results saved to: formation_control_results.png")
    print()
    print("Simulation complete!")
