# Fixed-Time Formation-Containment Control Implementation

Implementation of the paper: **"A Fixed-Time Formation-Containment Control Scheme for Multi-Agent Systems With Motion Planning: Applications to Quadcopter UAVs"** by Su et al. (2024), IEEE Transactions on Vehicular Technology.

## TODO

1. The "formation_control.py" doesn't have scaling vector for choke point navigation
2. Rotation matrix needs both counterclock wise and clockwise rotation. We still need to implement angle between relative position vector between obstacle and virtual leader agent.
3. Set up experiment so that the initial formation of UAVs are random to force them to the desired formation.  
4. Go through adjacency matrix 

## Overview

This is a 2D top-down simulation of quadcopter agents performing fixed-time formation-containment control with obstacle avoidance.

### Key Features

1. **Fixed-Time Convergence**: Agents achieve formation/containment within a guaranteed fixed time T (default: 20 seconds)
2. **Formation-Containment Control**:
   - Leader agents maintain a prescribed geometric formation
   - Follower agents stay inside the convex hull (safe region) formed by leaders
3. **Motion Planning**: Virtual leader navigates to goal using Artificial Potential Fields (APF)
4. **Obstacle Avoidance**: Repulsive forces and curl-free vector fields prevent collisions

## Mathematical Foundation

### Agent Dynamics (Double Integrator)
```
ṗᵢ = vᵢ  (position derivative = velocity)
v̇ᵢ = uᵢ  (velocity derivative = control input)
```

### Formation Error (Leaders)
```
ξᵢ = Σⱼ∈Nᵢ aᵢⱼ(pᵢ - pⱼ - δᵢⱼ)
```

### Containment Error (Followers)
```
ξₖ = Σⱼ∈Nₖ aₖⱼ(pₖ - pⱼ)
```

### Fixed-Time Control Law (Theorem 2)
```
uᵢ = -1/γᵢ[(α + β*μ̇/μ)ξᵢ + (α + β*μ̇/μ)ζᵢ - Σⱼ aᵢⱼv̇ⱼ]
```

Where:
- `μ(t) = (T/(T-t))^h` is the time-varying function
- `α, β` are control gains
- `γᵢ = Σⱼ aᵢⱼ` is sum of edge weights
- `ζᵢ` is velocity consensus error

### Motion Planning
- **Attractive Potential**: Pulls toward goal
- **Repulsive Potential**: Pushes away from obstacles
- **Curl-Free Vector Field**: Prevents local minima traps

## Installation

### 1. Create Virtual Environment
```bash
python3 -m venv formation_control_env
source formation_control_env/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install numpy matplotlib scipy
```

## Usage

### Run the Simulation
```bash
python formation_control.py
```

### Expected Output
- Console progress updates
- Final metrics (formation error, containment error)
- 4-panel visualization saved as `formation_control_results.png`:
  1. Final formation configuration
  2. Formation tracking error over time
  3. Containment error over time
  4. Agent trajectories

## Code Structure

### Classes

1. **Agent**: Represents a single quadcopter
   - Attributes: position, velocity, acceleration, is_leader
   - Method: `update(dt)` - integrates dynamics

2. **VirtualLeader**: Reference trajectory generator
   - Follows motion planning algorithm
   - Provides tracking reference for all agents

3. **FormationContainmentController**: Main control algorithm
   - Implements Theorem 2 (fixed-time control law)
   - Computes formation/containment errors
   - Generates control inputs

4. **MotionPlanner**: High-level path planning
   - Artificial Potential Field (APF) method
   - Attractive force toward goal
   - Repulsive force from obstacles
   - Curl-free vector field to escape local minima

5. **Simulation**: Main simulation environment
   - Coordinates all components
   - Records data and generates plots

### Key Parameters

#### Control Parameters (FormationContainmentController)
- `T`: Fixed convergence time (seconds) - default: 20.0
- `alpha`: Control gain - default: 0.05
- `beta`: Time-varying gain - default: 0.4
- `h`: Time function exponent - default: 5.0

#### Motion Planning Parameters (MotionPlanner)
- `ka`: Attractive gain - default: 1.0
- `kb`: Repulsive gain - default: 2.0
- `Q_star`: Repulsive field threshold - default: 1.5
- `d_star_g`: Goal proximity threshold - default: 0.3
- `rs`: Safe region radius - default: 1.0
- `P_star`: Local minimum detection threshold - default: 0.1

## Customization

### Create Custom Scenario

```python
# Define agents
agents = [
    Agent(0, [1.0, 1.0], is_leader=True),
    Agent(1, [2.0, 1.0], is_leader=True),
    Agent(2, [1.5, 2.0], is_leader=False),
]

# Create virtual leader
virtual_leader = VirtualLeader(position=[1.5, 1.5], goal=[8.0, 5.0])

# Define communication graph (adjacency matrix)
# Size: (N+1) x (N+1), where N = number of agents
adjacency_matrix = np.array([
    [0, 1, 0, 1],  # Agent 0 connections
    [1, 0, 0, 1],  # Agent 1 connections
    [1, 1, 0, 0],  # Agent 2 connections
    [0, 0, 0, 0],  # Virtual leader (row not used)
])

# Define formation offsets
formation_offsets = {
    (0, 3): np.array([-0.5, 0.0]),  # Agent 0 to virtual leader
    (1, 3): np.array([0.5, 0.0]),   # Agent 1 to virtual leader
    (0, 1): np.array([-1.0, 0.0]),  # Agent 0 to Agent 1
}

# Create controller
controller = FormationContainmentController(
    agents, virtual_leader, adjacency_matrix, formation_offsets
)

# Define obstacles
obstacles = [np.array([4.0, 3.0]), np.array([5.0, 4.0])]

# Create motion planner
motion_planner = MotionPlanner(virtual_leader, goal=[8.0, 5.0], obstacles=obstacles)

# Create and run simulation
sim = Simulation(agents, virtual_leader, controller, motion_planner, obstacles)
sim.run(steps=1000)
sim.plot_results()
```

## Example Scenario

The default scenario (`create_example_scenario()`) creates:
- **4 Leaders**: Form a diamond shape
- **2 Followers**: Stay inside leader formation
- **3 Obstacles**: Create a navigation challenge
- **Goal**: Position [8.0, 5.0]
- **Fixed Time**: 20 seconds

## Interpretation of Results

### Formation Error Plot
- Should converge to near-zero by time T (20 seconds)
- Vertical red line marks the fixed convergence time
- Small oscillations in steady-state are normal due to numerical integration

### Containment Error Plot
- Measures how well followers stay inside leader convex hull
- Should also converge by time T
- Zero error = followers are on the convex hull boundary

### Trajectory Plot
- Shows paths taken by all agents
- Leaders (blue triangles) maintain formation
- Followers (orange circles) stay protected inside
- Virtual leader (green) navigates around obstacles to goal

## Theory vs Implementation Notes

### Simplifications Made
1. **2D instead of 3D**: Paper uses 3D quadcopters, we use 2D for visualization
2. **Direct dynamics control**: We directly control acceleration instead of cascaded PID
3. **Perfect state information**: Assumes perfect position/velocity measurements
4. **Simplified obstacle detection**: Omnidirectional sensing instead of limited FOV sensors

### Key Theoretical Results Implemented
- ✅ Theorem 2: Fixed-time formation-containment control
- ✅ Algorithm 1: High-level motion planning with APF
- ✅ Equation (23): Fixed-time control protocol
- ✅ Equation (20): Time-varying function μ(t)
- ✅ Equations (30-38): Attractive/repulsive potential fields

## Troubleshooting

### Issue: Agents diverge or become unstable
**Solution**: Reduce control gains `alpha` and `beta`, or increase time step `dt`

### Issue: Formation doesn't converge by time T
**Solution**: Increase `beta` gain or reduce `T` (though this defeats the fixed-time purpose)

### Issue: Agents collide with obstacles
**Solution**: Increase repulsive gain `kb` or threshold `Q_star`

### Issue: Virtual leader gets stuck in local minimum
**Solution**: Reduce `P_star` threshold to activate curl-free field earlier

## References

1. Su, Y.-H., Bhowmick, P., & Lanzon, A. (2024). A Fixed-Time Formation-Containment Control Scheme for Multi-Agent Systems With Motion Planning: Applications to Quadcopter UAVs. *IEEE Transactions on Vehicular Technology*, 73(7), 9495-9507.

2. Related Papers:
   - Formation control: Ren et al. (2007) - Information consensus in multivehicle cooperative control
   - Fixed-time control: Wang et al. (2019) - Fixed-time formation control of multirobot systems
   - APF: Chang et al. (2016) - UAV formation control design with obstacle avoidance

## License

This implementation is for educational purposes. Please cite the original paper when using this code.

## Contact

For questions about the implementation, please refer to the original paper or create an issue in the repository.
