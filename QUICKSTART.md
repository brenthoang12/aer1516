# Quick Start Guide

## Installation (First Time Only)

```bash
# Navigate to project directory
cd "/Users/brenthoang/Documents/University Material/Winter 2026/AER 1516/Paper Presentation/implementation"

# Virtual environment is already created!
# Just activate it:
source formation_control_env/bin/activate

# Verify installation
python -c "import numpy, matplotlib, scipy; print('All packages installed!')"
```

## Running the Simulation

```bash
# Make sure environment is activated
source formation_control_env/bin/activate

# Run simulation
python formation_control.py

# You'll see:
# - Real-time progress updates
# - Final metrics (errors, goal reached)
# - Generated plot: formation_control_results.png
```

## What You'll Get

### Console Output
```
============================================================
Fixed-Time Formation-Containment Control Simulation
Based on: Su et al. (2024) - IEEE TVT
============================================================

Creating simulation scenario...
Number of agents: 6
  Leaders: 4
  Followers: 2
Fixed convergence time: 20.0 seconds

Running simulation...
  Time: 50.0s / 50.0s
Simulation completed!

Final Metrics:
  Final formation error: 0.3685
  Final containment error: 0.0043
  Virtual leader reached goal: True

Results saved to: formation_control_results.png
```

### Generated Plot (4 Panels)

1. **Final Formation**: Where agents ended up
2. **Formation Error**: How leaders converged to formation
3. **Containment Error**: How followers converged to safe region
4. **Trajectories**: Complete paths taken by all agents

## Understanding the Results

### Good Results Look Like:
- ✅ Formation error drops to < 0.5 by T=20s
- ✅ Containment error drops to < 0.1 by T=20s
- ✅ Virtual leader reaches goal (distance < 0.5m)
- ✅ Smooth trajectories with no collisions

### Common Issues:

**Agents diverge or oscillate wildly**
→ Reduce control gains: `alpha=0.02, beta=0.2`

**Doesn't converge by T=20s**
→ Increase beta gain: `beta=0.6` or increase T: `T=30.0`

**Collides with obstacles**
→ Increase repulsive gain: `kb=3.0` or threshold: `Q_star=2.0`

## Quick Modifications

### Change Formation Shape

Open `formation_control.py`, find `create_example_scenario()`, modify:

```python
# Current: Diamond formation
formation_offsets = {
    (0, 6): np.array([0.0, 0.5]),   # L1 above
    (1, 6): np.array([0.5, 0.0]),   # L2 right
    (2, 6): np.array([0.0, -0.5]),  # L3 below
    (3, 6): np.array([-0.5, 0.0]),  # L4 left
}

# Change to: Square formation
formation_offsets = {
    (0, 6): np.array([-0.5, 0.5]),   # L1 top-left
    (1, 6): np.array([0.5, 0.5]),    # L2 top-right
    (2, 6): np.array([0.5, -0.5]),   # L3 bottom-right
    (3, 6): np.array([-0.5, -0.5]),  # L4 bottom-left
}
```

### Change Goal Position

```python
# Find this line in create_example_scenario():
virtual_leader = VirtualLeader(position=[2.0, 2.0], goal=[8.0, 5.0])

# Change to:
virtual_leader = VirtualLeader(position=[2.0, 2.0], goal=[9.0, 6.0])
```

### Add More Obstacles

```python
# Find obstacles list:
obstacles = [
    np.array([4.0, 2.0]),
    np.array([4.0, 3.5]),
    np.array([6.0, 4.0]),
]

# Add more:
obstacles = [
    np.array([4.0, 2.0]),
    np.array([4.0, 3.5]),
    np.array([6.0, 4.0]),
    np.array([5.0, 1.5]),  # New
    np.array([7.0, 4.5]),  # New
]
```

### Change Fixed Time

```python
# Find in create_example_scenario():
controller = FormationContainmentController(
    # ...
    T=20.0,  # Change this (seconds)
    # ...
)
```

## File Overview

```
implementation/
├── formation_control.py        ← MAIN FILE (run this)
├── requirements.txt            ← Dependencies list
├── README.md                   ← Full documentation
├── IMPLEMENTATION_SUMMARY.md   ← Technical details
├── QUICKSTART.md              ← This file
└── formation_control_results.png  ← Output (after running)
```

## Next Steps

1. **Run the default simulation** to see it work
2. **Read README.md** for full documentation
3. **Read IMPLEMENTATION_SUMMARY.md** for theory
4. **Modify parameters** to experiment
5. **Create your own scenarios** (see README)

## Deactivate Environment

When done:
```bash
deactivate
```

## Key Equations Implemented

### Control Law (Theorem 2)
```
uᵢ = -1/γᵢ[(α + β*μ̇/μ)ξᵢ + (α + β*μ̇/μ)ζᵢ - Σⱼ aᵢⱼv̇ⱼ]
```

Where:
- `ξᵢ` = position error
- `ζᵢ` = velocity error
- `μ(t) = (T/(T-t))^h` = time-varying urgency function

### Result
Guarantees formation and containment by time T!

## Questions?

- Check **README.md** for detailed explanations
- Check **IMPLEMENTATION_SUMMARY.md** for theory
- Refer to the paper: Su et al. (2024), IEEE TVT

## Citation

If using this code:
```
Su, Y.-H., Bhowmick, P., & Lanzon, A. (2024).
A Fixed-Time Formation-Containment Control Scheme for
Multi-Agent Systems With Motion Planning: Applications
to Quadcopter UAVs. IEEE Transactions on Vehicular
Technology, 73(7), 9495-9507.
```

---

**Have fun experimenting with multi-agent formation control! 🚁✨**
