"""
Safe Artificial Potential Field (Safe APF) Implementation
A novel local path planning algorithm maintaining safe distance from obstacles.

Extends the Khatib APF with:
- Vortex repulsive field for improved navigation around obstacles
- SAFE rotation matrix for blending attractive and repulsive fields
- Better handling of local minima through tangential guidance
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Import functions from Khatib APF implementation
from khatib_apf_2d import (
    attractive_potential,
    attractive_force,
    repulsive_potential,
    repulsive_force,
    compute_path_from_field,
    save_visualization,
    # Parameters
    K_ATTR,
    K_REP,
    RHO_0,
    START_POS,
    GOAL_POS,
    GRID_MIN,
    GRID_MAX,
    GRID_RESOLUTION,
    ARROW_DENSITY_RATIO,
    ROBOT_MASS,
    ROBOT_DAMPING_COEFF,
    OBSTACLES_REGULAR,
    OBSTACLES_MINIMA,
    OBSTACLES_NARROWCORRIDOR
)


# ============================================================
# SAFE APF PARAMETERS
# ============================================================

# Vortex field gain (higher = stronger tangential forces)
K_VORTEX = 5

# Safe distance - minimum distance from obstacle (pure repulsive zone)
D_SAFE = 0.3

# Desired vortex distance - where vortex influence is maximum
D_VORT = 3.0

# Rotation blending factor (0 = no rotation, 1 = full rotation)
ROTATION_GAIN = 1.0


# ============================================================
# VORTEX REPULSIVE FIELD FUNCTIONS
# ============================================================
'''
def vortex_repulsive_potential(distance, k_vortex, rho_0, epsilon=1e-6):
    """
    Vortex repulsive potential field.
    Creates a tangential component to help navigate around obstacles.
    
    U_vortex = k_vortex * distance * (1/d - 1/rho_0)  if d <= rho_0
    U_vortex = 0                                         if d > rho_0
    
    Args:
        distance: distance from obstacle(s)
        k_vortex: vortex field strength
        rho_0: influence distance
        epsilon: small value to avoid division by zero
    
    Returns:
        vortex potential at each point
    """
    potential = np.zeros_like(distance)
    inside = distance <= rho_0
    
    distance_clamped = np.maximum(distance[inside], epsilon)
    # Vortex potential increases with distance from obstacle boundary
    potential[inside] = k_vortex * distance_clamped * (1/distance_clamped - 1/rho_0)
    
    return potential

'''

def vortex_window_function(distance, d_safe, d_vort):
    """
    Window function γ(d) for vortex field application.
    
    Creates a symmetric band around d_vort where vortex influence is strong.
    - At d <= d_safe: γ = 0 (pure repulsive zone)
    - At d = d_vort: γ = 1 (maximum vortex influence)
    - At d >= 2*d_vort - d_safe: γ = 0 (vortex influence fades out)
    
    Uses a triangular window for smooth transitions.
    
    Args:
        distance: distance from obstacle
        d_safe: safe distance (inner boundary)
        d_vort: vortex center distance (peak of bell curve)
    
    Returns:
        gamma: window function value (0 to 1)
    """
    d_outer = 2 * d_vort - d_safe  # Outer boundary of vortex zone
    
    gamma = np.zeros_like(distance)
    
    # Inner slope: d_safe < d < d_vort increases from 0 to 1
    inner_region = (distance > d_safe) & (distance < d_vort)
    gamma[inner_region] = (distance[inner_region] - d_safe) / (d_vort - d_safe)
    
    # At peak: d = d_vort 
    at_peak = np.isclose(distance, d_vort)
    gamma[at_peak] = 1.0
    
    # Outer slope: d_vort < d < d_outer decreases from 1 to 0
    outer_region = (distance >= d_vort) & (distance < d_outer)
    gamma[outer_region] = (d_outer - distance[outer_region]) / (d_outer - d_vort)
    
    # Clamp to [0, 1]
    gamma = np.clip(gamma, 0.0, 1.0)
    
    return gamma


def vortex_repulsive_force(x, y, obstacle, k_vortex, d_safe, d_vort, epsilon=1e-6):
    """
    Vortex repulsive force - creates tangential forces in a band around obstacles.
    
    The vortex force is only applied strongly in a specific distance band:
      d_safe < d < 2*d_vort - d_safe
    
    Where:
    - d_safe: pure repulsive zone (minimum safe distance)
    - d_vort: desired sliding distance (peak vortex influence)
    - The band is symmetric around d_vort
    
    Args:
        x, y: position(s)
        obstacle: Obstacle2D object
        k_vortex: vortex field strength
        d_safe: safe distance from obstacle
        d_vort: vortex band center - where vortex influence peaks
        epsilon: small value to avoid division by zero
    
    Returns:
        fx_vortex, fy_vortex: tangential force components
    """
    distance = obstacle.distance(x, y)
    distance = np.maximum(distance, epsilon)
    
    # Compute gradient of distance (points away from obstacle)
    delta = 0.1
    dist_x_plus = obstacle.distance(x + delta, y)
    dist_y_plus = obstacle.distance(x, y + delta)
    
    grad_x = (dist_x_plus - distance) / delta
    grad_y = (dist_y_plus - distance) / delta
    
    # Normalize gradient to get unit vector away from obstacle
    grad_norm = np.sqrt(grad_x**2 + grad_y**2) + epsilon
    grad_x_norm = grad_x / grad_norm
    grad_y_norm = grad_y / grad_norm
    
    # Perpendicular vector (tangent) - rotate 90 degrees counterclockwise
    # If normal is (nx, ny), tangent is (-ny, nx)
    tangent_x = -grad_y_norm
    tangent_y = grad_x_norm
    
    # Compute window function for vortex zone
    gamma = vortex_window_function(distance, d_safe, d_vort)
    
    # Vortex force magnitude: weighted by window function and distance from d_vort
    # Force is stronger the more we deviate from the desired sliding distance
    distance_deviation = np.abs(distance - d_vort)
    
    # Base repulsive-like magnitude modulated by window function
    force_mag = k_vortex * gamma * (1 / distance - 1 / d_vort)
    force_mag = np.maximum(force_mag, 0.0)  # Only positive (outward) forces
    
    # Apply vortex force in tangential direction
    fx_vortex = force_mag * tangent_x
    fy_vortex = force_mag * tangent_y
    
    return fx_vortex, fy_vortex


# ============================================================
# SAFE PATH PLANNING
# ============================================================

# ============================================================
# SAFE APF VISUALIZATION
# ============================================================

def visualize_safe_apf(obstacles, start=START_POS, goal=GOAL_POS,
                       k_attr=K_ATTR, k_rep=K_REP, rho_0=RHO_0,
                       k_vortex=K_VORTEX, d_safe=D_SAFE, d_vort=D_VORT,
                       rotation_gain=ROTATION_GAIN,
                       grid_min=GRID_MIN, grid_max=GRID_MAX, 
                       grid_res=GRID_RESOLUTION, figsize=(12, 10)):
    """
    Visualize SAFE APF with potential field, force vectors, and planned path.
    
    Args:
        obstacles: list of Obstacle2D objects
        start: start position
        goal: goal position
        k_attr: attractive force gain
        k_rep: repulsive force gain
        rho_0: repulsive influence distance
        k_vortex: vortex field strength
        rotation_gain: SAFE rotation blending factor
        grid_min, grid_max: grid bounds
        grid_res: grid resolution
        figsize: figure size
    
    Returns:
        fig, ax: matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create grid
    x = np.linspace(grid_min, grid_max, grid_res)
    y = np.linspace(grid_min, grid_max, grid_res)
    X, Y = np.meshgrid(x, y)
    
    # Compute potential and force fields
    # U = np.zeros_like(X, dtype=float)
    Fx = np.zeros_like(X, dtype=float)
    Fy = np.zeros_like(Y, dtype=float)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            xi, yi = X[i, j], Y[i, j]
            
            # Attractive potential and force
            # u_att = attractive_potential(xi, yi, goal[0], goal[1], k_attr)
            fx_att, fy_att = attractive_force(xi, yi, goal[0], goal[1], k_attr)
            
            # # U[i, j] = u_att
            Fx[i, j] = fx_att
            Fy[i, j] = fy_att
            
            # Repulsive, vortex, and SAFE forces
            for obstacle in obstacles:
                dist = obstacle.distance(xi, yi)
                # u_rep = repulsive_potential(np.array([dist]), k_rep, rho_0)[0]
                fx_rep, fy_rep = repulsive_force(xi, yi, obstacle, k_rep, rho_0)
                fx_vortex, fy_vortex = vortex_repulsive_force(xi, yi, obstacle, k_vortex, d_safe, d_vort)
                
                # U[i, j] += u_rep
                Fx[i, j] += fx_rep + fx_vortex
                Fy[i, j] += fy_rep + fy_vortex
            
    
    # # Plot potential contours
    # contour_levels = np.linspace(U.min(), np.percentile(U, 95), 20)
    # contour = ax.contour(X, Y, U, levels=contour_levels, colors='gray', 
    #                      alpha=0.3, linewidths=0.5)
    
    # Normalize and plot force vectors
    F_norm = np.sqrt(Fx**2 + Fy**2)
    F_norm = np.maximum(F_norm, 1e-6)
    Fx_norm = Fx / F_norm
    Fy_norm = Fy / F_norm
    
    # Subsample for arrow visualization
    skip = ARROW_DENSITY_RATIO
    X_arrows = X[::skip, ::skip]
    Y_arrows = Y[::skip, ::skip]
    Fx_arrows = Fx_norm[::skip, ::skip]
    Fy_arrows = Fy_norm[::skip, ::skip]
    F_arrows = F_norm[::skip, ::skip]
    
    ax.quiver(X_arrows, Y_arrows, Fx_arrows, Fy_arrows, F_arrows, cmap='plasma', 
             alpha=0.6, scale=60, width=0.0015)
    
    # Plot obstacles
    for obstacle in obstacles:
        obstacle.plot(ax, color='red', alpha=0.4)
    
    # Plot start and goal
    ax.plot(start[0], start[1], 'go', markersize=12, label='Start', zorder=10)
    ax.plot(goal[0], goal[1], 'b*', markersize=20, label='Goal', zorder=10)
    
    # Compute and plot path following the pre-computed force field
    path = compute_path_from_field(start, goal, X, Y, Fx, Fy, 
                                         grid_min=grid_min, grid_max=grid_max)
    if len(path) > 1:
        ax.plot(path[:, 0], path[:, 1], 'g--', linewidth=2, 
               label='Path (from field)', alpha=0.7, zorder=8)
    
    # Formatting
    ax.set_xlim(grid_min, grid_max)
    ax.set_ylim(grid_min, grid_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Safe Artificial Potential Field (SAFE APF)')
    ax.legend(loc='upper left', labelspacing=1.2, handlelength=1.5)
    
    plt.tight_layout()
    return fig, ax


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == '__main__':
    # Define obstacles - same as in Khatib APF
    obstacles = OBSTACLES_NARROWCORRIDOR

    # Create visualization
    fig, ax = visualize_safe_apf(
        obstacles=obstacles,
        start=START_POS,
        goal=GOAL_POS,
        k_attr=K_ATTR,
        k_rep=K_REP,
        rho_0=RHO_0,
        k_vortex=K_VORTEX,
        d_safe=D_SAFE,
        d_vort=D_VORT,
        rotation_gain=ROTATION_GAIN,
        grid_min=GRID_MIN,
        grid_max=GRID_MAX,
        grid_res=GRID_RESOLUTION
    )
    
    plt.show()
    filename = f'safe_apf_kattr{K_ATTR}_krep{K_REP}_rho0{RHO_0}_kvor{K_VORTEX}_d_safe{D_SAFE}_d_vort{D_VORT}_res{GRID_RESOLUTION}_mass{ROBOT_MASS}_damp{ROBOT_DAMPING_COEFF}.png'
    save_visualization(fig, 'safe_apf_2d', filename)
