"""
Artificial Potential Fields (APF) Implementation
Based on Khatib, O., "Real-Time Obstacle Avoidance for Manipulators and Mobile Robots"
International Journal of Robotics Research, 5(1), 1986

This script visualizes 2D artificial potential fields with:
- Attractive potential from goal
- Repulsive potential from obstacles
- Vector field visualization
- Path planning from start to goal
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from scipy.integrate import odeint

# ============================================================
# GLOBAL PARAMETERS - MODIFY THESE TO CHANGE BEHAVIOR
# ============================================================

# Attractive potential gain (higher = stronger attraction to goal)
K_ATTR = 0.1

# Repulsive potential gain (higher = stronger repulsion from obstacles)
K_REP = 0.3

# Distance of influence for obstacle repulsion
RHO_0 = 0.8

# Start and goal positions
START_POS = np.array([0.5, 0.5])
GOAL_POS = np.array([9.0, 9.0])

# Grid parameters for visualization
GRID_MIN = 0.0
GRID_MAX = 10.0
GRID_RESOLUTION = int(GRID_MAX * 20)  # Number of grid points per dimension
ARROW_DENSITY_RATIO = int(GRID_RESOLUTION / 50)  # Display 1 arrow per N grid points (reduces arrow clutter)

# Path following parameters
PATH_STEP_SIZE = 0.05
PATH_MAX_STEPS = 5000
PATH_GOAL_TOLERANCE = 0.1

# Robot dynamics parameters
ROBOT_MASS = 0.8  # Vehicle mass (higher = more inertia, smoother motion)
ROBOT_DAMPING_COEFF = 1.1  # Velocity-dependent damping (higher = more viscous friction, 0.1-0.5 typical)

# ============================================================
# OBSTACLE DEFINITIONS
# ============================================================

class Obstacle2D:
    """Base class for 2D obstacles"""
    def distance(self, x, y):
        """Return signed distance from point (x, y) to obstacle.
        Positive distance = outside obstacle
        """
        raise NotImplementedError
    
    def plot(self, ax, color='red', alpha=0.3):
        """Plot the obstacle on matplotlib axis"""
        raise NotImplementedError

class CircleObstacle(Obstacle2D):
    """Circular obstacle"""
    def __init__(self, center, radius):
        self.cx, self.cy = center
        self.r = radius
    
    def distance(self, x, y):
        """Euclidean distance to circle boundary"""
        return np.sqrt((x - self.cx)**2 + (y - self.cy)**2) - self.r
    
    def plot(self, ax, color='red', alpha=0.3):
        circle = patches.Circle((self.cx, self.cy), self.r, 
                               color=color, alpha=alpha, zorder=5)
        ax.add_patch(circle)

class RectangleObstacle(Obstacle2D):
    """Axis-aligned rectangular obstacle with optional rotation"""
    def __init__(self, center, width, height, angle=0):
        """
        center: (x, y) tuple for rectangle center
        width: width of rectangle
        height: height of rectangle
        angle: rotation angle in degrees (0-180), counterclockwise around center
        """
        self.cx, self.cy = center
        self.w = width
        self.h = height
        self.angle = angle  # Store in degrees for plotting
        self.angle_rad = np.radians(angle)  # Convert to radians for computations
    
    def distance(self, x, y):
        """Distance to rotated rectangle boundary"""
        # Translate to rectangle center
        dx_rel = x - self.cx
        dy_rel = y - self.cy
        
        # Rotate back to rectangle's local frame (negative rotation to undo the rotation)
        cos_a = np.cos(-self.angle_rad)
        sin_a = np.sin(-self.angle_rad)
        
        dx_local = cos_a * dx_rel - sin_a * dy_rel
        dy_local = sin_a * dx_rel + cos_a * dy_rel
        
        # Compute distance in local frame (as if axis-aligned)
        dx = np.abs(dx_local) - self.w / 2
        dy = np.abs(dy_local) - self.h / 2
        dx = np.maximum(dx, 0)
        dy = np.maximum(dy, 0)
        return np.sqrt(dx**2 + dy**2)
    
    def plot(self, ax, color='red', alpha=0.3):
        from matplotlib.transforms import Affine2D
        
        # Create rectangle centered at origin
        rect = patches.Rectangle((-self.w/2, -self.h/2), 
                                self.w, self.h,
                                color=color, alpha=alpha, zorder=5)
        
        # Create rotation+translation transform around the center point
        t = Affine2D().rotate_deg_around(self.cx, self.cy, self.angle) + ax.transData
        rect.set_transform(t)
        
        # Set the position so the rectangle center aligns with (cx, cy)
        rect.set_xy((self.cx - self.w/2, self.cy - self.h/2))
        
        ax.add_patch(rect)

class PolygonObstacle(Obstacle2D):
    """Polygonal obstacle (convex or concave)"""
    def __init__(self, vertices):
        """
        vertices: list of (x, y) tuples defining the polygon in order
        """
        self.vertices = np.array(vertices)
    
    def distance(self, x, y):
        """Distance to polygon (simplified: distance to nearest edge)"""
        min_dist = np.inf
        vertices = self.vertices
        
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]
            
            # Distance from point to line segment
            dist = self._point_to_segment_distance(np.array([x, y]), v1, v2)
            min_dist = np.minimum(min_dist, dist)
        
        return min_dist
    
    def _point_to_segment_distance(self, p, a, b):
        """Distance from point p to line segment ab"""
        # Vector from a to b
        ab = b - a
        # Vector from a to p
        ap = p - a
        
        # Parameter t for projection
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        
        # Closest point on segment
        closest = a + t[:, np.newaxis] if isinstance(t, np.ndarray) else a + t * ab
        if isinstance(t, np.ndarray):
            return np.linalg.norm(p - closest, axis=1)
        else:
            return np.linalg.norm(p - closest)
    
    def plot(self, ax, color='red', alpha=0.3):
        polygon = patches.Polygon(self.vertices, color=color, 
                                 alpha=alpha, zorder=5)
        ax.add_patch(polygon)


# Define obstacles
OBSTACLES_REGULAR = [
    CircleObstacle(center=(3.0, 7.0), radius=0.8),
    CircleObstacle(center=(7.0, 3.0), radius=0.7),
    RectangleObstacle(center=(5.0, 5.0), width=1.5, height=0.6, angle=0),
    
    # Horseshoe obstacle pointed down
    # PolygonObstacle(vertices=[(2.0, 4.0), (4.0, 4.0), (4.0, 2.0), (3.5, 2.0), (3.5, 3.5), (2.5, 3.5), (2.5, 2.0), (2.0, 2.0)])
    
    # Horseshoe obstacle rotated 45 degrees clockwise
    # PolygonObstacle(vertices=[(3.09, 4.38), (4.50, 2.96), (3.09, 1.55), (2.73, 1.90), (3.80, 2.96), (3.09, 3.67), (2.03, 2.61), (1.67, 2.96)]),
]

OBSTACLES_MINIMA = [
    CircleObstacle(center=(3.0, 7.0), radius=0.8),
    CircleObstacle(center=(7.0, 3.0), radius=0.7),
    RectangleObstacle(center=(5.0, 5.0), width=1.5, height=0.6, angle=0),
    
    # Horseshoe obstacle rotated 45 degrees clockwise
    PolygonObstacle(vertices=[(3.09, 4.38), (4.50, 2.96), (3.09, 1.55), (2.73, 1.90), (3.80, 2.96), (3.09, 3.67), (2.03, 2.61), (1.67, 2.96)]),
]

OBSTACLES_NARROWCORRIDOR = [
    CircleObstacle(center=(2.0, 8.0), radius=1.0),
    CircleObstacle(center=(7.0, 2.0), radius=1.5),
    RectangleObstacle(center=(4, 6), width=3, height=0.9, angle=45),
    RectangleObstacle(center=(5, 4), width=3, height=0.9, angle=45),
]


# ============================================================
# POTENTIAL FIELD FUNCTIONS
# ============================================================

def attractive_potential(x, y, goal_x, goal_y, k_attr):
    """
    Attractive potential from goal
    U_att = (1/2) * k_attr * |q - q_goal|^2
    """
    dist_sq = (x - goal_x)**2 + (y - goal_y)**2
    return 0.5 * k_attr * dist_sq


def attractive_force(x, y, goal_x, goal_y, k_attr):
    """
    Attractive force (negative gradient of attractive potential)
    F_att = -k_attr * (q - q_goal)
    """
    fx = -k_attr * (x - goal_x)
    fy = -k_attr * (y - goal_y)
    return fx, fy


def repulsive_potential(distance, k_rep, rho_0, epsilon=1e-6):
    """
    Repulsive potential from obstacles
    U_rep = (1/2) * k_rep * (1/d - 1/rho_0)^2  if d <= rho_0
    U_rep = 0                                     if d > rho_0
    """
    potential = np.zeros_like(distance)
    inside = distance <= rho_0
    # Clamp distance to avoid division by zero
    distance_clamped = np.maximum(distance[inside], epsilon)
    potential[inside] = 0.5 * k_rep * (1/distance_clamped - 1/rho_0)**2
    return potential


def repulsive_force(x, y, obstacle, k_rep, rho_0, epsilon=1e-6):
    """
    Repulsive force (negative gradient of repulsive potential)
    F_rep = k_rep * (1/rho_0 - 1/d) * (1/d^2) * grad(d)
    
    grad(d) is the unit vector pointing away from obstacle
    """
    distance = obstacle.distance(x, y)
    
    # Add small epsilon to avoid division by zero
    distance = np.maximum(distance, epsilon)
    
    # Compute repulsive potential
    inside = distance <= rho_0
    
    # Force magnitude
    force_mag = np.zeros_like(distance)
    force_mag[inside] = k_rep * (1/distance[inside] - 1/rho_0) / (distance[inside]**2)
    
    # Compute gradient of distance (unit vector away from obstacle)
    delta = 0.1  # Smaller step for accurate gradient at exact point location
    dist_x_plus = obstacle.distance(x + delta, y)
    dist_y_plus = obstacle.distance(x, y + delta)
    
    grad_x = (dist_x_plus - distance) / delta
    grad_y = (dist_y_plus - distance) / delta
    
    # Normalize gradient
    grad_norm = np.sqrt(grad_x**2 + grad_y**2) + epsilon
    grad_x = grad_x / grad_norm
    grad_y = grad_y / grad_norm
    
    # Force components
    fx = force_mag * grad_x
    fy = force_mag * grad_y
    
    return fx, fy


# ============================================================
# PATH PLANNING
# ============================================================
def compute_path_from_field(start, goal, X, Y, Fx, Fy,
                                  step_size=PATH_STEP_SIZE, max_steps=PATH_MAX_STEPS,
                                  goal_tolerance=PATH_GOAL_TOLERANCE,
                                  mass=ROBOT_MASS, damping_coeff=ROBOT_DAMPING_COEFF,
                                  grid_min=GRID_MIN, grid_max=GRID_MAX):
    """
    Compute path from start to goal by following the pre-computed force field grid.
    
    Uses improved physics dynamics with:
    - Velocity-dependent damping (more intuitive and less sensitive than multiplicative)
    - Semi-implicit Euler integration for better stability
    - Adaptive step sizing based on local force magnitude
    - Bilinear interpolation for smooth force field sampling
    
    Args:
        start: starting position [x, y]
        goal: goal position [x, y]
        X, Y: grid coordinates (from np.meshgrid)
        Fx, Fy: pre-computed force fields at grid points
        step_size: base integration step size
        max_steps: maximum number of steps
        goal_tolerance: distance to goal for convergence
        mass: robot mass (kg, affects inertia)
        damping_coeff: velocity-dependent damping coefficient (0.1-0.5 typical)
        grid_min, grid_max: grid bounds for interpolation boundary checking
    
    Returns:
        path: numpy array of [N, 2] waypoints
    """
    from scipy.interpolate import RectBivariateSpline
    
    # Create interpolation functions for force field
    # RectBivariateSpline expects 1D x, 1D y coordinates
    x_grid = X[0, :]  # First row contains all x values
    y_grid = Y[:, 0]  # First column contains all y values
    
    # Create 2D splines for force interpolation
    fx_interp = RectBivariateSpline(y_grid, x_grid, Fx, kx=1, ky=1)
    fy_interp = RectBivariateSpline(y_grid, x_grid, Fy, kx=1, ky=1)
    
    path = [start.copy()]
    pos = start.copy()
    vel = np.array([0.0, 0.0])
    
    for step in range(max_steps):
        # Clamp position to grid bounds to avoid extrapolation
        px_clamped = np.clip(pos[0], grid_min, grid_max)
        py_clamped = np.clip(pos[1], grid_min, grid_max)
        
        # Sample force field at current position using interpolation
        # Note: RectBivariateSpline expects (y, x) order
        fx = float(fx_interp(py_clamped, px_clamped)[0, 0])
        fy = float(fy_interp(py_clamped, px_clamped)[0, 0])
        
        force = np.array([fx, fy])
        force_norm = np.sqrt(fx**2 + fy**2)
        
        # Check if force is negligible
        if force_norm < 1e-6:
            break
        
        # Adaptive step size: reduce step size in high-force regions for stability
        # This prevents overshooting in narrow corridors
        adaptive_dt = step_size / (1.0 + force_norm * step_size * 0.1)
        
        # Velocity-dependent damping force: F_damp = -c * v
        # This is more physical and easier to tune than multiplicative damping
        damping_force = -damping_coeff * vel
        
        # Compute total acceleration: a = (F_external + F_damping) / m
        total_force = force + damping_force
        accel = total_force / mass
        
        # Semi-implicit Euler integration (update velocity before position)
        # This is more stable than standard Euler integration
        vel = vel + accel * adaptive_dt
        pos = pos + vel * adaptive_dt
        path.append(pos.copy())
        
        # Check if goal reached and velocity settled
        dist_to_goal = np.linalg.norm(pos - goal)
        vel_magnitude = np.linalg.norm(vel)
        if dist_to_goal < goal_tolerance and vel_magnitude < 0.05:
            path.append(goal.copy())
            break
    
    return np.array(path)

# ============================================================
# VISUALIZATION
# ============================================================

def visualize_apf(obstacles, start=START_POS, goal=GOAL_POS,
                 k_attr=K_ATTR, k_rep=K_REP, rho_0=RHO_0,
                 grid_min=GRID_MIN, grid_max=GRID_MAX, 
                 grid_res=GRID_RESOLUTION, figsize=(10, 10)):
    """
    Create comprehensive visualization of the artificial potential field
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create grid of points
    x = np.linspace(grid_min, grid_max, grid_res)
    y = np.linspace(grid_min, grid_max, grid_res)
    X, Y = np.meshgrid(x, y)
    
    # Compute potential field
    U = np.zeros_like(X, dtype=float)
    Fx = np.zeros_like(X, dtype=float)
    Fy = np.zeros_like(Y, dtype=float)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            xi, yi = X[i, j], Y[i, j]
            
            # Attractive potential and force
            u_att = attractive_potential(xi, yi, goal[0], goal[1], k_attr)
            fx_att, fy_att = attractive_force(xi, yi, goal[0], goal[1], k_attr)
            
            U[i, j] = u_att
            Fx[i, j] = fx_att
            Fy[i, j] = fy_att
            
            # Repulsive potential and forces
            for obstacle in obstacles:
                dist = obstacle.distance(xi, yi)
                u_rep = repulsive_potential(np.array([dist]), k_rep, rho_0)[0]
                fx_rep, fy_rep = repulsive_force(xi, yi, obstacle, k_rep, rho_0)
                
                U[i, j] += u_rep
                Fx[i, j] += fx_rep
                Fy[i, j] += fy_rep
    
    # Plot potential field contours
    contour_levels = np.linspace(U.min(), np.percentile(U, 95), 20)
    contour = ax.contour(X, Y, U, levels=contour_levels, colors='gray', 
                         alpha=0.3, linewidths=0.5)
    
    # Plot vector field
    # Normalize vectors for better visualization
    F_norm = np.sqrt(Fx**2 + Fy**2)
    F_norm = np.maximum(F_norm, 1e-6)
    Fx_norm = Fx / F_norm
    Fy_norm = Fy / F_norm
    
    # Subsample for arrow density
    skip = ARROW_DENSITY_RATIO
    X_arrows = X[::skip, ::skip]
    Y_arrows = Y[::skip, ::skip]
    Fx_arrows = Fx_norm[::skip, ::skip]
    Fy_arrows = Fy_norm[::skip, ::skip]
    F_arrows = F_norm[::skip, ::skip]
    
    ax.quiver(X_arrows, Y_arrows, Fx_arrows, Fy_arrows, F_arrows, cmap='viridis', 
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
               label='Planned Path', alpha=0.7, zorder=8)
    
    # Formatting
    ax.set_xlim(grid_min, grid_max)
    ax.set_ylim(grid_min, grid_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Artificial Potential Field (Khatib, 1986)')
    ax.legend(loc='upper left', labelspacing=1.2, handlelength=1.5)
    
    plt.tight_layout()
    return fig, ax

def save_visualization(fig, obstacles, filename = None):
    """Save the visualization to a file"""
    # Mapping of obstacle configuration ids to their abbreviations
    obstacle_name_map = {
        id(OBSTACLES_REGULAR): 'reg',
        id(OBSTACLES_MINIMA): 'min',
        id(OBSTACLES_NARROWCORRIDOR): 'nc',
    }
    
    # Deduce which global obstacle configuration was passed
    obstacles_abbrev = obstacle_name_map.get(id(obstacles), 'custom')
    
    if filename is None:
        # Will generate a name based on the parameters used in the visualization
        filename = f'apf_kattr{K_ATTR}_krep{K_REP}_rho0{RHO_0}_res{GRID_RESOLUTION}_mass{ROBOT_MASS}_dampcoeff{ROBOT_DAMPING_COEFF}_{obstacles_abbrev}.png'
    
    fig.savefig(f'imgs/{filename}', dpi=300, bbox_inches='tight')

# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == '__main__':

    obstacles  = OBSTACLES_NARROWCORRIDOR # Change as needed
    # Create visualization
    fig, ax = visualize_apf(
        obstacles=obstacles,
        start=START_POS,
        goal=GOAL_POS,
        k_attr=K_ATTR,
        k_rep=K_REP,
        rho_0=RHO_0,
        grid_min=GRID_MIN,
        grid_max=GRID_MAX,
        grid_res=GRID_RESOLUTION
    )
    
    plt.show()
    save_visualization(fig, obstacles, None)