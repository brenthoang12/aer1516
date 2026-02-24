"""
Visualization of Attractive and Repulsive Artificial Potential Fields

Visualizes the piecewise attractive potential field:

U_a(d) = { 
    (k_a / 2) * d^2,              if d <= d*_g  (quadratic near goal)
    d*_g * k_a * d - (k_a / 2) * (d*_g)^2,  if d > d*_g  (linear far from goal)
}

And the repulsive potential field:

U_b(D) = {
    (k_b / 2) * (1/D - 1/Q*)^2,   if D <= Q*  (bounded influence)
    0,                             if D > Q*   (no influence)
}

Where:
- d = distance from goal
- D = distance from obstacle
- k_a = attractive potential gain
- k_b = repulsive potential gain
- d*_g = attractive transition distance
- Q* = repulsive radius of influence

This definition provides:
- Smooth quadratic attraction near goal with linear behavior far away
- Repulsive forces only within a bounded influence radius
- Infinite repulsive force at obstacle surface (collision avoidance)
"""

import numpy as np
import matplotlib.pyplot as plt


def attractive_potential_quad_linear(d, k_a, d_g_star):
    """
    Piecewise attractive potential field.
    
    Args:
        d: distance from goal (scalar or array)
        k_a: attractive gain coefficient
        d_g_star: transition distance threshold
    
    Returns:
        potential: potential field value(s)
    """
    d = np.asarray(d)
    potential = np.zeros_like(d, dtype=float)
    
    # Quadratic region: d <= d*_g
    quad_region = d <= d_g_star
    potential[quad_region] = (k_a / 2) * d[quad_region]**2
    
    # Linear region: d > d*_g
    linear_region = d > d_g_star
    potential[linear_region] = d_g_star * k_a * d[linear_region] - (k_a / 2) * d_g_star**2
    
    return potential


def attractive_force_quad_linear(d, k_a, d_g_star):
    """
    Force magnitude from attractive potential (negative gradient).
    Force = -dU/dd
    
    Args:
        d: distance from goal (scalar or array)
        k_a: attractive gain coefficient
        d_g_star: transition distance threshold
    
    Returns:
        force: force magnitude (pulling toward goal)
    """
    d = np.asarray(d)
    force = np.zeros_like(d, dtype=float)
    
    # Quadratic region: F = -d(U)/dd = -k_a * d
    quad_region = d <= d_g_star
    force[quad_region] = k_a * d[quad_region]
    
    # Linear region: F = -d(U)/dd = -d*_g * k_a (constant)
    linear_region = d > d_g_star
    force[linear_region] = d_g_star * k_a
    
    return force


def visualize_attractive_potential(k_a=1.0, d_g_star=2.0, d_max=8.0):
    """
    Create visualization of attractive potential and force field.
    
    Args:
        k_a: attractive gain coefficient
        d_g_star: transition distance threshold
        d_max: maximum distance to display
    """
    # Create distance range
    d = np.linspace(0, d_max, 1000)
    
    # Compute potential and force
    U = attractive_potential_quad_linear(d, k_a, d_g_star)
    F = attractive_force_quad_linear(d, k_a, d_g_star)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))
    
    # Plot 1: Potential field
    ax1.plot(d, U, 'b-', linewidth=2.5, label='Attractive Potential')
    ax1.axvline(d_g_star, color='red', linestyle='--', linewidth=2, 
                label=f'Transition at $d^*_g={d_g_star}$', alpha=0.7)
    ax1.fill_between(d[d <= d_g_star], 0, U[d <= d_g_star], 
                     alpha=0.2, color='blue', label='Quadratic Region')
    ax1.fill_between(d[d > d_g_star], 0, U[d > d_g_star], 
                     alpha=0.2, color='cyan', label='Linear Region')
    
    ax1.set_xlabel('Distance from Goal, $d$ (m)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Potential, $U_a(d)$ (J)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Attractive Potential Field ($k_a={k_a}$, $d^*_g={d_g_star}$)', 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_xlim(0, d_max)
    ax1.set_ylim(bottom=0)
    
    # Plot 2: Force field (magnitude)
    ax2.plot(d, F, 'g-', linewidth=2.5, label='Attractive Force Magnitude')
    ax2.axvline(d_g_star, color='red', linestyle='--', linewidth=2, 
                label=f'Transition at $d^*_g={d_g_star}$', alpha=0.7)
    ax2.fill_between(d[d <= d_g_star], 0, F[d <= d_g_star], 
                     alpha=0.2, color='green', label='Increasing ($F = k_a \cdot d$)')
    ax2.fill_between(d[d > d_g_star], 0, F[d > d_g_star], 
                     alpha=0.2, color='lightgreen', label='Constant ($F = d^*_g \cdot k_a$)')
    
    ax2.set_xlabel('Distance from Goal, $d$ (m)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Force Magnitude, $F_a(d)$ (N)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Attractive Force Field ($k_a={k_a}$, $d^*_g={d_g_star}$)', 
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_xlim(0, d_max)
    ax2.set_ylim(bottom=0)
    
    # Add text annotations explaining the regions
    ax2.text(d_g_star * 0.3, d_g_star * k_a * 0.7, 'Quadratic\nRegion', 
            fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.text(d_g_star * 2.5, d_g_star * k_a * 1.05, 'Linear Region\n(Constant Force)', 
            fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    return fig, ax1, ax2


def compare_potentials(k_a=1.0, d_max=8.0):
    """
    Compare the piecewise potential with standard quadratic potential.
    """
    d = np.linspace(0.01, d_max, 1000)
    d_g_star = 2.0
    
    # Piecewise potential
    U_piecewise = attractive_potential_quad_linear(d, k_a, d_g_star)
    
    # Standard quadratic potential (for comparison)
    U_quadratic = (k_a / 2) * d**2
    
    # Standard linear potential (for comparison)
    U_linear = k_a * d
    
    fig, ax = plt.subplots(figsize=(11, 7))
    
    ax.plot(d, U_piecewise, 'b-', linewidth=3, label='Piecewise (Quadratic + Linear)')
    ax.plot(d, U_quadratic, 'r--', linewidth=2, alpha=0.7, label='Pure Quadratic')
    ax.plot(d, U_linear, 'orange', linestyle='--', linewidth=2, alpha=0.7, label='Pure Linear')
    
    ax.axvline(d_g_star, color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax.fill_between(d[d <= d_g_star], np.min(U_quadratic), np.max(U_quadratic), 
                    alpha=0.1, color='blue', label='Quadratic Dominates')
    ax.fill_between(d[d > d_g_star], np.min(U_quadratic), np.max(U_quadratic), 
                    alpha=0.1, color='cyan', label='Linear Dominates')
    
    ax.set_xlabel('Distance from Goal, $d$ (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Potential, $U(d)$ (J)', fontsize=12, fontweight='bold')
    ax.set_title(f'Attractive Potential: Piecewise vs Pure Quadratic/Linear ($k_a={k_a}$, $d^*_g={d_g_star}$)', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_xlim(0, d_max)
    
    plt.tight_layout()
    return fig, ax


# ============================================================
# REPULSIVE POTENTIAL FUNCTIONS
# ============================================================

def repulsive_potential(D, k_b, Q_star, epsilon=1e-6):
    """
    Repulsive potential field with bounded influence radius.
    
    U_b(D) = {
        (k_b / 2) * (1/D - 1/Q*)^2,  if D <= Q*
        0,                            if D > Q*
    }
    
    Args:
        D: distance from obstacle (scalar or array)
        k_b: repulsive potential gain coefficient
        Q_star: radius of influence (influence distance)
        epsilon: small value to avoid division by zero
    
    Returns:
        potential: repulsive potential value(s)
    """
    D = np.asarray(D)
    potential = np.zeros_like(D, dtype=float)
    
    # Clamp D to avoid division by zero very close to obstacle
    D_safe = np.maximum(D, epsilon)
    
    # Repulsive region: D <= Q*
    inside = D <= Q_star
    potential[inside] = (k_b / 2) * (1.0 / D_safe[inside] - 1.0 / Q_star)**2
    
    # Outside region: D > Q* (potential = 0, already initialized)
    
    return potential


def repulsive_force(D, k_b, Q_star, epsilon=1e-6):
    """
    Force magnitude from repulsive potential (magnitude of negative gradient).
    Force = -dU/dD = k_b * (1/D - 1/Q*) * (1/D^2)
    
    Args:
        D: distance from obstacle (scalar or array)
        k_b: repulsive potential gain coefficient
        Q_star: radius of influence
        epsilon: small value to avoid division by zero
    
    Returns:
        force: repulsive force magnitude (pushing away from obstacle)
    """
    D = np.asarray(D)
    force = np.zeros_like(D, dtype=float)
    
    # Clamp D to avoid division by zero
    D_safe = np.maximum(D, epsilon)
    
    # Repulsive region: D <= Q*
    inside = D <= Q_star
    force[inside] = k_b * (1.0 / D_safe[inside] - 1.0 / Q_star) * (1.0 / D_safe[inside]**2)
    
    # Outside region: D > Q* (force = 0, already initialized)
    
    return force


def visualize_repulsive_potential(k_b=1.0, Q_star=2.0, D_max=5.0):
    """
    Create visualization of repulsive potential and force field.
    
    Args:
        k_b: repulsive gain coefficient
        Q_star: radius of influence
        D_max: maximum distance to display
    """
    # Create distance range (avoid exactly zero)
    D = np.linspace(0.05, D_max, 1000)
    
    # Compute potential and force
    U = repulsive_potential(D, k_b, Q_star)
    F = repulsive_force(D, k_b, Q_star)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))
    
    # Plot 1: Potential field
    ax1.plot(D, U, 'r-', linewidth=2.5, label='Repulsive Potential')
    ax1.axvline(Q_star, color='orange', linestyle='--', linewidth=2, 
                label=f'Influence Radius $Q^*={Q_star}$', alpha=0.7)
    ax1.fill_between(D[D <= Q_star], 0, U[D <= Q_star], 
                     alpha=0.2, color='red', label='Active Repulsion Region')
    ax1.fill_between(D[D > Q_star], 0, np.max(U), 
                     alpha=0.1, color='gray', label='No Influence')
    
    ax1.set_xlabel('Distance from Obstacle, $D$ (m)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Potential, $U_b(D)$ (J)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Repulsive Potential Field ($k_b={k_b}$, $Q^*={Q_star}$)', 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xlim(0, D_max)
    ax1.set_ylim(bottom=0)
    
    # Plot 2: Force field (magnitude)
    ax2.plot(D, F, 'orangered', linewidth=2.5, label='Repulsive Force Magnitude')
    ax2.axvline(Q_star, color='orange', linestyle='--', linewidth=2, 
                label=f'Influence Radius $Q^*={Q_star}$', alpha=0.7)
    ax2.fill_between(D[D <= Q_star], 0, F[D <= Q_star], 
                     alpha=0.2, color='red', label='Active Repulsion Region')
    ax2.fill_between(D[D > Q_star], 0, np.max(F), 
                     alpha=0.1, color='gray', label='No Force')
    
    ax2.set_xlabel('Distance from Obstacle, $D$ (m)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Force Magnitude, $F_b(D)$ (N)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Repulsive Force Field ($k_b={k_b}$, $Q^*={Q_star}$)', 
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_xlim(0, D_max)
    ax2.set_ylim(bottom=0)
    
    # Add text annotations
    ax2.text(Q_star * 0.4, np.max(F) * 0.6, 'Infinite Force\nat Surface', 
            fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.text(Q_star * 1.5, np.max(F) * 0.2, f'Decreases with $1/D^2$', 
            fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    plt.tight_layout()
    return fig, ax1, ax2


def compare_repulsive_potentials(k_b=1.0, D_max=5.0):
    """
    Compare different repulsive potential formulations.
    """
    D = np.linspace(0.1, D_max, 1000)
    Q_star = 2.0
    
    # Bounded repulsive potential (bounded influence)
    U_bounded = repulsive_potential(D, k_b, Q_star)
    
    # Unbounded repulsive potential (for comparison)
    D_safe = np.maximum(D, 1e-6)
    U_unbounded = (k_b / 2) * (1.0 / D_safe)**2
    
    fig, ax = plt.subplots(figsize=(11, 7))
    
    ax.plot(D, U_bounded, 'r-', linewidth=3, label='Bounded (with $Q^*$ influence radius)')
    ax.plot(D, U_unbounded, 'b--', linewidth=2, alpha=0.7, label='Unbounded (traditional)')
    
    ax.axvline(Q_star, color='orange', linestyle=':', linewidth=2, alpha=0.7, 
               label=f'Influence radius $Q^*={Q_star}$')
    ax.fill_between(D[D <= Q_star], 0, np.max(U_unbounded), 
                    alpha=0.1, color='red', label='Bounded Region Active')
    ax.fill_between(D[D > Q_star], 0, np.max(U_unbounded), 
                    alpha=0.1, color='gray', label='Bounded Region Inactive')
    
    ax.set_xlabel('Distance from Obstacle, $D$ (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Potential, $U_b(D)$ (J)', fontsize=12, fontweight='bold')
    ax.set_title(f'Repulsive Potential: Bounded vs Unbounded ($k_b={k_b}$, $Q^*={Q_star}$)', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(0, D_max)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    return fig, ax


if __name__ == '__main__':
    print("=" * 60)
    print("ATTRACTIVE POTENTIAL FIELD VISUALIZATIONS")
    print("=" * 60)
    
    # Parameters for attractive potential
    k_a = 1.0           # Attractive gain coefficient
    d_g_star = 2.0      # Transition distance
    d_max = 8.0         # Maximum distance for visualization
    
    print("Generating attractive potential field visualizations...")
    
    # Main visualization: potential and force
    fig1, ax1a, ax1b = visualize_attractive_potential(k_a, d_g_star, d_max)
    
    # Comparison visualization
    fig2, ax2 = compare_potentials(k_a, d_max)
    
    # Print some values for reference
    print("\n=== Attractive Potential Field Values ===")
    d_test_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
    print(f"{'Distance (m)':<15} {'Potential':<15} {'Force':<15}")
    print("-" * 45)
    for d_val in d_test_values:
        u_val = attractive_potential_quad_linear(d_val, k_a, d_g_star)
        f_val = attractive_force_quad_linear(d_val, k_a, d_g_star)
        print(f"{d_val:<15.2f} {u_val:<15.4f} {f_val:<15.4f}")
    
    print("\n" + "=" * 60)
    print("REPULSIVE POTENTIAL FIELD VISUALIZATIONS")
    print("=" * 60)
    
    # Parameters for repulsive potential
    k_b = 1.0           # Repulsive gain coefficient
    Q_star = 2.0        # Influence radius
    D_max = 5.0         # Maximum distance for visualization
    
    print("Generating repulsive potential field visualizations...")
    
    # Main visualization: potential and force
    fig3, ax3a, ax3b = visualize_repulsive_potential(k_b, Q_star, D_max)
    
    # Comparison visualization
    fig4, ax4 = compare_repulsive_potentials(k_b, D_max)
    
    # Print some values for reference
    print("\n=== Repulsive Potential Field Values ===")
    D_test_values = [0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    print(f"{'Distance (m)':<15} {'Potential':<15} {'Force':<15}")
    print("-" * 45)
    for D_val in D_test_values:
        u_val = repulsive_potential(D_val, k_b, Q_star)
        f_val = repulsive_force(D_val, k_b, Q_star)
        print(f"{D_val:<15.2f} {u_val:<15.4f} {f_val:<15.4f}")
    
    plt.show()
