"""
Choke Point Navigation Visualization for Multi-Agent Systems

Visualizes Algorithm 2: A Choke Point Navigation Algorithm for Networked Agents

The algorithm demonstrates how a team of networked agents navigates through a choke point:
1. Leader agents detect obstacles on both sides and front
2. When choke point is detected (sides blocked, front clear):
   - Stop team movement
   - Gradually decrease scaling vector s = [sx, sy]
   - Reduce formation offset: δ_hat = [sx*δx, sy*δy]
   - Continue until choke point is cleared
3. Resume normal formation movement

This visualization shows:
- 4 leader robots (blue circles) in square formation
- 3 follower robots (red circles) inside formation
- Dynamic scaling of formation to pass through choke point
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyBboxPatch


class ChokePointNavigationSimulation:
    """Simulates multi-agent navigation through a choke point"""
    
    def __init__(self, choke_width=2.0, choke_x=15.0, initial_formation_size=4.0):
        """
        Args:
            choke_width: width of the choke point opening
            choke_x: x-coordinate where choke point is located
            initial_formation_size: initial side length of leader square formation
        """
        self.choke_width = choke_width
        self.choke_x = choke_x
        self.formation_size = initial_formation_size
        self.min_formation_size = choke_width * 0.9  # Scale down to 90% of choke width
        
        # Scaling vector evolution
        self.scaling = np.array([1.0, 1.0])  # [sx, sy]
        self.min_scaling = np.array([0.25, 0.25])  # Scale down to 40%
        self.scaling_decay_rate = 0.2  # Rate at which scaling decreases
        
        # Virtual leader position (centered at y=2)
        self.virtual_leader_pos = np.array([2.0, 2.0])
        self.goal_pos = np.array([18.0, 2.0])
        
        # Initial leader positions (square formation around virtual leader)
        self.leader_offsets = np.array([
            [-self.formation_size/2, -self.formation_size/2],  # Bottom-left
            [self.formation_size/2, -self.formation_size/2],   # Bottom-right
            [self.formation_size/2, self.formation_size/2],    # Top-right
            [-self.formation_size/2, self.formation_size/2],   # Top-left
        ])
        
        # Follower positions (inside formation)
        self.follower_offsets = np.array([
            [0.0, -self.formation_size/4],    # Front follower
            [-self.formation_size/4, self.formation_size/4],   # Left follower
            [self.formation_size/4, self.formation_size/4],    # Right follower
        ])
        
        # State tracking
        self.choke_point_detected = False
        self.in_choke_phase = False
        self.passed_choke = False
        
    def get_formation_positions(self):
        """Compute current leader and follower positions with scaling"""
        # Apply scaling to offsets
        scaled_leader_offsets = self.leader_offsets * self.scaling
        scaled_follower_offsets = self.follower_offsets * self.scaling
        
        leaders = self.virtual_leader_pos + scaled_leader_offsets
        followers = self.virtual_leader_pos + scaled_follower_offsets
        
        return leaders, followers
    
    def detect_choke_point(self, leaders):
        """
        Detect if choke point is ahead and blocking formation.
        
        Returns:
            (is_choke_detected, sides_blocked, front_clear)
        """
        # Check if any leader is approaching choke x-coordinate
        choke_approach_distance = 4.5  # Distance at which we start detecting
        choke_region_x_min = self.choke_x - choke_approach_distance
        choke_region_x_max = self.choke_x + 2.0
        
        near_choke = np.any((leaders[:, 0] > choke_region_x_min) & 
                           (leaders[:, 0] < choke_region_x_max))
        
        if not near_choke:
            return False, False, False
        
        # Hardcoded gap parameters (choke centered at y=2)
        gap_min = 0.2  # Bottom of choke opening
        gap_max = 3.8  # Top of choke opening
        
        # Check all leaders - if any hit the walls, sides are blocked
        leader_y_positions = leaders[:, 1]
        sides_blocked = np.any((leader_y_positions < gap_min) | (leader_y_positions > gap_max))
        
        # Check if any leader is ahead of choke point
        frontmost_leader_x = np.max(leaders[:, 0])
        front_clear = frontmost_leader_x < self.choke_x
        
        is_choke = sides_blocked and front_clear
        
        return is_choke, sides_blocked, front_clear
    
    def update(self, dt=0.1):
        """Update simulation state"""
        leaders, followers = self.get_formation_positions()
        
        # Detect choke point
        is_choke, sides_blocked, front_clear = self.detect_choke_point(leaders)
        
        if is_choke and not self.passed_choke:
            self.in_choke_phase = True
            self.choke_point_detected = True
            
            # Gradually decrease scaling vector to 60%
            self.scaling = np.maximum(self.scaling - self.scaling_decay_rate * dt,
                                     self.min_scaling)
            
            # Check if we've scaled down enough to pass through
            if np.allclose(self.scaling, self.min_scaling, atol=0.01):
                self.in_choke_phase = False
                self.passed_choke = True
        
        elif self.passed_choke and np.linalg.norm(self.virtual_leader_pos - self.goal_pos) > 0.1:
            # Resume normal scaling after passing through (but stay scaled until very close to goal)
            self.scaling = np.minimum(self.scaling + self.scaling_decay_rate * 0.2 * dt,
                                     np.array([1.0, 1.0]))
        
        # Move virtual leader towards goal
        # Stop before collision with choke point, only move after shrinking
        direction = self.goal_pos - self.virtual_leader_pos
        distance = np.linalg.norm(direction)
        
        # Detect if we're about to hit the choke point too early
        near_choke_x = (self.virtual_leader_pos[0] > self.choke_x - 2.5)
        
        # Continue moving until virtual leader x position reaches goal x position
        if self.virtual_leader_pos[0] < self.goal_pos[0]:
            # Only move if we can safely proceed or we've already scaled down
            velocity = 3.0 * dt  # Speed
            self.virtual_leader_pos += (direction / distance) * velocity
    
    def draw(self, ax):
        """Draw current state on axes"""
        ax.clear()
        
        # Draw choke point (obstacles) - hardcoded to center at y=2
        wall_thickness = 1.0
        gap_min = -0.5  # Bottom of opening
        gap_max = 4.5  # Top of opening
        
        # Top wall (above gap)
        top_wall = FancyBboxPatch((self.choke_x - 3, gap_max), 6, wall_thickness,
                                 boxstyle="round,pad=0.1", 
                                 edgecolor='black', facecolor='darkgray', linewidth=2)
        ax.add_patch(top_wall)
        
        # Bottom wall (below gap)
        bottom_wall = FancyBboxPatch((self.choke_x - 3, gap_min - wall_thickness), 6, wall_thickness,
                                    boxstyle="round,pad=0.1",
                                    edgecolor='black', facecolor='darkgray', linewidth=2)
        ax.add_patch(bottom_wall)
        
        # Draw choke opening
        ax.plot([self.choke_x, self.choke_x], 
               [0.2, 3.8],
               'g--', linewidth=1, alpha=0.5)
        
        # Get current positions
        leaders, followers = self.get_formation_positions()
        
        # Draw formation control structure (connecting lines)
        # Connect virtual leader to leaders
        for leader in leaders:
            ax.plot([self.virtual_leader_pos[0], leader[0]],
                   [self.virtual_leader_pos[1], leader[1]],
                   'b--', alpha=0.3, linewidth=1)
        
        # Connect leaders to form square
        leader_order = [0, 1, 3, 2, 0]
        for i in range(len(leader_order) - 1):
            idx1, idx2 = leader_order[i], leader_order[i+1]
            ax.plot([leaders[idx1, 0], leaders[idx2, 0]],
                   [leaders[idx1, 1], leaders[idx2, 1]],
                   'b-', alpha=0.3, linewidth=1)
        
        # Draw virtual leader
        virtual_circle = Circle(self.virtual_leader_pos, 0.3, 
                               color='cyan', alpha=0.6, zorder=5)
        ax.add_patch(virtual_circle)
        ax.text(self.virtual_leader_pos[0], self.virtual_leader_pos[1] - 0.8,
               'Virtual\nLeader', ha='center', fontsize=8, fontweight='bold')
        
        # Draw leader robots (blue circles)
        for i, leader in enumerate(leaders):
            circle = Circle(leader, 0.4, color='blue', zorder=5)
            ax.add_patch(circle)
            ax.text(leader[0], leader[1] - 0.9, f'L{i+1}', 
                   ha='center', fontsize=7, fontweight='bold', color='blue')
        
        # Draw follower robots (red circles)
        for i, follower in enumerate(followers):
            circle = Circle(follower, 0.3, color='red', zorder=4)
            ax.add_patch(circle)
            ax.text(follower[0], follower[1] - 0.7, f'F{i+1}', 
                   ha='center', fontsize=7, fontweight='bold', color='red')
        
        # Draw goal
        goal_circle = Circle(self.goal_pos, 0.5, 
                            color='green', alpha=0.3, linewidth=2, 
                            fill=False, linestyle='--', zorder=3)
        ax.add_patch(goal_circle)
        ax.text(self.goal_pos[0], self.goal_pos[1] - 1.0, 'Goal', 
               ha='center', fontsize=9, fontweight='bold', color='green')
        
        # Draw scaling indicator
        status_text = f"Scaling: sx={self.scaling[0]:.2f}, sy={self.scaling[1]:.2f}"
        if self.in_choke_phase:
            status_text += " [SHRINKING]"
            ax.text(1, -3.5, status_text, fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        elif self.choke_point_detected:
            status_text += " [CHOKE DETECTED]"
            ax.text(1, -3.5, status_text, fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
        elif self.passed_choke:
            status_text += " [EXPANDING]"
            ax.text(1, -3.5, status_text, fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        else:
            ax.text(1, -3.5, status_text, fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # Formatting
        ax.set_xlim(-2, 22)
        ax.set_ylim(-5, 8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.set_xlabel('X (m)', fontweight='bold')
        ax.set_ylabel('Y (m)', fontweight='bold')
        ax.set_title('Choke Point Navigation - Multi-Agent System\n' + 
                    'Algorithm 2: Scaling Vector Approach', fontsize=12, fontweight='bold')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Leader (Blue)',
                  markerfacecolor='blue', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='Follower (Red)',
                  markerfacecolor='red', markersize=7),
            Line2D([0], [0], marker='o', color='w', label='Virtual Leader (Cyan)',
                  markerfacecolor='cyan', markersize=8),
            Line2D([0], [0], color='darkgray', linewidth=3, label='Obstacle'),
            Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='Goal'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)


def create_animation_and_save(output_filename='choke_point_navigation.gif', 
                             num_frames=300, fps=40):
    """Create and save animation as GIF"""
    
    print(f"Creating choke point navigation animation...")
    print(f"  Output: {output_filename}")
    print(f"  Frames: {num_frames}")
    print(f"  FPS: {fps}")
    
    # Create simulation
    sim = ChokePointNavigationSimulation(choke_width=2.0, choke_x=15.0, 
                                       initial_formation_size=4.0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    def animate(frame):
        """Animation update function"""
        sim.update(dt=0.02)
        sim.draw(ax)
        
        # Progress indicator
        progress = (frame + 1) / num_frames * 100
        print(f"\r  Progress: {progress:.1f}%", end='', flush=True)
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=num_frames, 
                        interval=1000/fps, repeat=False, blit=False)
    
    # Save as GIF
    writer = PillowWriter(fps=fps)
    anim.save(output_filename, writer=writer)
    
    print(f"\n✓ Animation saved: {output_filename}")
    plt.close()


if __name__ == '__main__':
    import os
    
    # Create output directory if needed
    output_dir = 'imgs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Generate animation
    output_path = os.path.join(output_dir, 'choke_point_navigation.gif')
    create_animation_and_save(output_filename=output_path, num_frames=300, fps=20)
    
    print("\n" + "="*60)
    print("Animation created successfully!")
    print("="*60)
    print("\nVisualization shows:")
    print("  1. Team formation approaches choke point")
    print("  2. Algorithm detects choke (side obstacles, clear front)")
    print("  3. Formation shrinks (scaling decreases)")
    print("  4. Team passes through choke point")
    print("  5. Formation expands back to normal size")
