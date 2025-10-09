"""
Visualize the space-time point generation for understanding the circle_points structure.

Terminology:
- Block (inner_points): The conditioning set - points we're making predictions for
- Neighbors (outer_points): The nearest neighbors used in the Vecchia approximation
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys
sys.path.append('/home/panq/Documents/MixedPrecisionSBV/ablation_study_space_time')
from utils import generate_circle_points

def visualize_spacetime_2d(n=15, n_points=300, time_lag=2, quality='best', seed=42):
    """
    Visualize the space-time point generation in 2D for easier understanding.
    
    Parameters:
    - n: neighbor size parameter (number of nearest neighbors)
    - n_points: number of spatial points to generate
    - time_lag: number of time steps
    - quality: 'best', 'good', or 'worst' approximation quality
    """
    np.random.seed(seed)
    
    # Generate points with 2D spatial dimension (easier to visualize)
    points, block_points, neighbor_points = generate_circle_points(
        n=n, n_points=n_points, quality=quality, 
        time_lag=time_lag, dim_length=2
    )
    
    # Calculate radii
    neighbor_region_radius = 1.0 / n
    block_radius = 1.0 / (n * np.power(3, 1/2))
    limit = np.max(np.abs(neighbor_points[:, 0])) * 1.5
    print(f"Limit: {limit}")
    
    # Create figure with subplots for each time step
    fig, axes = plt.subplots(1, time_lag, figsize=(8*time_lag, 7))
    if time_lag == 1:
        axes = [axes]
    
    for t in range(time_lag):
        ax = axes[t]
        
        # Extract points at this time step
        points_at_t = points[points[:, -1] == t]
        block_at_t = block_points[block_points[:, -1] == t] if len(block_points) > 0 else np.array([])
        neighbors_at_t = neighbor_points[neighbor_points[:, -1] == t] if len(neighbor_points) > 0 else np.array([])
        
        # Draw circles showing regions
        circle_neighbor_region = Circle((0, 0), neighbor_region_radius, fill=False, 
                            edgecolor='blue', linewidth=2, linestyle='--', 
                            # label=f'Neighbor region radius = {neighbor_region_radius:.4f}'
                            )
        circle_block = Circle((0, 0), block_radius, fill=False, 
                            edgecolor='red', linewidth=2, linestyle='--',
                            # label=f'Block radius = {block_radius:.4f}'
                            )
        ax.add_patch(circle_neighbor_region)
        ax.add_patch(circle_block)
        
        # Plot neighbors (blue) - these are the conditioning variables
        if len(neighbors_at_t) > 0:
            ax.scatter(neighbors_at_t[:, 0], neighbors_at_t[:, 1], 
                      c='blue', s=50, alpha=0.6, label=f'Neighbors ({len(neighbors_at_t)})',
                      edgecolors='black', linewidth=0.5)
        
        # Plot block points (red) - these are what we're predicting
        if len(block_at_t) > 0:
            ax.scatter(block_at_t[:, 0], block_at_t[:, 1], 
                      c='red', s=100, alpha=0.8, label=f'Block ({len(block_at_t)})',
                      edgecolors='black', linewidth=1, marker='s')
        
        # Set equal aspect and limits - enlarge the view
        ax.set_aspect('equal')
        # if quality == 'best':

        # elif quality == 'good':
        #     limit = neighbor_region_radius * np.sqrt(n) * 1.5
        # else:  # worst
        #     limit = neighbor_region_radius * 3.0
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        
        # Labels and title
        ax.set_xlabel('x', fontsize=18)
        ax.set_ylabel('y', fontsize=18)
        ax.set_title(f'Time Step t={t}', fontsize=18, 
        # fontweight='bold'
        )
        ax.legend(loc='upper left', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    
    # fig.suptitle(f'Space-Time Point Generation (n={n}, quality={quality}, time_lag={time_lag})', 
    #              fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Print statistics
    print(f"\n{'='*80}")
    print(f"Space-Time Point Generation Summary (n={n}, quality='{quality}', time_lag={time_lag})")
    print(f"{'='*80}")
    print(f"Neighbor region radius: {neighbor_region_radius:.6f}")
    print(f"Block radius: {block_radius:.6f}")
    print(f"Radius ratio: {neighbor_region_radius/block_radius:.2f}")
    print(f"\nTotal points: {len(points)}")
    print(f"  - Block points (target): {len(block_points)}")
    print(f"  - Neighbor points (conditioning): {len(neighbor_points)}")
    print(neighbor_points[neighbor_points[:, -1] == (time_lag)])
    
    for t in range(time_lag):
        points_at_t = points[points[:, -1] == t]
        block_at_t = block_points[block_points[:, -1] == t] if len(block_points) > 0 else np.array([])
        neighbors_at_t = neighbor_points[neighbor_points[:, -1] == t] if len(neighbor_points) > 0 else np.array([])
        print(f"\nTime step t={t}:")
        print(f"  - Total points: {len(points_at_t)}")
        print(f"  - Block points: {len(block_at_t)}")
        print(f"  - Neighbor points: {len(neighbors_at_t)}")
    
    if quality == 'best':
        print("   - BEST: Neighbors at last time step remain close to the block")
        print("   - No transformation applied")
    elif quality == 'good':
        print("   - GOOD: Neighbors at last time step are moved FAR from the block")
        print(f"   - Neighbors are EXPANDED by sqrt(n)={np.sqrt(n):.3f}")
    elif quality == 'worst':
        print("   - WORST: Neighbors at last time step are moved FAR from the block")
        print("   - Neighbors are SHIFTED by +0.5 in all dimensions")
    print(f"{'='*80}\n")
    
    return fig

if __name__ == "__main__":
    # Visualize all three quality settings
    for quality in ['best', 'good', 'worst']:
        fig = visualize_spacetime_2d(n=15, n_points=300, time_lag=2, quality=quality, seed=42)
        plt.savefig(f'/home/panq/Documents/MixedPrecisionSBV/ablation_study_space_time/spacetime_visualization_{quality}.png', 
                    dpi=150, bbox_inches='tight')
        print(f"Saved visualization for quality='{quality}'")
    
    plt.show()

