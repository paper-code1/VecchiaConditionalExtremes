#!/usr/bin/env python3
"""
Spatial Function Visualization Demo

This script demonstrates the beautiful visualization functions for 
Gaussian Process function values at different spatial locations.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from utils import MaternKernel, generate_circle_points, plot_simple_spatial_function

def generate_sample_data(n: int = 100, nu: float = 1.5, n_trials: int = 3, quality: str = 'best'):
    """Generate sample data for visualization."""
    print(f"Generating sample data with n={n}, nu={nu}")
    
    # Create kernel
    kernel = MaternKernel(nu=nu, length_scale=0.3/nu, variance=1.0)
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate spatial points
    all_points, inner_points, outer_points = generate_circle_points(n, quality=quality)
    
    print(f"Generated {len(all_points)} total points:")
    print(f"  - Inner region: {len(inner_points)} points")
    print(f"  - Outer region: {len(outer_points)} points")
    
    # Generate function values from GP prior
    K = kernel(all_points, precision='double')
    K += 1e-5 * np.eye(len(all_points))  # Add noise for numerical stability
    
    # Sample function values
    y_true_all = np.random.multivariate_normal(
        np.zeros(len(all_points)), K
    )
    
    # Split into inner and outer
    y_true_inner = y_true_all[:len(inner_points)]
    y_true_outer = y_true_all[len(inner_points):len(inner_points)+len(outer_points)]
    
    return inner_points, outer_points, y_true_inner, y_true_outer

def create_visualization_gallery():
    """Create a gallery of visualizations with different parameters."""
    print("Creating visualization gallery...")
    print("=" * 50)
    
    # Different parameter combinations
    param_combinations = [
        (100, 1.5, 'best', "Matérn Kernel (ν=1.5) - Medium Scale, Best Approximation"), 
        (100, 1.5, 'good', "Matérn Kernel (ν=1.5) - Medium Scale, Good Approximation"), 
        (100, 1.5, 'worst', "Matérn Kernel (ν=1.5) - Medium Scale, Worst Approximation"), 
        
    ]
    
    for i, (n, nu, quality, description) in enumerate(param_combinations, 1):
        print(f"\nCreating visualization {i}: {description}")
        
        # Generate data
        inner_points, outer_points, y_true_inner, y_true_outer = generate_sample_data(n, nu, quality=quality)
        
        # Print some statistics
        print(f"Function value statistics:")
        print(f"  - Inner region: mean={np.mean(y_true_inner):.3f}, std={np.std(y_true_inner):.3f}")
        print(f"  - Outer region: mean={np.mean(y_true_outer):.3f}, std={np.std(y_true_outer):.3f}")
        print(f"  - Global range: [{np.min([y_true_inner.min(), y_true_outer.min()]):.3f}, "
              f"{np.max([y_true_inner.max(), y_true_outer.max()]):.3f}]")
        
        # # Create detailed visualization
        # title = f"{description}\nSpatial Gaussian Process Realization"
        # save_path = f"./fig/spatial_visualization_detailed_{i}.pdf"
        
        # fig1 = plot_spatial_function(
        #     inner_points, outer_points, y_true_inner, y_true_outer,
        #     title=title, save_path=save_path, figsize=(14, 10)
        # )
        
        # Create simple visualization
        simple_title = f"{description}"
        simple_save_path = f"./fig/spatial_visualization_simple_{i}.pdf"
        
        fig2 = plot_simple_spatial_function(
            inner_points, outer_points, y_true_inner, y_true_outer,
            title=simple_title, save_path=simple_save_path, figsize=(10, 8)
        )
        
        # print(f"  - Detailed plot saved: {save_path}")
        print(f"  - Simple plot saved: {simple_save_path}")
        
        # Close figures to save memory
        # plt.close(fig1)
        plt.close(fig2)

def main():
    """Main function to run all visualizations."""
    print("Spatial Function Visualization Demo")
    print("=" * 50)
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs('./fig', exist_ok=True)
    
    # Create visualization gallery
    create_visualization_gallery()

if __name__ == "__main__":
    main()
