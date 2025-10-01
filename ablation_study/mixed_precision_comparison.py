#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import time
from typing import Tuple, Dict, Any, List
import warnings
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

from utils import MaternKernel, AblationGaussianProcess, generate_circle_points

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def run_mixed_precision_experiment(n_values: List[int], nu_values: List[float], n_trials: int = 10) -> Dict[str, Any]:
    """Run mixed precision comparison experiment."""
    print("Mixed Precision vs Full Double Precision Comparison")
    print("=" * 60)
    
    # Define the specific mixed precision configuration
    mixed_precision_config = {
        # Double precision operations (critical for accuracy)
        'kernel_train_gen': 'single',
        'kernel_cross_gen': 'double', 
        'kernel_test_gen': 'double',
        'chol_train': 'single',
        'solve_train_cross': 'double', # make sure the llh is not NaN
        'gemm_train': 'double',
        'cov_subtraction': 'double',
        # Single precision operations (less critical)
        'chol_cond': 'single',
        'solve_train_y': 'single',
        'gemv_train': 'single',
        'solve_cond': 'single',
        'log_diag': 'single',
        'inner_product': 'single'
    }
    
    # Full double precision configuration (baseline)
    full_double_config = {}  # Empty dict means all double precision
    
    results = {
        'n_values': n_values,
        'nu_values': nu_values,
        'mixed_precision_config': mixed_precision_config,
        'experiments': []
    }
    
    total_experiments = len(n_values) * len(nu_values)
    exp_count = 0
    
    for n in n_values:
        for nu in nu_values:
            exp_count += 1
            print(f"\nExperiment {exp_count}/{total_experiments}: n={n}, ν={nu}")
            
            # Create kernel and GP
            kernel = MaternKernel(nu=nu, length_scale=0.1/nu, variance=1.0)
            gp = AblationGaussianProcess(kernel, noise_variance=1e-5)
            
            # Storage for trial results
            double_ll_values = []
            mixed_ll_values = []
            ll_differences = []
            relative_errors = []
            double_times = []
            mixed_times = []
            
            # Generate data for trials
            np.random.seed(42 + n + int(nu*10))  # Reproducible
            
            for trial in range(n_trials):
                print(f"  Trial {trial+1}/{n_trials}", end="")
                
                # Generate spatial points and reference function
                all_points, inner_points, outer_points = generate_circle_points(n)
                
                # Generate reference function values (always double precision)
                y_true_all = np.random.multivariate_normal(
                    np.zeros(len(all_points)), 
                    kernel(all_points, precision='double')
                )
                # y_true_all = np.zeros(len(all_points))
                y_true_inner = y_true_all[:len(inner_points)]
                y_true_outer = y_true_all[len(inner_points):len(inner_points)+len(outer_points)]
                
                try:
                    # Compute full double precision log-likelihood with timing
                    start_time = time.time()
                    ll_double, diag_double = gp.conditional_log_likelihood_ablation(
                        outer_points, y_true_outer, inner_points, y_true_inner, full_double_config
                    )
                    double_time = time.time() - start_time
                    
                    # Compute mixed precision log-likelihood with timing
                    start_time = time.time()
                    ll_mixed, diag_mixed = gp.conditional_log_likelihood_ablation(
                        outer_points, y_true_outer, inner_points, y_true_inner, mixed_precision_config
                    )
                    mixed_time = time.time() - start_time
                    
                    if ll_double is not np.nan and ll_mixed is not np.nan:  # Both succeeded
                        double_ll_values.append(ll_double)
                        mixed_ll_values.append(ll_mixed)
                        double_times.append(double_time)
                        mixed_times.append(mixed_time)
                        
                        ll_diff = abs(ll_mixed - ll_double)
                        ll_differences.append(ll_diff)
                        
                        rel_error = ll_diff / abs(ll_double) if abs(ll_double) > 1e-12 else 0
                        relative_errors.append(rel_error)
                        
                        speedup = double_time / mixed_time if mixed_time > 0 else np.nan
                        print(f" ✓ (diff: {ll_diff:.2e}, speedup: {speedup:.2f}x)")
                    else:
                        print(f" ✗ (failed)")
                        
                except Exception as e:
                    print(f" ✗ (error: {str(e)[:30]})")
                    continue
            
            # Store experiment results
            experiment_result = {
                'n': n,
                'nu': nu,
                'n_inner': len(inner_points) if 'inner_points' in locals() else 0,
                'n_outer': len(outer_points) if 'outer_points' in locals() else 0,
                'n_successful_trials': len(double_ll_values),
                'double_ll_mean': np.mean(double_ll_values) if double_ll_values else np.nan,
                'double_ll_std': np.std(double_ll_values) if double_ll_values else np.nan,
                'mixed_ll_mean': np.mean(mixed_ll_values) if mixed_ll_values else np.nan,
                'mixed_ll_std': np.std(mixed_ll_values) if mixed_ll_values else np.nan,
                'mean_ll_difference': np.mean(ll_differences) if ll_differences else np.nan,
                'max_ll_difference': np.max(ll_differences) if ll_differences else np.nan,
                'mean_relative_error': np.mean(relative_errors) if relative_errors else np.nan,
                'max_relative_error': np.max(relative_errors) if relative_errors else np.nan,
                'success_rate': len(double_ll_values) / n_trials * 100,
                # Timing statistics
                'double_time_mean': np.mean(double_times) if double_times else np.nan,
                'double_time_std': np.std(double_times) if double_times else np.nan,
                'mixed_time_mean': np.mean(mixed_times) if mixed_times else np.nan,
                'mixed_time_std': np.std(mixed_times) if mixed_times else np.nan,
                'speedup_mean': np.mean([dt/mt for dt, mt in zip(double_times, mixed_times) if mt > 0]) if double_times and mixed_times else np.nan,
                'speedup_std': np.std([dt/mt for dt, mt in zip(double_times, mixed_times) if mt > 0]) if double_times and mixed_times else np.nan
            }
            
            results['experiments'].append(experiment_result)
            
            print(f"  Results: Mean LL diff = {experiment_result['mean_ll_difference']:.2e}, "
                  f"Mean rel error = {experiment_result['mean_relative_error']:.2e}, "
                  f"Mean speedup = {experiment_result['speedup_mean']:.2f}x")
    
    return results

def visualize_mixed_precision_results(results: Dict[str, Any]):
    """Create comprehensive visualizations of mixed precision comparison results."""
    
    experiments = results['experiments']
    n_values = results['n_values']
    nu_values = results['nu_values']
    
    # Prepare data matrices
    n_n = len(n_values)
    n_nu = len(nu_values)
    
    ll_diff_matrix = np.full((n_n, n_nu), np.nan)
    rel_error_matrix = np.full((n_n, n_nu), np.nan)
    success_matrix = np.full((n_n, n_nu), np.nan)
    max_error_matrix = np.full((n_n, n_nu), np.nan)
    speedup_matrix = np.full((n_n, n_nu), np.nan)
    double_time_matrix = np.full((n_n, n_nu), np.nan)
    mixed_time_matrix = np.full((n_n, n_nu), np.nan)
    
    for exp in experiments:
        i = n_values.index(exp['n'])
        j = nu_values.index(exp['nu'])
        ll_diff_matrix[i, j] = exp['mean_ll_difference']
        rel_error_matrix[i, j] = exp['mean_relative_error']
        success_matrix[i, j] = exp['success_rate']
        max_error_matrix[i, j] = exp['max_ll_difference']
        speedup_matrix[i, j] = exp['speedup_mean']
        double_time_matrix[i, j] = exp['double_time_mean']
        mixed_time_matrix[i, j] = exp['mixed_time_mean']
    
    # Plot 1: Success Rate
    plt.figure(figsize=(8, 6))
    im1 = plt.imshow(success_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    # plt.title('Success Rate (%)')
    plt.xticks(range(n_nu), [f'ν={nu}' for nu in nu_values])
    plt.yticks(range(n_n), [f'n={n}' for n in n_values])
    # plt.ylabel([f'n={n}' for n in n_values])
    plt.colorbar(im1)
    
    # Add text annotations
    for i in range(n_n):
        for j in range(n_nu):
            if not np.isnan(success_matrix[i, j]):
                text = plt.text(j, i, f'{success_matrix[i, j]:.0f}%',
                                     ha="center", va="center", 
                                     color="black" if success_matrix[i, j] > 50 else "white",
                                     fontsize=16)
    plt.savefig('./fig/mixed_precision_comparison_success_rate.pdf', dpi=300, bbox_inches='tight')
    
    # Plot 2: Maximum Error (with log colorbar)

    plt.figure(figsize=(8, 6))
    # To avoid issues with log scale, set minimum positive value for log scale
    min_nonzero = np.nanmin(max_error_matrix[max_error_matrix > 0]) if np.any(max_error_matrix > 0) else 1e-12
    im2 = plt.imshow(max_error_matrix, cmap='Reds', aspect='auto', norm=LogNorm(vmin=min_nonzero, vmax=np.nanmax(max_error_matrix)))
    # im2 = plt.imshow(max_error_matrix, cmap='Reds', aspect='auto')
    # plt.title('Maximum Log-Likelihood Difference')
    plt.xticks(range(n_nu), [f'ν={nu}' for nu in nu_values])
    plt.yticks(range(n_n), [f'n={n}' for n in n_values])
    # plt.ylabel([f'n={n}' for n in n_values])
    plt.colorbar(im2)
    
    # Add text annotations
    for i in range(n_n):
        for j in range(n_nu):
            if not np.isnan(max_error_matrix[i, j]):
                text = plt.text(j, i, f'{max_error_matrix[i, j]:.1e}',
                                     ha="center", va="center", color="white", fontsize=16)
    
    plt.tight_layout()
    plt.savefig('./fig/mixed_precision_comparison_max_error.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: Speedup Comparison
    plt.figure(figsize=(8, 6))
    im3 = plt.imshow(speedup_matrix, cmap='Blues', aspect='auto', vmin=np.nanmin(speedup_matrix), vmax=np.nanmax(speedup_matrix))
    # plt.title('Mixed Precision Speedup (Double Time / Mixed Time)')
    plt.xticks(range(n_nu), [f'ν={nu}' for nu in nu_values])
    plt.yticks(range(n_n), [f'n={n}' for n in n_values])
    plt.colorbar(im3)
    
    # Add text annotations
    for i in range(n_n):
        for j in range(n_nu):
            if not np.isnan(speedup_matrix[i, j]):
                text = plt.text(j, i, f'{speedup_matrix[i, j]:.2f}x',
                                     ha="center", va="center", color="white", fontsize=16)
    
    plt.tight_layout()
    plt.savefig('./fig/mixed_precision_speedup_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    

def create_summary_table(results: Dict[str, Any]):
    """Create a summary table of the results."""
    experiments = results['experiments']
    
    print("\n" + "=" * 80)
    print("MIXED PRECISION COMPARISON SUMMARY")
    print("=" * 80)
    
    print("\nMixed Precision Configuration:")
    for op, precision in results['mixed_precision_config'].items():
        print(f"  {op}: {precision}")
    
    print(f"\n{'n':>4} {'ν':>4} {'Success%':>8} {'Mean LL Diff':>12} {'Max LL Diff':>12} {'Mean Rel Err':>12} {'Max Rel Err':>12} {'Double Time':>12} {'Mixed Time':>12} {'Speedup':>8}")
    print("-" * 120)
    
    for exp in experiments:
        print(f"{exp['n']:>4} {exp['nu']:>4.1f} {exp['success_rate']:>7.0f}% "
              f"{exp['mean_ll_difference']:>11.2e} {exp['max_ll_difference']:>11.2e} "
              f"{exp['mean_relative_error']:>11.2e} {exp['max_relative_error']:>11.2e} "
              f"{exp['double_time_mean']:>11.3f}s {exp['mixed_time_mean']:>11.3f}s "
              f"{exp['speedup_mean']:>7.2f}x")
    
    # Overall statistics
    all_mean_diffs = [exp['mean_ll_difference'] for exp in experiments if not np.isnan(exp['mean_ll_difference'])]
    all_rel_errors = [exp['mean_relative_error'] for exp in experiments if not np.isnan(exp['mean_relative_error'])]
    all_success_rates = [exp['success_rate'] for exp in experiments]
    all_double_times = [exp['double_time_mean'] for exp in experiments if not np.isnan(exp['double_time_mean'])]
    all_mixed_times = [exp['mixed_time_mean'] for exp in experiments if not np.isnan(exp['mixed_time_mean'])]
    all_speedups = [exp['speedup_mean'] for exp in experiments if not np.isnan(exp['speedup_mean'])]
    
    print("-" * 120)
    print(f"Overall Statistics:")
    print(f"  Average Success Rate: {np.mean(all_success_rates):.1f}%")
    print(f"  Average Mean LL Difference: {np.mean(all_mean_diffs):.2e}")
    print(f"  Average Mean Relative Error: {np.mean(all_rel_errors):.2e}")
    print(f"  Maximum LL Difference: {np.max([exp['max_ll_difference'] for exp in experiments if not np.isnan(exp['max_ll_difference'])]):.2e}")
    print(f"  Average Double Time: {np.mean(all_double_times):.3f}s")
    print(f"  Average Mixed Time: {np.mean(all_mixed_times):.3f}s")
    print(f"  Average Speedup: {np.mean(all_speedups):.2f}x")
    print(f"  Maximum Speedup: {np.max(all_speedups):.2f}x")

def main():
    """Main function to run mixed precision comparison."""
    # Experimental parameters
    n_values = [10, 20, 50, 80, 100, 200, 500, 1000]
    nu_values = [0.5, 1.5, 2.5]
    # n_values = [10, 20]
    # nu_values = [0.5]
    n_trials = 10
    
    # Run experiments
    results = run_mixed_precision_experiment(n_values, nu_values, n_trials)
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualize_mixed_precision_results(results)
    
    # Create summary table
    print("\nCreating summary table...")
    create_summary_table(results)
    
    return results

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    results = main()
