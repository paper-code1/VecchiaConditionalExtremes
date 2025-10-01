#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import time
from typing import Tuple, Dict, Any, List, Optional
import warnings
import matplotlib
from pathlib import Path

matplotlib.rcParams.update({'font.size': 16})

from utils import (
    MaternKernel,
    MaternKernelHighDimTime,
    AblationGaussianProcess,
    generate_circle_points,
    prepare_results_dir,
    save_json,
    write_text_report,
    format_experiment_filename,
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def run_mixed_precision_experiment(
    n_values: List[int],
    nu_values: List[float],
    n_trials: int = 10,
    quality_values: Optional[List[str]] = None,
    seperablity_values: Optional[List[float]] = None,
    time_scale_values: Optional[List[float]] = None,
    dim_scale_values: Optional[List[List[float]]] = None,
    dim_length_values: Optional[List[int]] = None,
    time_lag_values: Optional[List[int]] = None,
    nu_time_values: Optional[List[float]] = None,
    results_paths: Optional[Dict[str, Path]] = None,
) -> Dict[str, Any]:
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
        'quality_values': quality_values or ['best'],
        'mixed_precision_config': mixed_precision_config,
        'experiments': [],
        'n_trials': n_trials,
    }
    
    quality_values = quality_values or ['best']
    seperablity_values = seperablity_values or [0.0]
    time_scale_values = time_scale_values or [0.3]
    dim_scale_values = dim_scale_values or [[0.05, 0.05, 0.05, 5, 5, 5, 5, 5, 5, 5]]
    dim_length_values = dim_length_values or [10]
    time_lag_values = time_lag_values or [2]
    nu_time_values = nu_time_values or [0.5]

    total_experiments = (
        len(n_values)
        * len(nu_values)
        * len(quality_values)
        * len(seperablity_values)
        * len(time_scale_values)
        * len(dim_scale_values)
        * len(dim_length_values)
        * len(time_lag_values)
        * len(nu_time_values)
    )
    exp_count = 0
    
    for quality in quality_values:
        for n in n_values:
            for nu in nu_values:
                for seperablity in seperablity_values:
                    for time_scale in time_scale_values:
                        for dim_scale in dim_scale_values:
                            for dim_length in dim_length_values:
                                for time_lag in time_lag_values:
                                    for nu_time in nu_time_values:
                                        exp_count += 1
                                        print(f"\nExperiment {exp_count}/{total_experiments}: quality={quality}, n={n}, ν={nu}, ν_t={nu_time}, sep={seperablity}")

                                    # Create kernel and GP
                                    kernel = MaternKernelHighDimTime(
                                        nu_space=nu,
                                        nu_time=nu_time,
                                        variance=1.0,
                                        length_scale=dim_scale,
                                        length_dim=dim_length,
                                        time_scale=time_scale,
                                        time_lag=time_lag,
                                        seperablity=seperablity,
                                    )
                                    gp = AblationGaussianProcess(kernel, noise_variance=1e-5)
                                    
                                    # Storage for trial results
                                    double_ll_values = []
                                    mixed_ll_values = []
                                    ll_differences = []
                                    relative_errors = []
                                    double_times = []
                                    mixed_times = []

                                    # Generate data for trials
                                    np.random.seed(42 + n + int(nu*10))
                                    
                                    for trial in range(n_trials):
                                        print(f"  Trial {trial+1}/{n_trials}", end="")

                                        all_points, inner_points, outer_points = generate_circle_points(
                                            n,
                                            quality=quality,
                                            time_lag=time_lag,
                                            dim_length=dim_length,
                                        )
                                        
                                        y_true_all = np.random.multivariate_normal(
                                            np.zeros(len(all_points)),
                                            kernel(all_points, precision='double'),
                                        )
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
                                                # print(f" ✓ (diff: {ll_diff:.2e}, speedup: {speedup:.2f}x)")
                                            else:
                                                print(f" ✗ (failed)")
                                                
                                        except Exception as e:
                                            print(f" ✗ (error: {str(e)[:30]})")
                                            continue
                                    
                                    # Store experiment results
                                    experiment_result = {
                                        'quality': quality,
                                        'n': n,
                                        'nu': nu,
                                        'nu_time': nu_time,
                                        'seperablity': seperablity,
                                        'time_scale': time_scale,
                                        'dim_scale': dim_scale,
                                        'dim_length': dim_length,
                                        'time_lag': time_lag,
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
                                        'double_time_mean': np.mean(double_times) if double_times else np.nan,
                                        'double_time_std': np.std(double_times) if double_times else np.nan,
                                        'mixed_time_mean': np.mean(mixed_times) if mixed_times else np.nan,
                                        'mixed_time_std': np.std(mixed_times) if mixed_times else np.nan,
                                        'speedup_mean': np.mean([dt/mt for dt, mt in zip(double_times, mixed_times) if mt > 0]) if double_times and mixed_times else np.nan,
                                        'speedup_std': np.std([dt/mt for dt, mt in zip(double_times, mixed_times) if mt > 0]) if double_times and mixed_times else np.nan,
                                    }

                                    results['experiments'].append(experiment_result)

                                    if results_paths is not None:
                                        data_dir = results_paths.get('data', results_paths['root'])
                                        filename = format_experiment_filename(
                                            'mixed_precision',
                                            {
                                                'n': n,
                                                'nu': nu,
                                                'nu_time': nu_time,
                                                'sep': seperablity,
                                                'time_scale': time_scale,
                                                'dim_len': dim_length,
                                                'time_lag': time_lag,
                                            },
                                        )
                                        save_json(experiment_result, data_dir / filename)
                                    
                                    print(
                                        f"  Results: Mean LL diff = {experiment_result['mean_ll_difference']:.2e}, "
                                        f"Mean rel error = {experiment_result['mean_relative_error']:.2e}, "
                                        f"Mean speedup = {experiment_result['speedup_mean']:.2f}x",
                                    )
    
    return results

def visualize_mixed_precision_results(results: Dict[str, Any], results_paths: Optional[Dict[str, Path]] = None):
    """Create comprehensive visualizations of mixed precision comparison results."""
    
    experiments = results['experiments']
    n_values = results['n_values']
    
    # Extract unique parameter combinations
    separability_values = sorted(list(set(exp['seperablity'] for exp in experiments)))
    time_scale_values = sorted(list(set(exp['time_scale'] for exp in experiments)))
    
    # Create parameter combination labels
    param_combinations = []
    for n in n_values:
        for sep in separability_values:
            for ts in time_scale_values:
                param_combinations.append(f'n={n}\ns={sep:.1f}\nτ={ts:.1f}')
    
    n_combinations = len(param_combinations)
    
    # Prepare data arrays
    success_rates = np.full(n_combinations, np.nan)
    mean_errors = np.full(n_combinations, np.nan)
    
    # Fill data arrays
    combo_idx = 0
    for n in n_values:
        for sep in separability_values:
            for ts in time_scale_values:
                subset = [exp for exp in experiments 
                         if exp['n'] == n and exp['seperablity'] == sep and exp['time_scale'] == ts]
                if subset:
                    success_rates[combo_idx] = np.nanmean([exp['success_rate'] for exp in subset])
                    mean_errors[combo_idx] = np.mean([exp['mean_ll_difference'] for exp in subset])
                combo_idx += 1
    # Generate suffix for filenames based on parameters
    n_str = "_".join(map(str, n_values))
    sep_str = "_".join([f"{s:.1f}" for s in separability_values])
    ts_str = "_".join([f"{t:.1f}" for t in time_scale_values])
    param_suffix = f"_n{n_str}_s{sep_str}_t{ts_str}"
    
    # Plot 1: Success Rate
    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(range(n_combinations), success_rates, color='green', alpha=0.7)
    plt.xlabel('Parameter Combinations', fontsize=16)
    plt.ylabel('Success Rate (%)', fontsize=16)
    plt.xticks(range(n_combinations), param_combinations, rotation=45, ha='right', fontsize=12)
    plt.ylim(0, 100)
    
    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars1, success_rates)):
        if not np.isnan(rate):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{rate:.0f}%', ha='center', va='bottom', fontsize=14)
    
    plt.tight_layout()
    if results_paths:
        figures_dir = results_paths.get('figures', results_paths['root'])
        success_path = figures_dir / f'mixed_precision_comparison_success_rate{param_suffix}.pdf'
        plt.savefig(success_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.savefig(f'./fig/mixed_precision_comparison_success_rate{param_suffix}.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 2: Maximum Error (with log scale)
    plt.figure(figsize=(12, 6))
    bars2 = plt.bar(range(n_combinations), mean_errors, color='red', alpha=0.7)
    plt.xlabel('Parameter Combinations', fontsize=16)
    plt.ylabel('Mean Log-Likelihood Difference', fontsize=16)
    plt.xticks(range(n_combinations), param_combinations, rotation=45, ha='right', fontsize=12)
    plt.yscale('log')
    
    # Add horizontal background dashed lines for reference
    y_ticks = plt.yticks()[0]
    for y in y_ticks:
        plt.axhline(y, color='gray', linestyle='dashed', linewidth=0.7, alpha=0.5, zorder=0)

    # Add value labels on bars
    for i, (bar, error) in enumerate(zip(bars2, mean_errors)):
        if not np.isnan(error) and error > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                    f'{error:.1e}', ha='center', va='bottom', fontsize=16)
    
    plt.tight_layout()
    if results_paths:
        max_error_path = figures_dir / f'mixed_precision_comparison_mean_error{param_suffix}.pdf'
        plt.savefig(max_error_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.savefig(f'./fig/mixed_precision_comparison_mean_error{param_suffix}.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    

def create_summary_table(results: Dict[str, Any], results_paths: Optional[Dict[str, Path]] = None):
    """Create a summary table of the results."""
    experiments = results['experiments']
    
    lines = []
    lines.append("=" * 80)
    lines.append("MIXED PRECISION COMPARISON SUMMARY")
    lines.append("=" * 80)

    lines.append("\nMixed Precision Configuration:")
    for op, precision in results['mixed_precision_config'].items():
        lines.append(f"  {op}: {precision}")

    header = (
        f"\n{'n':>4} {'ν':>4} {'Success%':>8} {'Mean LL Diff':>12} {'Max LL Diff':>12} "
        f"{'Mean Rel Err':>12} {'Max Rel Err':>12} {'Double Time':>12} {'Mixed Time':>12} {'Speedup':>8}"
    )
    lines.append(header)
    lines.append("-" * 120)

    for exp in experiments:
        line = (
            f"{exp['n']:>4} {exp['nu']:>4.1f} {exp['success_rate']:>7.0f}% "
            f"{exp['mean_ll_difference']:>11.2e} {exp['max_ll_difference']:>11.2e} "
            f"{exp['mean_relative_error']:>11.2e} {exp['max_relative_error']:>11.2e} "
            f"{exp['double_time_mean']:>11.3f}s {exp['mixed_time_mean']:>11.3f}s "
            f"{exp['speedup_mean']:>7.2f}x"
        )
        lines.append(line)

    # Overall statistics
    all_mean_diffs = [exp['mean_ll_difference'] for exp in experiments if not np.isnan(exp['mean_ll_difference'])]
    all_rel_errors = [exp['mean_relative_error'] for exp in experiments if not np.isnan(exp['mean_relative_error'])]
    all_success_rates = [exp['success_rate'] for exp in experiments]
    all_double_times = [exp['double_time_mean'] for exp in experiments if not np.isnan(exp['double_time_mean'])]
    all_mixed_times = [exp['mixed_time_mean'] for exp in experiments if not np.isnan(exp['mixed_time_mean'])]
    all_speedups = [exp['speedup_mean'] for exp in experiments if not np.isnan(exp['speedup_mean'])]

    lines.append("-" * 120)
    lines.append("Overall Statistics:")
    lines.append(f"  Average Success Rate: {np.mean(all_success_rates):.1f}%")
    lines.append(f"  Average Mean LL Difference: {np.mean(all_mean_diffs):.2e}")
    lines.append(f"  Average Mean Relative Error: {np.mean(all_rel_errors):.2e}")
    lines.append(
        f"  Maximum LL Difference: {np.max([exp['max_ll_difference'] for exp in experiments if not np.isnan(exp['max_ll_difference'])]):.2e}"
    )
    lines.append(f"  Average Double Time: {np.mean(all_double_times):.3f}s")
    lines.append(f"  Average Mixed Time: {np.mean(all_mixed_times):.3f}s")
    lines.append(f"  Average Speedup: {np.mean(all_speedups):.2f}x")
    lines.append(f"  Maximum Speedup: {np.max(all_speedups):.2f}x")

    summary_text = "\n".join(lines)
    print(summary_text)

    if results_paths:
        log_dir = results_paths.get('logs', results_paths['root'])
        log_dir.mkdir(parents=True, exist_ok=True)
        write_text_report(summary_text.splitlines(), log_dir / 'mixed_precision_summary.txt')

def main(
    results_root: Optional[str] = None,
    timestamp: bool = True,
    n_trials_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Main function to run mixed precision comparison."""
    results_paths = prepare_results_dir(
        "mixed_precision_comparison",
        root=results_root,
        timestamp=timestamp,
        subdirs=["figures", "data", "logs"],
    )

    # Experimental parameters (matching operation_llh_error_impact_analysis.py settings)
    n_quality = ['best', 'good', 'worst']  # , 'good', 'worst'
    n_values = [10, 100, 1000]
    nu_values = [3.5]
    n_trials_default = 10
    n_trials = n_trials_override or n_trials_default

    separability_grid = [0.0, 0.5, 1.0]
    time_scale_grid = [0.1, 0.5, 0.8]
    dim_scale_grid = [[0.05, 0.05, 0.05, 5, 5, 5, 5, 5, 5, 5]]
    dim_length_grid = [10]
    time_lag_grid = [2]
    nu_time_grid = [0.5]

    # Run experiments
    results = run_mixed_precision_experiment(
        n_values,
        nu_values,
        n_trials,
        quality_values=n_quality,
        seperablity_values=separability_grid,
        time_scale_values=time_scale_grid,
        dim_scale_values=dim_scale_grid,
        dim_length_values=dim_length_grid,
        time_lag_values=time_lag_grid,
        nu_time_values=nu_time_grid,
        results_paths=results_paths,
    )

    # Create visualizations
    print("\nCreating visualizations...")
    visualize_mixed_precision_results(results, results_paths=results_paths)

    # Create summary table
    print("\nCreating summary table...")
    create_summary_table(results, results_paths=results_paths)

    overall_path = results_paths['data'] / 'mixed_precision_overall_results.json'
    save_json(results, overall_path)
    print(f"All results saved to: {results_paths['root']}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the mixed precision comparison experiments.")
    parser.add_argument(
        "--results-root",
        type=str,
        default=None,
        help="Directory where experiment outputs should be stored. Defaults to ./results.",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Disable timestamped subdirectories for repeatable output paths.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override the number of trials per configuration.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    args = parse_args()
    results = main(
        results_root=args.results_root,
        timestamp=not args.no_timestamp,
        n_trials_override=args.n_trials,
    )
