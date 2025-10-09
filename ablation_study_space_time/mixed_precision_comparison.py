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
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_existing_results(results_paths: Dict[str, Path]) -> Optional[Dict[str, Any]]:
    """Load existing results from the specified directory."""
    overall_path = results_paths['data'] / 'mixed_precision_overall_results.json'
    
    if overall_path.exists():
        print(f"Loading existing results from: {overall_path}")
        try:
            with open(overall_path, 'r') as f:
                results = json.load(f)
            print(f"Successfully loaded {len(results.get('experiments', []))} experiments")
            return results
        except Exception as e:
            print(f"Error loading existing results: {e}")
            return None
    else:
        print(f"No existing results found at: {overall_path}")
        return None

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
    print("Double vs Mixed vs Single Precision Comparison")
    print("=" * 60)
    
    # Define the specific mixed precision configuration
    mixed_precision_config = {
        # Double precision operations (critical for accuracy)
        'kernel_train_gen': 'double',
        'kernel_cross_gen': 'double', 
        'kernel_test_gen': 'double',
        'chol_train': 'double',
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
    
    # Full single precision configuration (for comparison)
    full_single_config = {
        'kernel_train_gen': 'double',
        'kernel_cross_gen': 'double', 
        'kernel_test_gen': 'double',
        'chol_train': 'single',
        'solve_train_cross': 'single',
        'gemm_train': 'single',
        'cov_subtraction': 'double',
        'chol_cond': 'single',
        'solve_train_y': 'single',
        'gemv_train': 'single',
        'solve_cond': 'single',
        'log_diag': 'single',
        'inner_product': 'single'
    }
    
    results = {
        'n_values': n_values,
        'nu_values': nu_values,
        'quality_values': quality_values or ['best'],
        'mixed_precision_config': mixed_precision_config,
        'full_single_config': full_single_config,
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
                                        print(f"\nExperiment {exp_count}/{total_experiments}: quality={quality}, n={n}, at={time_scale}, sep={seperablity}")

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
                                    single_ll_values = []
                                    mixed_double_differences = []
                                    single_double_differences = []
                                    mixed_relative_errors = []
                                    single_relative_errors = []
                                    double_times = []
                                    mixed_times = []
                                    single_times = []
                                    
                                    # Success counters for each precision mode
                                    double_successes = 0
                                    mixed_successes = 0
                                    single_successes = 0

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
                                            # print(f"Double llh: {ll_double}")
                                            
                                            # Compute mixed precision log-likelihood with timing
                                            start_time = time.time()
                                            ll_mixed, diag_mixed = gp.conditional_log_likelihood_ablation(
                                                outer_points, y_true_outer, inner_points, y_true_inner, mixed_precision_config
                                            )
                                            mixed_time = time.time() - start_time
                                            # print(f"Mixed llh: {ll_mixed}")
                                            
                                            # Compute single precision log-likelihood with timing
                                            start_time = time.time()
                                            ll_single, diag_single = gp.conditional_log_likelihood_ablation(
                                                outer_points, y_true_outer, inner_points, y_true_inner, full_single_config
                                            )
                                            single_time = time.time() - start_time
                                            # print(f"Single llh: {ll_single}")
                                            
                                            # Track individual successes
                                            double_success = ll_double is not np.nan
                                            mixed_success = ll_mixed is not np.nan
                                            single_success = ll_single is not np.nan
                                            
                                            if double_success:
                                                double_successes += 1
                                                double_ll_values.append(ll_double)
                                                double_times.append(double_time)
                                            
                                            if mixed_success:
                                                mixed_successes += 1
                                                mixed_ll_values.append(ll_mixed)
                                                mixed_times.append(mixed_time)
                                            
                                            if single_success:
                                                single_successes += 1
                                                single_ll_values.append(ll_single)
                                                single_times.append(single_time)
                                            
                                            # Only compute comparisons if double succeeded and at least one other succeeded
                                            if double_success and (mixed_success or single_success):
                                                if mixed_success:
                                                    # Mixed vs double comparison
                                                    mixed_diff = abs(ll_mixed - ll_double)
                                                    mixed_double_differences.append(mixed_diff)
                                                    mixed_rel_error = mixed_diff / abs(ll_double) if abs(ll_double) > 1e-12 else 0
                                                    mixed_relative_errors.append(mixed_rel_error)
                                                
                                                if single_success:
                                                    # Single vs double comparison
                                                    single_diff = abs(ll_single - ll_double)
                                                    single_double_differences.append(single_diff)
                                                    single_rel_error = single_diff / abs(ll_double) if abs(ll_double) > 1e-12 else 0
                                                    single_relative_errors.append(single_rel_error)
                                                
                                                # print(f" ✓ (mixed diff: {mixed_diff:.2e}, single diff: {single_diff:.2e})")
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
                                        'double_success_rate': double_successes / n_trials * 100,
                                        'mixed_success_rate': mixed_successes / n_trials * 100,
                                        'single_success_rate': single_successes / n_trials * 100,
                                        
                                        # Log-likelihood values
                                        'double_ll_mean': np.mean(double_ll_values) if double_ll_values else np.nan,
                                        'double_ll_std': np.std(double_ll_values) if double_ll_values else np.nan,
                                        'mixed_ll_mean': np.mean(mixed_ll_values) if mixed_ll_values else np.nan,
                                        'mixed_ll_std': np.std(mixed_ll_values) if mixed_ll_values else np.nan,
                                        'single_ll_mean': np.mean(single_ll_values) if single_ll_values else np.nan,
                                        'single_ll_std': np.std(single_ll_values) if single_ll_values else np.nan,
                                        
                                        # Mixed vs Double comparison
                                        'mixed_mean_ll_difference': np.mean(mixed_double_differences) if mixed_double_differences else np.nan,
                                        'mixed_max_ll_difference': np.max(mixed_double_differences) if mixed_double_differences else np.nan,
                                        'mixed_mean_relative_error': np.mean(mixed_relative_errors) if mixed_relative_errors else np.nan,
                                        'mixed_max_relative_error': np.max(mixed_relative_errors) if mixed_relative_errors else np.nan,
                                        
                                        # Single vs Double comparison
                                        'single_mean_ll_difference': np.mean(single_double_differences) if single_double_differences else np.nan,
                                        'single_max_ll_difference': np.max(single_double_differences) if single_double_differences else np.nan,
                                        'single_mean_relative_error': np.mean(single_relative_errors) if single_relative_errors else np.nan,
                                        'single_max_relative_error': np.max(single_relative_errors) if single_relative_errors else np.nan,
                                        
                                        # Timing and speedup
                                        'double_time_mean': np.mean(double_times) if double_times else np.nan,
                                        'double_time_std': np.std(double_times) if double_times else np.nan,
                                        'mixed_time_mean': np.mean(mixed_times) if mixed_times else np.nan,
                                        'mixed_time_std': np.std(mixed_times) if mixed_times else np.nan,
                                        'single_time_mean': np.mean(single_times) if single_times else np.nan,
                                        'single_time_std': np.std(single_times) if single_times else np.nan,
                                        'mixed_speedup_mean': np.mean([dt/mt for dt, mt in zip(double_times, mixed_times) if mt > 0]) if double_times and mixed_times else np.nan,
                                        'mixed_speedup_std': np.std([dt/mt for dt, mt in zip(double_times, mixed_times) if mt > 0]) if double_times and mixed_times else np.nan,
                                        'single_speedup_mean': np.mean([dt/st for dt, st in zip(double_times, single_times) if st > 0]) if double_times and single_times else np.nan,
                                        'single_speedup_std': np.std([dt/st for dt, st in zip(double_times, single_times) if st > 0]) if double_times and single_times else np.nan,
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
                                        f"  Results: Mixed LL diff = {experiment_result['mixed_mean_ll_difference']:.2e}, "
                                        f"Single LL diff = {experiment_result['single_mean_ll_difference']:.2e}, "
                                        f"Mixed speedup = {experiment_result['mixed_speedup_mean']:.2f}x, "
                                        f"Single speedup = {experiment_result['single_speedup_mean']:.2f}x",
                                    )
    
    return results

def visualize_mixed_precision_results(results: Dict[str, Any], results_paths: Optional[Dict[str, Path]] = None):
    """Create comprehensive visualizations of mixed precision comparison results."""
    
    experiments = results['experiments']
    n_values = results['n_values']
    quality_values = results.get('quality_values', ['best'])
    
    # Extract unique parameter combinations
    separability_values = sorted(list(set(exp['seperablity'] for exp in experiments)))
    time_scale_values = sorted(list(set(exp['time_scale'] for exp in experiments)))
    
    # Create parameter combination labels (without quality - quality will be in subplots)
    param_combinations = []
    # for n in n_values:
    #     for sep in separability_values:
    #         for ts in time_scale_values:
    #             param_combinations.append(f'n={n}\ns={sep:.1f}\nτ={ts:.1f}')
    for sep in separability_values:
        for ts in time_scale_values:
            param_combinations.append(f'$a_t$={ts:.1f}\n$\\beta$={sep:.1f}')
    

    n_combinations = len(param_combinations)
    
    # Prepare data arrays for each quality
    mixed_success_rates_by_quality = {q: np.full(n_combinations, np.nan) for q in quality_values}
    single_success_rates_by_quality = {q: np.full(n_combinations, np.nan) for q in quality_values}
    mixed_mean_errors_by_quality = {q: np.full(n_combinations, np.nan) for q in quality_values}
    single_mean_errors_by_quality = {q: np.full(n_combinations, np.nan) for q in quality_values}
    
    # Fill data arrays
    for quality in quality_values:
        combo_idx = 0
        for n in n_values:
            for sep in separability_values:
                for ts in time_scale_values:
                    subset = [exp for exp in experiments 
                             if exp['n'] == n and exp['seperablity'] == sep 
                             and exp['time_scale'] == ts and exp['quality'] == quality]
                    if subset:
                        mixed_success_rates_by_quality[quality][combo_idx] = np.nanmean([exp['mixed_success_rate'] for exp in subset])
                        single_success_rates_by_quality[quality][combo_idx] = np.nanmean([exp['single_success_rate'] for exp in subset])
                        mixed_mean_errors_by_quality[quality][combo_idx] = np.mean([exp['mixed_mean_ll_difference'] for exp in subset])
                        single_mean_errors_by_quality[quality][combo_idx] = np.mean([exp['single_mean_ll_difference'] for exp in subset])
                    combo_idx += 1
    
    # Generate suffix for filenames based on parameters
    n_str = "_".join(map(str, n_values))
    sep_str = "_".join([f"{s:.1f}" for s in separability_values])
    ts_str = "_".join([f"{t:.1f}" for t in time_scale_values])
    param_suffix = f"_n{n_str}_s{sep_str}_t{ts_str}"
    
    # Plot 1: Success Rate - separate plot for each quality (with grouped bars)
    for quality in quality_values:
        plt.figure(figsize=(14, 6))
        mixed_success_rates = mixed_success_rates_by_quality[quality]
        single_success_rates = single_success_rates_by_quality[quality]
        
        # Set up bar positions for grouped bars
        bar_width = 0.35
        x_positions = np.arange(n_combinations)
        
        # Create grouped bars
        bars_mixed = plt.bar(x_positions - bar_width/2, mixed_success_rates, bar_width, 
                           label='Mixed Precision', color='orange', alpha=0.7)
        bars_single = plt.bar(x_positions + bar_width/2, single_success_rates, bar_width, 
                            label='Single Precision', color='red', alpha=0.7)
        
        plt.xlabel('Parameter Combinations', fontsize=16)
        plt.ylabel('Success Rate (%)', fontsize=16)
        # plt.title(f'Success Rate - Quality: {quality.capitalize()}', fontsize=18, fontweight='bold')
        plt.xticks(x_positions, param_combinations, rotation=0, ha='center', fontsize=16)
        plt.ylim(0, 105)
        plt.legend(fontsize=14)
        
        # Add value labels on bars
        for i, (bar_mixed, bar_single, rate_mixed, rate_single) in enumerate(zip(bars_mixed, bars_single, mixed_success_rates, single_success_rates)):
            if not np.isnan(rate_mixed) and rate_mixed < 99:
                plt.text(bar_mixed.get_x() + bar_mixed.get_width()/2, bar_mixed.get_height() + 1, 
                        f'{rate_mixed:.0f}%', ha='center', va='bottom', fontsize=14, rotation=45)
            if not np.isnan(rate_single) and rate_single < 99:
                plt.text(bar_single.get_x() + bar_single.get_width()/2, bar_single.get_height() + 1, 
                        f'{rate_single:.0f}%', ha='center', va='bottom', fontsize=14, rotation=45)
        
        plt.tight_layout()
        if results_paths:
            figures_dir = results_paths.get('figures', results_paths['root'])
            success_path = figures_dir / f'mixed_precision_comparison_success_rate_{quality}{param_suffix}.pdf'
            plt.savefig(success_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.savefig(f'./fig/mixed_precision_comparison_success_rate_{quality}{param_suffix}.pdf', dpi=300, bbox_inches='tight')
            plt.close()
    
    # Plot 2: Mean Error - separate plot for each quality (with log scale)
    for quality in quality_values:
        plt.figure(figsize=(14, 6))
        mixed_errors = mixed_mean_errors_by_quality[quality]
        single_errors = single_mean_errors_by_quality[quality]
        
        # Set up bar positions for grouped bars
        bar_width = 0.35
        x_positions = np.arange(n_combinations)
        
        # Create grouped bars
        bars_mixed = plt.bar(x_positions - bar_width/2, mixed_errors, bar_width, 
                           label='Mixed Precision', color='orange', alpha=0.7)
        bars_single = plt.bar(x_positions + bar_width/2, single_errors, bar_width, 
                            label='Single Precision', color='red', alpha=0.7)
        
        plt.xlabel('Parameter Combinations', fontsize=16)
        plt.ylabel('Mean Log-Likelihood Difference', fontsize=16)
        # plt.title(f'Mean Error - Quality: {quality.capitalize()}', fontsize=18, fontweight='bold')
        plt.xticks(x_positions, param_combinations, rotation=0, ha='center', fontsize=16)
        plt.yscale('log')
        plt.legend(fontsize=14)
        
        # Add horizontal background dashed lines for reference
        y_ticks = plt.yticks()[0]
        for y in y_ticks:
            plt.axhline(y, color='gray', linestyle='dashed', linewidth=0.7, alpha=0.5, zorder=0)

        # Add value labels on bars
        for i, (bar_mixed, bar_single, error_mixed, error_single) in enumerate(zip(bars_mixed, bars_single, mixed_errors, single_errors)):
            if not np.isnan(error_mixed) and error_mixed > 0:
                plt.text(bar_mixed.get_x() + bar_mixed.get_width()/2, bar_mixed.get_height() * 1.1, 
                        f'{error_mixed:.1e}', ha='center', va='bottom', fontsize=14, rotation=45)
            if not np.isnan(error_single) and error_single > 0:
                plt.text(bar_single.get_x() + bar_single.get_width()/2, bar_single.get_height() * 1.1, 
                        f'{error_single:.1e}', ha='center', va='bottom', fontsize=14, rotation=45)
        
        plt.tight_layout()
        if results_paths:
            max_error_path = figures_dir / f'mixed_precision_comparison_mean_error_{quality}.pdf'
            plt.savefig(max_error_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.savefig(f'./fig/mixed_precision_comparison_mean_error_{quality}.pdf', dpi=300, bbox_inches='tight')
            plt.close()
    

def create_summary_table(results: Dict[str, Any], results_paths: Optional[Dict[str, Path]] = None):
    """Create a summary table of the results."""
    experiments = results['experiments']
    
    lines = []
    lines.append("=" * 80)
    lines.append("DOUBLE vs MIXED vs SINGLE PRECISION COMPARISON SUMMARY")
    lines.append("=" * 80)

    lines.append("\nMixed Precision Configuration:")
    for op, precision in results['mixed_precision_config'].items():
        lines.append(f"  {op}: {precision}")
        
    lines.append("\nSingle Precision Configuration:")
    for op, precision in results['full_single_config'].items():
        lines.append(f"  {op}: {precision}")

    header = (
        f"\n{'n':>4} {'ν':>4} {'Mixed Succ%':>11} {'Single Succ%':>12} "
        f"{'Mixed LL Diff':>13} {'Single LL Diff':>14} "
        f"{'Mixed Rel Err':>13} {'Single Rel Err':>14} "
        f"{'Double Time':>12} {'Mixed Time':>12} {'Single Time':>12} "
        f"{'Mixed Speedup':>13} {'Single Speedup':>14}"
    )
    lines.append(header)
    lines.append("-" * 185)

    for exp in experiments:
        line = (
            f"{exp['n']:>4} {exp['nu']:>4.1f} {exp['mixed_success_rate']:>10.0f}% {exp['single_success_rate']:>11.0f}% "
            f"{exp['mixed_mean_ll_difference']:>12.2e} {exp['single_mean_ll_difference']:>13.2e} "
            f"{exp['mixed_mean_relative_error']:>12.2e} {exp['single_mean_relative_error']:>13.2e} "
            f"{exp['double_time_mean']:>11.3f}s {exp['mixed_time_mean']:>11.3f}s {exp['single_time_mean']:>11.3f}s "
            f"{exp['mixed_speedup_mean']:>12.2f}x {exp['single_speedup_mean']:>13.2f}x"
        )
        lines.append(line)

    # Overall statistics
    all_mixed_mean_diffs = [exp['mixed_mean_ll_difference'] for exp in experiments if not np.isnan(exp['mixed_mean_ll_difference'])]
    all_single_mean_diffs = [exp['single_mean_ll_difference'] for exp in experiments if not np.isnan(exp['single_mean_ll_difference'])]
    all_mixed_rel_errors = [exp['mixed_mean_relative_error'] for exp in experiments if not np.isnan(exp['mixed_mean_relative_error'])]
    all_single_rel_errors = [exp['single_mean_relative_error'] for exp in experiments if not np.isnan(exp['single_mean_relative_error'])]
    all_mixed_success_rates = [exp['mixed_success_rate'] for exp in experiments]
    all_single_success_rates = [exp['single_success_rate'] for exp in experiments]
    all_double_times = [exp['double_time_mean'] for exp in experiments if not np.isnan(exp['double_time_mean'])]
    all_mixed_times = [exp['mixed_time_mean'] for exp in experiments if not np.isnan(exp['mixed_time_mean'])]
    all_single_times = [exp['single_time_mean'] for exp in experiments if not np.isnan(exp['single_time_mean'])]
    all_mixed_speedups = [exp['mixed_speedup_mean'] for exp in experiments if not np.isnan(exp['mixed_speedup_mean'])]
    all_single_speedups = [exp['single_speedup_mean'] for exp in experiments if not np.isnan(exp['single_speedup_mean'])]

    lines.append("-" * 185)
    lines.append("Overall Statistics:")
    lines.append("  Success Rates:")
    lines.append(f"    Mixed Precision: {np.mean(all_mixed_success_rates):.1f}%")
    lines.append(f"    Single Precision: {np.mean(all_single_success_rates):.1f}%")
    lines.append("")
    lines.append("  Mixed Precision vs Double:")
    lines.append(f"    Average Mean LL Difference: {np.mean(all_mixed_mean_diffs):.2e}")
    lines.append(f"    Average Mean Relative Error: {np.mean(all_mixed_rel_errors):.2e}")
    lines.append(f"    Maximum LL Difference: {np.max([exp['mixed_max_ll_difference'] for exp in experiments if not np.isnan(exp['mixed_max_ll_difference'])]):.2e}")
    lines.append(f"    Average Speedup: {np.mean(all_mixed_speedups):.2f}x")
    lines.append(f"    Maximum Speedup: {np.max(all_mixed_speedups):.2f}x")
    lines.append("")
    lines.append("  Single Precision vs Double:")
    lines.append(f"    Average Mean LL Difference: {np.mean(all_single_mean_diffs):.2e}")
    lines.append(f"    Average Mean Relative Error: {np.mean(all_single_rel_errors):.2e}")
    lines.append(f"    Maximum LL Difference: {np.max([exp['single_max_ll_difference'] for exp in experiments if not np.isnan(exp['single_max_ll_difference'])]):.2e}")
    lines.append(f"    Average Speedup: {np.mean(all_single_speedups):.2f}x")
    lines.append(f"    Maximum Speedup: {np.max(all_single_speedups):.2f}x")
    lines.append("")
    lines.append("  Timing Summary:")
    lines.append(f"    Average Double Time: {np.mean(all_double_times):.3f}s")
    lines.append(f"    Average Mixed Time: {np.mean(all_mixed_times):.3f}s")
    lines.append(f"    Average Single Time: {np.mean(all_single_times):.3f}s")

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
    force_rerun: bool = False,
) -> Dict[str, Any]:
    """Main function to run mixed precision comparison."""
    
    # If results_root is provided and we're not forcing rerun, check if it exists
    # and disable timestamp to use the existing directory
    if results_root and not force_rerun:
        potential_path = Path(results_root)
        if potential_path.exists() and (potential_path / "data" / "mixed_precision_overall_results.json").exists():
            print(f"Found existing results at: {results_root}")
            timestamp = False  # Use existing directory structure
    
    results_paths = prepare_results_dir(
        root=results_root,
        timestamp=timestamp,
        subdirs=["figures", "data", "logs"],
    )

    # Try to load existing results first (unless forced to rerun)
    results = None
    if not force_rerun:
        results = load_existing_results(results_paths)
    
    if results is None:
        print("Running new experiments...")
        
        # Experimental parameters (matching operation_llh_error_impact_analysis.py settings)
        n_quality = ['best', 'good', 'worst']  # , 'good', 'worst'

        n_values = [100]  # Reduced from [10, 100, 1000]
        nu_values = [1.5]  # Reduced from [0.5, 1.5, 2.5]
        separability_grid = [0.0, 0.5, 1.0]
        time_scale_grid = [1,  5, 10]
        dim_scale_grid = [[0.23, 0.23]]
        dim_length_grid = [2]
        time_lag_grid = [2]
        nu_time_grid = [0.5]

        n_trials_default = 10
        n_trials = n_trials_override or n_trials_default

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

        # Save the results
        overall_path = results_paths['data'] / 'mixed_precision_overall_results.json'
        save_json(results, overall_path)
        print(f"Experiment results saved to: {overall_path}")
    else:
        print("Using existing results, skipping experiments...")

    # Create visualizations
    print("\nCreating visualizations...")
    visualize_mixed_precision_results(results, results_paths=results_paths)

    # Create summary table
    print("\nCreating summary table...")
    create_summary_table(results, results_paths=results_paths)

    print(f"All outputs saved to: {results_paths['root']}")

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
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force rerunning experiments even if results already exist.",
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
        force_rerun=args.force_rerun,
    )
