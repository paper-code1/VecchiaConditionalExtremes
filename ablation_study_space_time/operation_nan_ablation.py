#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json

from scipy.special import gamma, kv
from scipy.linalg import cholesky, solve_triangular
import time
from typing import Tuple, Dict, Any, List, Optional
import warnings
import matplotlib
from pathlib import Path

matplotlib.rcParams.update({'font.size': 18})

from utils import (
    MaternKernel,
    MaternKernelHighDimTime,
    AblationGaussianProcess,
    generate_circle_points,
    prepare_results_dir,
    save_json,
    format_experiment_filename,
)

def run_ablation_experiment(
    n: int,
    nu: float,
    n_trials: int = 5,
    quality: str = 'best',
    seperablity: float = 0.0,
    time_scale: float = 1.0,
    dim_scale: List[float] = [0.05, 0.05, 0.05, 5, 5, 5, 5, 5, 5, 5],
    dim_length: int = 10,
    time_lag: int = 2,
    nu_time: float = 0.5,
    results_paths: Optional[Dict[str, Path]] = None,
) -> Dict[str, Any]:
    """Run ablation study for specific n and nu values."""
    print(f"Running ablation study: n={n}, ν={nu}, \
        seperablity={seperablity}, time_scale={time_scale}, \
        dim_scale={dim_scale}, dim_length={dim_length}, \
        time_lag={time_lag}, nu_time={nu_time}")
    
    # Create kernel and GP
    kernel = MaternKernelHighDimTime(nu_space=nu, nu_time=nu_time, variance=1.0,
                                    length_scale=dim_scale, length_dim=dim_length,
                                    time_scale=time_scale, time_lag=time_lag,
                                    seperablity=seperablity)
    gp = AblationGaussianProcess(kernel, noise_variance=1e-5) # no jitter
    
    all_points = []
    inner_points = []
    outer_points = []
    y_true_all = []
    y_true_outer = []
    y_true_inner = []

    # Generate spatial points and reference function
    np.random.seed(42 + n + int(nu*10))  # Reproducible
    for trial in range(n_trials):
        _all_points, _inner_points, _outer_points = generate_circle_points(
            n, time_lag=time_lag, 
            quality=quality, dim_length=dim_length
        )
    
        # Generate reference function values (always double precision)
        _y_true_all = np.random.multivariate_normal(
            np.zeros(len(_all_points)), 
            # kernel(_all_points, precision='double') # no jitter
            kernel(_all_points, precision='double') + 1e-5 * np.eye(len(_all_points))
        )
        _y_true_inner = _y_true_all[:len(_inner_points)]
        _y_true_outer = _y_true_all[len(_inner_points):len(_inner_points)+len(_outer_points)]

        all_points.append(_all_points)
        inner_points.append(_inner_points)
        outer_points.append(_outer_points)
        y_true_all.append(_y_true_all)
        y_true_outer.append(_y_true_outer)
        y_true_inner.append(_y_true_inner)
    
    # Define atomic operations to test
    operations = [
        # 'kernel_train_gen', 
        # 'kernel_test_gen',
        # 'kernel_cross_gen',
        'chol_train', 'solve_train_y', 'solve_train_cross', 'gemv_train',
        'gemm_train', 
        # 'cov_subtraction', 
        'chol_cond', 
        'solve_cond', 'log_diag', 'inner_product'
    ]
    
    results = {
        'n': n,
        'nu': nu,
        'nu_time': nu_time,
        'seperablity': seperablity,
        'time_scale': time_scale,
        'dim_scale': dim_scale,
        'dim_length': dim_length,
        'time_lag': time_lag,
        'quality': quality,
        'n_inner': np.mean([len(inner_points[i]) for i in range(n_trials)]),
        'n_outer': np.mean([len(outer_points[i]) for i in range(n_trials)]),
        'baseline_ll': None,
        'operation_results': {}
    }
    
    # Compute baseline (all double precision)
    baseline_ll_values = []
    baseline_diagnostics = []
    
    for trial in range(n_trials):
        ll, diag = gp.conditional_log_likelihood_ablation(
            outer_points[trial], y_true_outer[trial], inner_points[trial], y_true_inner[trial], {}
        )   
        # if ll  -1e9:  # Valid result
        baseline_ll_values.append(ll)
        baseline_diagnostics.append(diag)
    # print(baseline_ll_values)
    if baseline_ll_values:
        results['baseline_ll'] = np.mean(baseline_ll_values)
        results['baseline_std'] = np.std(baseline_ll_values)
        results['baseline_cond_train'] = np.mean([d['cond_K_train'] for d in baseline_diagnostics])
        results['baseline_cond_cov'] = np.mean([d['cond_cov_matrix'] for d in baseline_diagnostics])
        results['baseline_ll_trials'] = baseline_ll_values
    else:
        print(f"  Baseline failed for n={n}, ν={nu}")
        return results
    
    # Test each operation in single precision
    for op in operations:
        print(f"  Testing operation: {op}")
        op_precision = {op: 'single'}
        
        ll_values = []
        ll_errors = []
        cond_trains = []
        cond_covs = []
        failures = 0
        
        for trial in range(n_trials):
            ll, diag = gp.conditional_log_likelihood_ablation(
                outer_points[trial], y_true_outer[trial], inner_points[trial], y_true_inner[trial], op_precision
            )
            print(f"Trial {trial}, ll single: {ll}")
            if ll is np.nan:  # Failed
                failures += 1
            else:
                ll_values.append(ll)
                ll_errors.append(abs(ll - results['baseline_ll']))
                cond_trains.append(diag.get('cond_K_train', np.nan))
                cond_covs.append(diag.get('cond_cov_matrix', np.nan))
        
        # Store results
        results['operation_results'][op] = {
            'success_rate': (n_trials - failures) / n_trials * 100,
            'mean_ll': np.mean(ll_values) if ll_values else np.nan,
            'std_ll': np.std(ll_values) if ll_values else np.nan,
            'mean_ll_error': np.mean(ll_errors) if ll_errors else np.nan,
            'max_ll_error': np.max(ll_errors) if ll_errors else np.nan,
            'mean_cond_train': np.mean([c for c in cond_trains if not np.isnan(c)]) if cond_trains else np.nan,
            'mean_cond_cov': np.mean([c for c in cond_covs if not np.isnan(c)]) if cond_covs else np.nan,
            'failures': failures,
        }

    if results_paths is not None:
        data_dir = results_paths.get('data', results_paths['root'])
        experiment_params = {
            'quality': quality,
            'n': n,
            'nu': nu,
            'seperablity': seperablity,
            'time_scale': time_scale,
            'dim_scale': dim_scale,
            'dim_length': dim_length,
            'time_lag': time_lag,
            'nu_time': nu_time,
        }
        filename = format_experiment_filename('nan_ablation', experiment_params)
        save_json(results, data_dir / filename)
    
    return results

def load_results_from_json(results_file: str) -> List[Dict[str, Any]]:
    """Load results from a JSON file."""
    print(f"Loading results from: {results_file}")
    with open(results_file, 'r') as f:
        results = json.load(f)
    print(f"Loaded {len(results)} experiment results")
    return results

def visualize_ablation_results(all_results: List[Dict[str, Any]]):
    """Create comprehensive visualizations of ablation study results focusing on n, separability, and time_scale."""
    
    # Extract data organization - focus only on the three key parameters
    n_values = sorted(set(r['n'] for r in all_results))
    seperablity_values = sorted(set(r['seperablity'] for r in all_results))
    time_scale_values = sorted(set(r['time_scale'] for r in all_results))
    
    operations = [
        # 'kernel_train_gen', 'kernel_test_gen', 'kernel_cross_gen',
        'chol_train', 'solve_train_y', 'solve_train_cross', 'gemv_train',
        'gemm_train', 
        # 'cov_subtraction', 
        'chol_cond', 
        'solve_cond', 'log_diag', 'inner_product'
    ]
    
    # Create subplots - one for each n value
    # Increased figure size for better readability
    fig, axes = plt.subplots(1, len(n_values), figsize=(12 * len(n_values), 5))
    if len(n_values) == 1:
        axes = [axes]
    
    # For each n, create a heatmap of operations vs (time_scale, separability) combinations
    for n_idx, n in enumerate(n_values):
        ax = axes[n_idx]
        
        # Create combinations of time_scale and separability
        ts_sep_combinations = [(ts, sep) for ts in time_scale_values for sep in seperablity_values]
        success_data = np.zeros((len(operations), len(ts_sep_combinations)))
        
        # Fill success data
        for i, op in enumerate(operations):
            for j, (time_scale, separability) in enumerate(ts_sep_combinations):
                success_rates = []
                for result in all_results:
                    if (result['n'] == n and 
                        result['seperablity'] == separability and 
                        result['time_scale'] == time_scale and 
                        op in result.get('operation_results', {})):
                        success_rates.append(result['operation_results'][op]['success_rate'])
                
                success_data[i, j] = np.mean(success_rates) if success_rates else 0
        
        # Create heatmap
        im = ax.imshow(success_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        # Set labels - keep all x-labels, only show y-labels on leftmost subplot
        ax.set_xticks(range(len(ts_sep_combinations)))
        ax.set_xticklabels([f'$a_t$={ts:.1f}\n$\\beta$={sep:.1f}' for ts, sep in ts_sep_combinations], 
        # rotation=45, 
        ha='center', fontsize=15)
        
        ax.set_yticks(range(len(operations)))
        if n_idx == 0:  # Only leftmost subplot gets y-labels
            ax.set_yticklabels(operations, rotation=0, fontsize=15)
        else:
            ax.set_yticklabels([])  # Empty labels for other subplots
        # ax.set_title(f'n = {n}')
        
        # Add colorbar to the rightmost subplot
        if n_idx == len(n_values) - 1:
            plt.colorbar(im, ax=ax, label='Success Rate (%)')
        
    
        # Add text annotations with smaller font size
        for i in range(len(operations)):
            for j in range(len(ts_sep_combinations)):
                # text_color = "black" if success_data[i, j] > 50 else "white"
                text_color = "black"
                # ax.text(j, i, f'{success_data[i, j]:.0f}%',
                #        ha="center", va="center", color=text_color, fontsize=14)
                ax.text(j, i, f'{success_data[i, j]:.1f}',
                       ha="center", va="center", color=text_color, fontsize=15)
    
    # plt.suptitle('Success Rate (%) by Operation, Time Scale, and Separability for Different Sample Sizes', fontsize=16)
    # Add x label for the main figure (shared for all subplots)
    # fig.supxlabel('(Time scale, Separability)', fontsize=18)
    plt.tight_layout()
    return fig

def main(results_root: Optional[str] = None, timestamp: bool = True, load_results: Optional[str] = None) -> None:
    """Main function to run ablation study or load existing results for visualization."""
    print("Precision Ablation Study for Conditional Log-Likelihood")
    print("=" * 60)
    
    results_paths = prepare_results_dir(
        "operation_nan_ablation",
        root=results_root,
        timestamp=timestamp,
        subdirs=["figures", "data", "logs"],
    )

    # If loading existing results, skip experiments and go straight to visualization
    if load_results:
        print(f"Loading existing results from: {load_results}")
        all_results = load_results_from_json(load_results)
        
        # Create visualizations for each quality type
        quality_types = sorted(set(r.get('quality', 'unknown') for r in all_results))
        for quality in quality_types:
            print(f"\nCreating visualizations for quality: {quality}...")
            quality_results = [res for res in all_results if res.get('quality') == quality]
            if quality_results:
                fig = visualize_ablation_results(quality_results)
                figure_path = results_paths['figures'] / f'operation_nan_ablation_results_{quality}.pdf'
                plt.savefig(figure_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved visualization to: {figure_path}")
        
        print(f"\nVisualization complete! Check: {results_paths['figures']}")
        return

    # Original experiment code continues here...

    # Experimental parameters (reduced for faster execution)
    n_quality = ['best', 'good', 'worst']  # best: the best approximation, good: the good approximation, worst: the worst approximation['best', 'good', 'worst']
    n_values = [100]  # Reduced from [10, 100, 1000]
    n_seperablity = [0.0,  0.5, 1.0] # seperablity[0.0, 0.5, 1.0]
    n_time_scale = [1,  5, 10] # time scale
    n_dim_scale = [[0.23, 0.23]] # dimension scale
    n_dim_length = [2] # dimension length
    nu_values = [1.5]  # Reduced from [0.5, 1.5, 2.5]
    n_time_lag = [2] # time lag
    n_nu_time = [0.5] # time smoothness
    
    n_trials = 10  # Reduced from 10
    
    all_results = []
    
    # Run experiments
    total_experiments = len(n_values) * len(nu_values) * len(n_quality) * len(n_seperablity) * len(n_time_scale) * len(n_dim_scale) * len(n_dim_length) * len(n_time_lag) * len(n_nu_time)
    
    exp_count = 0
    
    for quality in n_quality:
        for n in n_values:
            for nu in nu_values:
                for seperablity in n_seperablity:
                    for time_scale in n_time_scale:
                        for dim_scale in n_dim_scale:
                            for dim_length in n_dim_length:
                                for time_lag in n_time_lag:
                                    for nu_time in n_nu_time:
                                        exp_count += 1
                                        print(f"\nExperiment {exp_count}/{total_experiments}")
                                        
                                        result = run_ablation_experiment(
                                            n, nu, n_trials, quality, seperablity, 
                                            time_scale, dim_scale, dim_length, time_lag, 
                                            nu_time, results_paths=results_paths)
                                        all_results.append(result)
        
        # Create visualizations
        print("\nCreating visualizations...")
        quality_results = [res for res in all_results if res.get('quality') == quality]
        fig = visualize_ablation_results(quality_results)
        figure_path = results_paths['figures'] / f'operation_nan_ablation_results_{quality}.pdf'
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        summary_path = results_paths['data'] / f'operation_nan_ablation_summary_{quality}.json'
        save_json(quality_results, summary_path)
        # plt.show()
    
    # Print detailed summary
    overall_path = results_paths['data'] / 'operation_nan_ablation_all_results.json'
    save_json(all_results, overall_path)
    print("\n" + "=" * 60)
    print("ABLATION STUDY SUMMARY")
    print("=" * 60)
    print(f"Results saved to: {results_paths['root']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the precision ablation experiment.")
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
        "--load-results",
        type=str,
        default=None,
        help="Path to existing results JSON file to load for visualization only. Skips running experiments.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    args = parse_args()
    main(results_root=args.results_root, timestamp=not args.no_timestamp, load_results=args.load_results)
