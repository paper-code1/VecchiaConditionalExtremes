#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import time
import json
from typing import Tuple, Dict, Any, List, Optional
import warnings
from pathlib import Path
from matplotlib.colors import LogNorm
from utils import (
    MaternKernel,
    MaternKernelHighDimTime,
    AblationGaussianProcess,
    generate_circle_points,
    prepare_results_dir,
    save_json,
    load_json,
    load_all_experiment_results,
    format_experiment_filename
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def run_ablation_experiment(
    n: int,
    nu: float,
    n_trials: int = 5,
    quality: str = 'best',
    seperablity: float = 0.0,
    time_scale: float = 1.0,
    dim_scale: Optional[List[float]] = None,
    dim_length: int = 10,
    time_lag: int = 2,
    nu_time: float = 0.5,
    results_paths: Optional[Dict[str, Path]] = None,
) -> Dict[str, Any]:
    """Run ablation study for specific n and nu values."""
    print(f"Running experiment: n={n}, ν={nu}")
    
    # Create kernel and GP
    if dim_scale is None:
        dim_scale = [0.05, 0.05, 0.05, 5, 5, 5, 5, 5, 5, 5]

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
    
    # Generate spatial points and reference function
    all_points = []
    inner_points = []
    outer_points = []
    y_true_all = []
    y_true_outer = []
    y_true_inner = []

    # Reproducible
    # np.random.seed(42 + n + int(nu*10))  
    np.random.seed(42)
    for trial in range(n_trials):
        _all_points, _inner_points, _outer_points = generate_circle_points(
            n,
            quality=quality,
            time_lag=time_lag,
            dim_length=dim_length,
        )
        
        # Generate reference function values (always double precision)
        _y_true_all = np.random.multivariate_normal(
            np.zeros(len(_all_points)), 
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
        # 'kernel_train_gen', 'kernel_test_gen', 'kernel_cross_gen',
        'chol_train', 'solve_train_y', 'gemv_train', 
        'solve_train_cross',
        'gemm_train', 
        # 'cov_subtraction',
        'chol_cond', 'solve_cond', 'log_diag', 'inner_product'
    ]
    
    results = {
        'n': n,
        'nu': nu,
        'quality': quality,
        'seperablity': seperablity,
        'time_scale': time_scale,
        'dim_scale': dim_scale,
        'dim_length': dim_length,
        'time_lag': time_lag,
        'nu_time': nu_time,
        'n_trials': n_trials,
        'n_inner': np.mean([len(inner_points[i]) for i in range(n_trials)]),
        'n_outer': np.mean([len(outer_points[i]) for i in range(n_trials)]),
        'baseline_ll': None,
        'baseline_ll_all_trials': [],  # Store all individual trial results
        'operation_results': {},
        'trial_data': {  # Store detailed per-trial data
            'all_points': [pts.tolist() for pts in all_points],
            'inner_points': [pts.tolist() for pts in inner_points], 
            'outer_points': [pts.tolist() for pts in outer_points],
            'y_true_all': [y.tolist() for y in y_true_all],
            'y_true_inner': [y.tolist() for y in y_true_inner],
            'y_true_outer': [y.tolist() for y in y_true_outer]
        }
    }
    
    # Compute baseline (all double precision)
    baseline_ll_values = []
    baseline_diagnostics = []
    
    for trial in range(n_trials):
        ll, diag = gp.conditional_log_likelihood_ablation(
            outer_points[trial], y_true_outer[trial], inner_points[trial], y_true_inner[trial], {}
        )
        print(f"Trial {trial}, ll double: {ll}")
        if ll is not np.nan:  # Valid result
            baseline_ll_values.append(ll)
            baseline_diagnostics.append(diag)
    
    if baseline_ll_values:
        results['baseline_ll'] = np.mean(baseline_ll_values)  # Mean for compatibility
        results['baseline_ll_all_trials'] = np.asarray(baseline_ll_values).tolist()  # All trials
        results['baseline_std'] = np.std(baseline_ll_values)
        results['baseline_cond_train'] = np.mean([d['cond_K_train'] for d in baseline_diagnostics])
        results['baseline_cond_cov'] = np.mean([d['cond_cov_matrix'] for d in baseline_diagnostics])
        results['baseline_diagnostics'] = baseline_diagnostics  # Store all diagnostics
    else:
        print(f"  Baseline failed for n={n}, ν={nu}")
        return results
    
    # Test each operation in single precision
    for op in operations:
        op_precision = {op: 'single'}
        
        ll_values = []
        ll_errors = []
        failures = 0
        
        for trial in range(n_trials):
            ll, diag = gp.conditional_log_likelihood_ablation(
                outer_points[trial], y_true_outer[trial], inner_points[trial], y_true_inner[trial], op_precision
            )
            print(f"Trial {trial} op {op}, ll single: {ll}")
            if ll is np.nan:  # Failed
                failures += 1
            else:
                ll_values.append(ll)
                # Use the corresponding baseline value for this trial
                if trial < len(baseline_ll_values):
                    ll_errors.append(abs(ll - baseline_ll_values[trial]))
                else:
                    ll_errors.append(np.nan)  # If baseline failed for this trial
        
        # Store results
        results['operation_results'][op] = {
            'success_rate': (n_trials - failures) / n_trials * 100,
            'mean_ll': np.mean(ll_values) if ll_values else np.nan,
            'std_ll': np.std(ll_values) if ll_values else np.nan,
            'mean_ll_error': np.mean(ll_errors) if ll_errors else np.nan,
            'max_ll_error': np.max(ll_errors) if ll_errors else np.nan,
            'failures': failures,
            'll_values_all_trials': ll_values,  # All successful trial values
            'll_errors_all_trials': ll_errors,  # All successful trial errors
        }

    if results_paths is not None:
        data_dir = results_paths.get('data', results_paths['root'])
        experiment_params = {'quality': quality, 'n': n, 'nu': nu}
        filename = format_experiment_filename('llh_error_ablation', experiment_params)
        save_json(results, data_dir / filename)
    
    return results

def create_main_operation_ranking_plot(all_results: List[Dict[str, Any]]):
    """Create overall operation ranking plot based on average error impact with gradual colors."""
    
    operations = [
        # 'kernel_train_gen', 'kernel_test_gen', 'kernel_cross_gen',
        'chol_train', 'solve_train_y', 'gemv_train', 
        'solve_train_cross',
        'gemm_train', 
        # 'cov_subtraction',
        'chol_cond', 'solve_cond', 'log_diag', 'inner_product'
    ]
    
    # Create figure for main overview
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Compute average error across all parameter combinations
    avg_errors = {}
    error_counts = {}
    
    for op in operations:
        errors = []
        for result in all_results:
            if result['baseline_ll'] is not None and op in result.get('operation_results', {}):
                error = result['operation_results'][op]['mean_ll_error']
                if not np.isnan(error):
                    errors.append(error)
        
        avg_errors[op] = np.mean(errors) if errors else np.nan
        error_counts[op] = len(errors)
    
    # Sort operations by average error (highest impact first)
    sorted_ops = sorted([op for op in operations if not np.isnan(avg_errors[op])], 
                       key=lambda x: avg_errors[x], reverse=True)
    
    if sorted_ops:
        y_pos = np.arange(len(sorted_ops))
        errors = [avg_errors[op] for op in sorted_ops]
        
        # Create horizontal bar plot with gradual colors
        bars = ax.barh(y_pos, errors, alpha=0.8)
        
        # Apply dark red gradual color scheme
        if max(errors) > 0:
            # Normalize errors to [0, 1] for color mapping
            error_array = np.array(errors)
            normalized_errors = (error_array - min(error_array)) / (max(error_array) - min(error_array))
            
            # Use dark red gradual colors from dark red to even lighter red
            # Shift the minimum further toward white (lighter red)
            colors = plt.cm.Reds(0.2 + 0.75 * normalized_errors)  # Lighter overall gradient
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_ops, fontsize=16)
        ax.set_xlabel('Average Log-Likelihood Error', fontsize=16)
        # ax.set_title('Operations Ranked by Average Error Impact', fontsize=18, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, error) in enumerate(zip(bars, errors)):
            ax.text(error * 1.1, bar.get_y() + bar.get_height()/2, 
                    f'{error:.2e}', va='center', fontsize=16)
    
    plt.tight_layout()
    return fig

def create_ranking_comparison_plot(all_results: List[Dict[str, Any]]):
    """Create a plot showing how operation rankings change across all parameter combinations with gradual colors."""
    
    operations = [
        # 'kernel_train_gen', 'kernel_test_gen', 'kernel_cross_gen',
        'chol_train', 'solve_train_y', 'gemv_train', 
        'solve_train_cross',
        'gemm_train', 
        # 'cov_subtraction',
        'chol_cond', 'solve_cond', 'log_diag', 'inner_product'
    ]
    
    # Extract all parameter combinations
    param_combinations = []
    for result in all_results:
        if result['baseline_ll'] is not None:
            combo = {
                'n': result['n'],
                'nu': result['nu'],
                'separability': result.get('seperablity', 0.0),
                'time_scale': result.get('time_scale', 1.0),
                'result': result
            }
            param_combinations.append(combo)
        else:
            combo = {
                'n': result['n'],
                'nu': result['nu'],
                'separability': result.get('seperablity', 0.0),
                'time_scale': result.get('time_scale', 1.0),
                'result': 9999.
            }
            param_combinations.append(combo)
    
    # Sort by n, then separability, then time_scale
    param_combinations.sort(key=lambda x: (x['n'], x['separability'], x['time_scale']))
    
    # Create ranking matrix
    n_combinations = len(param_combinations)
    ranking_matrix = np.full((len(operations), n_combinations), np.nan)
    combination_labels = []
    
    for combo_idx, combo in enumerate(param_combinations):
        result = combo['result']
        
        # Extract errors and rank operations
        op_errors = {}
        for op in operations:
            if op in result.get('operation_results', {}):
                error = result['operation_results'][op]['mean_ll_error']
                if not np.isnan(error):
                    op_errors[op] = error
        
        if op_errors:
            # Sort by error (highest first) and assign ranks
            sorted_ops = sorted(op_errors.keys(), key=lambda x: op_errors[x], reverse=True)
            
            for rank, op in enumerate(sorted_ops, 1):
                op_idx = operations.index(op)
                ranking_matrix[op_idx, combo_idx] = rank
        
        # Create detailed labels with all parameters
        label = f'n={combo["n"]}\ns={combo["separability"]:.1f}\nτ={combo["time_scale"]:.1f}'
        combination_labels.append(label)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Create heatmap with gradual color scheme
    # Use a custom colormap that goes from yellow (rank 1) to purple (highest rank)
    im = ax.imshow(ranking_matrix, cmap='seismic_r', aspect='auto', vmin=1, vmax=len(operations))
    
    # Set ticks and labels
    ax.set_xticks(range(len(combination_labels)))
    ax.set_xticklabels(combination_labels, rotation=0, ha='center', fontsize=11)
    ax.set_yticks(range(len(operations)))
    ax.set_yticklabels(operations, fontsize=12)
    
    # Add text annotations with better contrast
    for i in range(len(operations)):
        for j in range(len(combination_labels)):
            if not np.isnan(ranking_matrix[i, j]):
                rank = int(ranking_matrix[i, j])
                # Choose text color based on rank for better contrast
                if rank <= 2:
                    color = 'black'
                    weight = 'bold'
                elif rank <= 4:
                    color = 'black'
                    weight = 'bold'
                else:
                    color = 'black'
                    weight = 'bold'
                
                ax.text(j, i, f'{rank}', ha="center", va="center", 
                       color=color, fontweight=weight, fontsize=18)
    
    # ax.set_title('Operation Error Impact Rankings Across All Parameter Combinations\n(1 = Highest Error Impact)', 
                # fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Parameter Combinations (n, separability, time_scale)', fontsize=16)
    ax.set_ylabel('Operations', fontsize=16)
    
    # Add colorbar with custom ticks
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Rank (1 = Highest Error Impact)', fontsize=16)
    cbar.set_ticks(range(1, len(operations) + 1))
    
    plt.tight_layout()
    return fig


def load_results_from_json(results_file: str) -> List[Dict[str, Any]]:
    """Load results from a JSON file."""
    print(f"Loading results from: {results_file}")
    with open(results_file, 'r') as f:
        results = load_json(results_file)
    print(f"Loaded {len(results)} experiment results")
    return results


def main(
    results_root: Optional[str] = None,
    timestamp: bool = True,
    n_trials_override: Optional[int] = None,
    load_results: Optional[str] = None,
) -> None:
    """Main function to run the error impact analysis."""
    print("Operation Error Impact Analysis for Mixed Precision Gaussian Processes")
    print("=" * 70)
    
    results_paths = prepare_results_dir(
        "operation_llh_error_impact",
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
                # Main ranking plot
                fig1 = create_main_operation_ranking_plot(quality_results)
                filename1 = results_paths['figures'] / f'operation_main_ranking_{quality}.pdf'
                plt.savefig(filename1, dpi=300, bbox_inches='tight')
                plt.close(fig1)
                print(f"Saved main ranking plot to: {filename1}")
                
                # Ranking comparison plot (if multiple parameter combinations)
                if len(quality_results) >= 2:
                    fig3 = create_ranking_comparison_plot(quality_results)
                    filename3 = results_paths['figures'] / f'operation_ranking_comparison_{quality}.pdf'
                    plt.savefig(filename3, dpi=300, bbox_inches='tight')
                    plt.close(fig3)
                    print(f"Saved ranking comparison to: {filename3}")
        
        print(f"\nVisualization complete! Check: {results_paths['figures']}")
        return

    # Original experiment code continues here...

    # Experimental parameters
    n_quality = ['best', 'good', 'worst'] # , 'good', 'worst'
    n_values = [100]  # Reduced from [10, 100, 1000]
    n_seperablity = [0.0,  0.5, 1.0] # seperablity[0.0, 0.5, 1.0]
    n_time_scale = [1,  5, 10] # time scale
    n_dim_scale = [[0.23, 0.23]] # dimension scale
    n_dim_length = [2] # dimension length
    nu_values = [1.5]  # Reduced from [0.5, 1.5, 2.5]
    n_time_lag = [2] # time lag
    n_nu_time = [0.5] # time smoothness
    
    n_trials_default = 10
    n_trials = n_trials_override or n_trials_default

    all_results = []

    # Run experiments
    total_experiments = (
        len(n_values)
        * len(nu_values)
        * len(n_quality)
        * len(n_seperablity)
        * len(n_time_scale)
        * len(n_dim_scale)
        * len(n_dim_length)
        * len(n_time_lag)
        * len(n_nu_time)
    )
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

                                        try:
                                            result = run_ablation_experiment(
                                                n,
                                                nu,
                                                n_trials,
                                                quality,
                                                seperablity=seperablity,
                                                time_scale=time_scale,
                                                dim_scale=dim_scale,
                                                dim_length=dim_length,
                                                time_lag=time_lag,
                                                nu_time=nu_time,
                                                results_paths=results_paths,
                                            )
                                            all_results.append(result)
                                        except Exception as e:
                                            print(f"Error in experiment n={n}, nu={nu}: {e}")
                                            continue
    
        print(f"\nSuccessfully completed {len(all_results)} experiments")
    
        # Create visualizations
        print("\nCreating error impact visualizations...")
    
        # Plot 1: Main operation ranking plot
        quality_results = [res for res in all_results if res.get('quality') == quality]
        fig1 = create_main_operation_ranking_plot(quality_results)
        
        # Save the main ranking plot
        filename1 = results_paths['figures'] / f'operation_main_ranking_{quality}.pdf'
        plt.figure(fig1.number)
        plt.savefig(filename1, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"Saved main ranking plot to: {filename1}")
        
        # # Plot 2: Parameter-specific subfigures
        # fig2 = create_parameter_subfigures(quality_results)
        
        # # Save the subfigures plot
        # filename2 = results_paths['figures'] / f'operation_parameter_subfigures_{quality}.pdf'
        # plt.figure(fig2.number)
        # plt.savefig(filename2, dpi=300, bbox_inches='tight')
        # plt.close(fig2)
        # print(f"Saved parameter subfigures to: {filename2}")
        
        # Plot 3: Ranking comparison plot
        if len(all_results) >= 2:  # Need at least 2 combinations for comparison
            fig3 = create_ranking_comparison_plot(all_results)
            
            # Save the ranking comparison plot
            filename3 = results_paths['figures'] / f'operation_ranking_comparison_{quality}.pdf'
            plt.figure(fig3.number)
            plt.savefig(filename3, dpi=300, bbox_inches='tight')
            plt.close(fig3)
            print(f"Saved ranking comparison to: {filename3}")

        quality_summary_path = results_paths['data'] / f'operation_llh_error_impact_{quality}.json'
        save_json(quality_results, quality_summary_path)
        
        # Print summary analysis
        print("\n" + "=" * 70)
        print("OPERATION ERROR IMPACT SUMMARY")
        print("=" * 70)
        
        operations = [
            # 'kernel_train_gen', 'kernel_test_gen', 'kernel_cross_gen',
            'chol_train', 'solve_train_y', 'gemv_train', 
            'solve_train_cross', 'gemm_train', 
            # 'cov_subtraction',
            'chol_cond', 'solve_cond', 'log_diag', 'inner_product'
        ]
        
        # Overall ranking analysis
        overall_errors = {}
        for op in operations:
            errors = []
            for result in all_results:
                if op in result.get('operation_results', {}):
                    error = result['operation_results'][op]['mean_ll_error']
                    if not np.isnan(error):
                        errors.append(error)
            overall_errors[op] = np.mean(errors) if errors else 0
        
        sorted_ops_overall = sorted(operations, key=lambda x: overall_errors[x], reverse=True)
        
        print("\nOVERALL OPERATION RANKING (by average error impact):")
        print("-" * 50)
        for i, op in enumerate(sorted_ops_overall, 1):
            avg_error = overall_errors[op]
            print(f"{i:2d}. {op:20s}: {avg_error:.2e}")
        
        # Parameter-specific analysis
        print(f"\nPARAMETER-SPECIFIC INSIGHTS:")
        print("-" * 30)
        
        for nu in sorted(set(r['nu'] for r in all_results)):
            print(f"\nKernel smoothness ν = {nu}:")
            nu_results = [r for r in all_results if r['nu'] == nu]
            
            # Find most critical operation for this nu
            nu_errors = {}
            for op in operations:
                errors = []
                for result in nu_results:
                    if op in result.get('operation_results', {}):
                        error = result['operation_results'][op]['mean_ll_error']
                        if not np.isnan(error):
                            errors.append(error)
                nu_errors[op] = np.mean(errors) if errors else 0
            
            if nu_errors:
                most_critical = max(nu_errors.keys(), key=lambda x: nu_errors[x])
                print(f"  Most critical operation: {most_critical}")
                print(f"  Average error: {nu_errors[most_critical]:.2e}")
        
        for n in sorted(set(r['n'] for r in all_results)):
            print(f"\nScale parameter n = {n}:")
            n_results = [r for r in all_results if r['n'] == n]
            
            # Find most critical operation for this n
            n_errors = {}
            for op in operations:
                errors = []
                for result in n_results:
                    if op in result.get('operation_results', {}):
                        error = result['operation_results'][op]['mean_ll_error']
                        if not np.isnan(error):
                            errors.append(error)
                n_errors[op] = np.mean(errors) if errors else 0
            
            if n_errors:
                most_critical = max(n_errors.keys(), key=lambda x: n_errors[x])
                print(f"  Most critical operation: {most_critical}")
                print(f"  Average error: {n_errors[most_critical]:.2e}")
        
        # Don't show plots in headless environment
        # plt.show()
        print("\nAll visualizations completed successfully!")

    overall_path = results_paths['data'] / 'operation_llh_error_impact_all_results.json'
    save_json(all_results, overall_path)
    print(f"All results saved to: {results_paths['root']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the log-likelihood error impact analysis.")
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
        help="Override the number of trials per experiment configuration.",
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
    
    # Run main function with load_results option
    main(
        results_root=args.results_root,
        timestamp=not args.no_timestamp,
        n_trials_override=args.n_trials,
        load_results=args.load_results,
    )
