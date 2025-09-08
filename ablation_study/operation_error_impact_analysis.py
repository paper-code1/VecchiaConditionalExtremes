#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import time
from typing import Tuple, Dict, Any, List
import warnings
from matplotlib.colors import LogNorm
from utils import MaternKernel, AblationGaussianProcess, generate_circle_points

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def run_ablation_experiment(n: int, nu: float, n_trials: int = 5) -> Dict[str, Any]:
    """Run ablation study for specific n and nu values."""
    print(f"Running experiment: n={n}, ν={nu}")
    
    # Create kernel and GP
    kernel = MaternKernel(nu=nu, length_scale=0.3/nu, variance=1.0)
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
        _all_points, _inner_points, _outer_points = generate_circle_points(n)
        
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
        'kernel_train_gen', 'kernel_test_gen', 'kernel_cross_gen',
        'chol_train', 'solve_train_y', 'gemv_train', 
        'solve_train_cross',
        'gemm_train', 'cov_subtraction',
        'chol_cond', 'solve_cond', 'log_diag', 'inner_product'
    ]
    
    results = {
        'n': n,
        'nu': nu,
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
        print(f"Trial {trial}, ll double: {ll}")
        if ll is not np.nan:  # Valid result
            baseline_ll_values.append(ll)
            baseline_diagnostics.append(diag)
    
    if baseline_ll_values:
        results['baseline_ll'] = np.asarray(baseline_ll_values)
        results['baseline_std'] = np.std(baseline_ll_values)
        results['baseline_cond_train'] = np.mean([d['cond_K_train'] for d in baseline_diagnostics])
        results['baseline_cond_cov'] = np.mean([d['cond_cov_matrix'] for d in baseline_diagnostics])
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
                ll_errors.append(abs(ll - results['baseline_ll'][trial]))
        
        # Store results
        results['operation_results'][op] = {
            'success_rate': (n_trials - failures) / n_trials * 100,
            'mean_ll': np.mean(ll_values) if ll_values else np.nan,
            'std_ll': np.std(ll_values) if ll_values else np.nan,
            'mean_ll_error': np.mean(ll_errors) if ll_errors else np.nan,
            'max_ll_error': np.max(ll_errors) if ll_errors else np.nan,
            'failures': failures
        }
    
    return results

def create_operation_error_impact_plots(all_results: List[Dict[str, Any]]):
    """Create detailed plots of operation error impact across different n and nu values."""
    
    # Extract data organization
    n_values = sorted(set(r['n'] for r in all_results if r['baseline_ll'] is not None))
    nu_values = sorted(set(r['nu'] for r in all_results if r['baseline_ll'] is not None))
    operations = [
        'kernel_train_gen', 'kernel_test_gen', 'kernel_cross_gen',
        'chol_train', 'solve_train_y', 'gemv_train', 
        'solve_train_cross',
        'gemm_train', 'cov_subtraction',
        'chol_cond', 'solve_cond', 'log_diag', 'inner_product'
    ]
    
    # Set up the figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Create a grid layout
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1, 1])
    
    # Plot 1: Overall error impact ranking (single large plot)
    ax1 = fig.add_subplot(gs[0, :])
    
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
        
        # Create horizontal bar plot
        bars = ax1.barh(y_pos, errors, alpha=0.7)
        
        # Color bars by error magnitude (log scale)
        if max(errors) > 0:
            norm = LogNorm(vmin=min([e for e in errors if e > 0]), vmax=max(errors))
            colors = plt.cm.Reds(norm(errors))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(sorted_ops, fontsize=14)
        ax1.set_xlabel('Average Log-Likelihood Error')
        # ax1.set_title('Operations Ranked by Average Error Impact (All Parameter Combinations)', fontsize=14, fontweight='bold')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, error) in enumerate(zip(bars, errors)):
            ax1.text(error * 1.1, bar.get_y() + bar.get_height()/2, 
                    f'{error:.2e}', va='center', fontsize=16)
    
    # Plot 2-10: Individual plots for each parameter combination
    plot_idx = 1
    for i, nu in enumerate(nu_values):
        for j, n in enumerate(n_values):
            if plot_idx > 9:  # Only show first 9 combinations
                break
                
            row = (plot_idx - 1) // 3 + 1  # Fixed: subtract 1 before division
            col = (plot_idx - 1) % 3       # Fixed: subtract 1 before modulo
            ax = fig.add_subplot(gs[row, col])
            
            # Get results for this specific n, nu combination
            param_results = [r for r in all_results 
                           if r['n'] == n and r['nu'] == nu and r['baseline_ll'] is not None]
            
            if param_results:
                result = param_results[0]  # Should be only one
                
                # Extract error data for this combination
                op_errors = {}
                for op in operations:
                    if op in result.get('operation_results', {}):
                        error = result['operation_results'][op]['mean_ll_error']
                        if not np.isnan(error):
                            op_errors[op] = error
                
                if op_errors:
                    # Sort operations by error for this combination
                    local_sorted_ops = sorted(op_errors.keys(), key=lambda x: op_errors[x], reverse=True)
                    
                    y_pos = np.arange(len(local_sorted_ops))
                    errors = [op_errors[op] for op in local_sorted_ops]
                    
                    # Create horizontal bar plot
                    bars = ax.barh(y_pos, errors, alpha=0.7)
                    
                    # Color by relative error magnitude
                    if max(errors) > 0:
                        max_error = max(errors)
                        for bar, error in zip(bars, errors):
                            intensity = error / max_error
                            bar.set_color(plt.cm.Reds(intensity))
                    
                    ax.set_yticks(y_pos)
                    # ax.set_yticklabels([op.replace('_', '\n') for op in local_sorted_ops], fontsize=7)
                    ax.set_yticklabels(local_sorted_ops, fontsize=14)
                    ax.set_xlabel('Log-Likelihood Error', fontsize=16)
                    ax.set_title(f'n={n}, ν={nu}', fontsize=16)
                    ax.set_xscale('log')
                    ax.set_xlim(1e-8, 15)
                    ax.grid(True, alpha=0.3, axis='x')
                    
                    # Add value labels on bars (only for top 3)
                    for k, (bar, error) in enumerate(zip(bars[:3], errors[:3])):
                        ax.text(error * 1.1, bar.get_y() + bar.get_height()/2, 
                               f'{error:.1e}', va='center', fontsize=16)
            
            plot_idx += 1
            
            if plot_idx > 9:
                break
        
        if plot_idx > 9:
            break
    
    plt.tight_layout()
    return fig

def create_ranking_comparison_plot(all_results: List[Dict[str, Any]]):
    """Create a plot showing how operation rankings change across parameter combinations."""
    
    n_values = sorted(set(r['n'] for r in all_results if r['baseline_ll'] is not None))
    nu_values = sorted(set(r['nu'] for r in all_results if r['baseline_ll'] is not None))
    operations = [
        'kernel_train_gen', 'kernel_test_gen', 'kernel_cross_gen',
        'chol_train', 'solve_train_y', 'gemv_train', 
        'solve_train_cross',
        'gemm_train', 'cov_subtraction',
        'chol_cond', 'solve_cond', 'log_diag', 'inner_product'
    ]
    
    # Create ranking matrix
    n_combinations = len(n_values) * len(nu_values)
    ranking_matrix = np.full((len(operations), n_combinations), np.nan)
    combination_labels = []
    
    combo_idx = 0
    for nu in nu_values:
        for n in n_values:
            param_results = [r for r in all_results 
                           if r['n'] == n and r['nu'] == nu and r['baseline_ll'] is not None]
            
            if param_results:
                result = param_results[0]
                
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
                
                combination_labels.append(f'n={n}\nν={nu}')
                combo_idx += 1
    
    # Truncate to actual combinations
    ranking_matrix = ranking_matrix[:, :combo_idx]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create heatmap
    im = ax.imshow(ranking_matrix, cmap='RdYlBu_r', aspect='auto', vmin=1, vmax=len(operations))
    
    # Set ticks and labels
    ax.set_xticks(range(len(combination_labels)))
    ax.set_xticklabels(combination_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(operations)))
    ax.set_yticklabels(operations)
    
    # Add text annotations
    for i in range(len(operations)):
        for j in range(len(combination_labels)):
            if not np.isnan(ranking_matrix[i, j]):
                rank = int(ranking_matrix[i, j])
                color = 'white' if rank <= 3 else 'black'
                ax.text(j, i, f'{rank}', ha="center", va="center", 
                       color=color, fontweight='bold', fontsize=8)
    
    # ax.set_title('Operation Error Impact Rankings Across Parameter Combinations\n(1 = Highest Error Impact)', 
    #             fontsize=14, fontweight='bold')
    ax.set_xlabel('Parameter Combinations')
    ax.set_ylabel('Operations')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Rank (1 = Highest Error Impact)')
    
    plt.tight_layout()
    return fig

def main():
    """Main function to run the error impact analysis."""
    print("Operation Error Impact Analysis for Mixed Precision Gaussian Processes")
    print("=" * 70)
    
    # Experimental parameters
    n_values = [10, 100, 1000]
    nu_values = [0.5, 1.5, 2.5]
    n_trials = 10
    
    all_results = []
    
    # Run experiments
    total_experiments = len(n_values) * len(nu_values)
    exp_count = 0
    
    for n in n_values:
        for nu in nu_values:
            exp_count += 1
            print(f"\nExperiment {exp_count}/{total_experiments}")
            
            try:
                result = run_ablation_experiment(n, nu, n_trials)
                # if result['baseline_ll'] is not None:  # Only keep successful results
                #     all_results.append(result)
                all_results.append(result)
            except Exception as e:
                print(f"Error in experiment n={n}, nu={nu}: {e}")
                continue
    
    print(f"\nSuccessfully completed {len(all_results)} experiments")
    
    # Create visualizations
    print("\nCreating error impact visualizations...")
    
    # Plot 1: Detailed operation error impact plots
    fig1 = create_operation_error_impact_plots(all_results)
    # fig1.suptitle('Operations Ranked by Average Error Impact - Detailed Analysis', 
    #              fontsize=16, fontweight='bold', y=0.98)
    
    # Save the first plot
    filename1 = './fig/operation_error_impact_detailed.pdf'
    plt.figure(fig1.number)
    plt.savefig(filename1, dpi=300, bbox_inches='tight')
    print(f"Saved detailed analysis to: {filename1}")
    
    # Plot 2: Ranking comparison plot
    if len(all_results) >= 2:  # Need at least 2 combinations for comparison
        fig2 = create_ranking_comparison_plot(all_results)
        
        # Save the second plot
        filename2 = './fig/operation_ranking_comparison.pdf'
        plt.figure(fig2.number)
        plt.savefig(filename2, dpi=300, bbox_inches='tight')
        print(f"Saved ranking comparison to: {filename2}")
    
    # Print summary analysis
    print("\n" + "=" * 70)
    print("OPERATION ERROR IMPACT SUMMARY")
    print("=" * 70)
    
    operations = [
        'kernel_train_gen', 'kernel_test_gen', 'kernel_cross_gen',
        'chol_train', 'solve_train_y', 'gemv_train', 
        'solve_train_cross',
        'gemm_train', 'cov_subtraction',
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
    
    return all_results

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    results = main()
