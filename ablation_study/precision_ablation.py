#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import gamma, kv
from scipy.linalg import cholesky, solve_triangular
import time
from typing import Tuple, Dict, Any, List
import warnings
import matplotlib
matplotlib.rcParams.update({'font.size': 18})

from utils import MaternKernel, AblationGaussianProcess, generate_circle_points

def run_ablation_experiment(n: int, nu: float, n_trials: int = 5) -> Dict[str, Any]:
    """Run ablation study for specific n and nu values."""
    print(f"Running ablation study: n={n}, ν={nu}")
    
    # Create kernel and GP
    kernel = MaternKernel(nu=nu, length_scale=0.3/nu, variance=1.0)
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
        _all_points, _inner_points, _outer_points = generate_circle_points(n)
    
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
        'kernel_train_gen', 
        'kernel_test_gen',
        'kernel_cross_gen',
        'chol_train', 'solve_train_y', 'solve_train_cross', 'gemv_train',
        'gemm_train', 'cov_subtraction', 'chol_cond', 
        'solve_cond', 'log_diag', 'inner_product'
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
        # if ll  -1e9:  # Valid result
        baseline_ll_values.append(ll)
        baseline_diagnostics.append(diag)
    # print(baseline_ll_values)
    if baseline_ll_values:
        results['baseline_ll'] = np.mean(baseline_ll_values)
        results['baseline_std'] = np.std(baseline_ll_values)
        results['baseline_cond_train'] = np.mean([d['cond_K_train'] for d in baseline_diagnostics])
        results['baseline_cond_cov'] = np.mean([d['cond_cov_matrix'] for d in baseline_diagnostics])
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
            'failures': failures
        }
    
    return results

def visualize_ablation_results(all_results: List[Dict[str, Any]]):
    """Create comprehensive visualizations of ablation study results."""
    
    # Extract data organization
    n_values = sorted(set(r['n'] for r in all_results))
    nu_values = sorted(set(r['nu'] for r in all_results))
    operations = [
        'kernel_train_gen', 'kernel_test_gen', 'kernel_cross_gen',
        'chol_train', 'solve_train_y', 'solve_train_cross', 'gemv_train',
        'gemm_train', 'cov_subtraction', 'chol_cond', 
        'solve_cond', 'log_diag', 'inner_product'
    ]
    
    # Create figure with subplots and set default font
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    ax = axes
    success_data = np.zeros((len(operations), len(nu_values)))
    
    for i, op in enumerate(operations):
        for j, nu in enumerate(nu_values):
            # Average success rate across all n values for this nu
            success_rates = []
            for result in all_results:
                if result['nu'] == nu and op in result.get('operation_results', {}):
                    success_rates.append(result['operation_results'][op]['success_rate'])
            
            success_data[i, j] = np.mean(success_rates) if success_rates else 0
    
    im1 = ax.imshow(success_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(range(len(nu_values)))
    ax.set_xticklabels([f'ν={nu}' for nu in nu_values])
    ax.set_yticks(range(len(operations)))
    ax.set_yticklabels(operations, rotation=0)
    # ax.set_title('Success Rate (%) by Operation and Kernel Smoothness')
    plt.colorbar(im1, ax=ax)
    
    # Add text annotations
    for i in range(len(operations)):
        for j in range(len(nu_values)):
            text = ax.text(j, i, f'{success_data[i, j]:.0f}%',
                          ha="center", va="center", color="black" if success_data[i, j] > 50 else "white")
    
    plt.tight_layout()
    return fig

def main():
    """Main function to run ablation study."""
    print("Precision Ablation Study for Conditional Log-Likelihood")
    print("=" * 60)
    
    # Experimental parameters (reduced for faster execution)
    # n_values = [10, 100, 1000]  # Reduced from [10, 100, 1000]
    n_values = [10, 100, 1000]  # Reduced from [10, 100, 1000]
    nu_values = [0.5, 1.5, 2.5]  # Reduced from [0.5, 1.5, 2.5]
    n_trials = 10  # Reduced from 10
    
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
                all_results.append(result)
            except Exception as e:
                print(f"Error in experiment n={n}, nu={nu}: {e}")
                continue
    
    # Create visualizations
    print("\nCreating visualizations...")
    fig = visualize_ablation_results(all_results)
    plt.savefig('./fig/ablation_results.pdf', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed summary
    print("\n" + "=" * 60)
    print("ABLATION STUDY SUMMARY")
    print("=" * 60)

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    results = main()
