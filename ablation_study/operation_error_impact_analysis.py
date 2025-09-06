#!/usr/bin/env python3
"""
Operation Error Impact Analysis for Mixed Precision Gaussian Processes

This script creates detailed visualizations of how different operations' error impacts
vary across different values of n (scale parameter) and nu (Matern kernel smoothness).

The analysis focuses on ranking operations by their average error impact and showing
how these rankings change across parameter combinations.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import scipy.spatial.distance as spdist
from scipy.special import gamma, kv
from scipy.linalg import cholesky, solve_triangular
import time
from typing import Tuple, Dict, Any, List
import warnings
from matplotlib.colors import LogNorm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class MaternKernel:
    """Matern kernel implementation with different smoothness parameters."""
    
    def __init__(self, nu: float, length_scale: float = 0.05, variance: float = 1.0):
        self.nu = nu
        self.length_scale = length_scale
        self.variance = variance
        
    def __call__(self, X1: np.ndarray, X2: np.ndarray = None, precision: str = 'double') -> np.ndarray:
        """Compute Matern kernel matrix with specified precision."""
        if X2 is None:
            X2 = X1
            
        # Set precision for computation
        dtype = np.float64 if precision == 'double' else np.float32
        X1 = X1.astype(dtype)
        X2 = X2.astype(dtype)
        
        # Compute pairwise distances
        dists = spdist.cdist(X1, X2, metric='euclidean').astype(dtype)
        
        # Avoid division by zero
        # dists = np.maximum(dists, np.finfo(dtype).eps)
        
        # Scaled distances
        scaled_dists = dists / self.length_scale
        
        if self.nu == 0.5:
            # Exponential kernel (Matern with nu=0.5)
            _item_poly = 1.0
        elif self.nu == 1.5:
            _item_poly = 1.0 + scaled_dists
        elif self.nu == 2.5:
            _item_poly = 1.0 + scaled_dists + scaled_dists**2 / 3
        elif self.nu == 3.5:
            _item_poly = 1.0 + scaled_dists + 2 * scaled_dists**2 / 5 + scaled_dists**3 / 15
        else:
            raise ValueError(f"Matern kernel with nu={self.nu} is not supported")
            # # General Matern kernel
            # scaled_dists = np.maximum(scaled_dists, np.finfo(dtype).eps)
            # temp = (2**(1-self.nu)) / gamma(self.nu)
            # K = self.variance * temp * (scaled_dists**self.nu) * kv(self.nu, scaled_dists)
        K = self.variance * _item_poly * np.exp(-scaled_dists)    
        # Set diagonal to variance (avoid numerical issues)
        if X1.shape == X2.shape and np.allclose(X1.astype(np.float64), X2.astype(np.float64)):
            np.fill_diagonal(K, self.variance)
            
        return K.astype(dtype)

class AblationGaussianProcess:
    """Gaussian Process with ablation study capabilities for precision effects."""
    
    def __init__(self, kernel: MaternKernel, noise_variance: float = 1e-6):
        self.kernel = kernel
        self.noise_variance = noise_variance
        
    def conditional_log_likelihood_ablation(self, X_train: np.ndarray, y_train: np.ndarray,
                                          X_test: np.ndarray, y_test: np.ndarray,
                                          operation_precision: Dict[str, str]) -> Tuple[float, Dict[str, Any]]:
        """
        Compute conditional log-likelihood with selective precision for atomic operations.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            operation_precision: Dictionary specifying precision for each atomic operation
                Keys: 'kernel_train_gen', 'kernel_test_gen', 'kernel_cross_gen',
                      'chol_train', 'solve_train_y', 'gemv_train', 
                      'solve_train_cross',
                      'gemm_train', 'cov_subtraction',
                      'chol_cond', 'solve_cond', 'log_diag', 'inner_product'
        
        Returns:
            log_likelihood: Computed log-likelihood
            diagnostics: Dictionary with intermediate values and condition numbers
        """
        
        # Default to double precision for all atomic operations
        default_ops = {
            'kernel_train_gen': 'double',
            'kernel_test_gen': 'double', 
            'kernel_cross_gen': 'double',
            'chol_train': 'double',
            'solve_train_y': 'double',
            'gemv_train': 'double',
            'solve_train_cross': 'double', 
            'gemm_train': 'double',
            'cov_subtraction': 'double',
            'chol_cond': 'double',
            'solve_cond': 'double',
            'log_diag': 'double',
            'inner_product': 'double'
        }
        default_ops.update(operation_precision)
        ops = default_ops
        
        diagnostics = {}
        
        try:
            # Always start with double precision inputs
            X_train_dp = X_train.astype(np.float64)
            y_train_dp = y_train.astype(np.float64)
            X_test_dp = X_test.astype(np.float64)
            y_test_dp = y_test.astype(np.float64)
            
            # =====================================
            # ATOMIC OPERATION 1: kernel_train_gen
            # =====================================
            # K_train_dp = self.kernel(X_train_dp, precision='double')
            if ops['kernel_train_gen'] == 'single':
                # K_train = K_train_dp.astype(np.float32)
                K_train_dp = self.kernel(X_train_dp, precision='single')
                noise_val = np.float32(self.noise_variance)
            else:
                # K_train = K_train_dp.astype(np.float64)
                K_train_dp = self.kernel(X_train_dp, precision='double')
                noise_val = np.float64(self.noise_variance)
            K_train = K_train_dp.astype(np.float64)
            K_train += noise_val * np.eye(len(X_train), dtype=K_train.dtype)
            diagnostics['cond_K_train'] = float(np.linalg.cond(K_train.astype(np.float64)))
            
            # =====================================
            # ATOMIC OPERATION 2: chol_train
            # =====================================
            if ops['chol_train'] == 'single':
                K_train_chol = K_train.astype(np.float32)
            else:
                K_train_chol = K_train.astype(np.float64)
                
            L_train = cholesky(K_train_chol, lower=True)
            L_train = L_train.astype(np.float64)
            
            # =====================================
            # ATOMIC OPERATION 3: kernel_test_gen
            # =====================================
            # K_test_dp = self.kernel(X_test_dp, precision='double')
            if ops['kernel_test_gen'] == 'single':
                # K_test = K_test_dp.astype(np.float32)
                K_test_dp = self.kernel(X_test_dp, precision='single')
            else:
                # K_test = K_test_dp.astype(np.float64)
                K_test_dp = self.kernel(X_test_dp, precision='double')
            K_test = K_test_dp.astype(np.float64)
            # =====================================
            # ATOMIC OPERATION 4: kernel_cross_gen
            # =====================================
            # K_cross_dp = self.kernel(X_train_dp, X_test_dp, precision='double')
            if ops['kernel_cross_gen'] == 'single':
                # K_cross = K_cross_dp.astype(np.float32)
                K_cross_dp = self.kernel(X_train_dp, X_test_dp, precision='single')
            else:
                # K_cross = K_cross_dp.astype(np.float64)
                K_cross_dp = self.kernel(X_train_dp, X_test_dp, precision='double')
            K_cross = K_cross_dp.astype(np.float64)
            # =====================================
            # ATOMIC OPERATION 5: solve_train_y (L @ alpha = y)
            # =====================================
            if ops['solve_train_y'] == 'single':
                L_train_solve1 = L_train.astype(np.float32)
                y_train_solve = y_train_dp.astype(np.float32)
            else:
                L_train_solve1 = L_train.astype(np.float64)
                y_train_solve = y_train_dp.astype(np.float64)
                
            alpha = solve_triangular(L_train_solve1, y_train_solve, lower=True)
            alpha = alpha.astype(np.float64)
            
            # =====================================
            # ATOMIC OPERATION 6: solve_train_cross (L @ beta = K_cross)
            # =====================================
            if ops['solve_train_cross'] == 'single':
                L_train_solve2 = L_train.astype(np.float32)
                K_cross_solve = K_cross.astype(np.float32)
            else:
                L_train_solve2 = L_train.astype(np.float64)
                K_cross_solve = K_cross.astype(np.float64)
                
            beta = solve_triangular(L_train_solve2, K_cross_solve, lower=True)
            beta = beta.astype(np.float64)
            
            # =====================================
            # ATOMIC OPERATION 7: gemv_train (mean = beta.T @ alpha)
            # =====================================
            if ops['gemv_train'] == 'single':
                beta_mean = beta.astype(np.float32)
                alpha_mean = alpha.astype(np.float32)
            else:
                beta_mean = beta.astype(np.float64)
                alpha_mean = alpha.astype(np.float64)
                
            mean = beta_mean.T @ alpha_mean
            mean = mean.astype(np.float64)
            # =====================================
            # ATOMIC OPERATION 8: gemm_train (beta.T @ beta)
            # =====================================
            if ops['gemm_train'] == 'single':
                beta_cov = beta.astype(np.float32)
            else:
                beta_cov = beta.astype(np.float64)
                
            beta_product = beta_cov.T @ beta_cov
            beta_product = beta_product.astype(np.float64)
            # =====================================
            # ATOMIC OPERATION 9: cov_subtraction (K_test - beta.T @ beta)
            # =====================================
            if ops['cov_subtraction'] == 'single':
                K_test_sub = K_test.astype(np.float32)
                beta_product_sub = beta_product.astype(np.float32)
                reg_val = np.float32(self.noise_variance)
            else:
                K_test_sub = K_test.astype(np.float64)
                beta_product_sub = beta_product.astype(np.float64)
                reg_val = np.float64(self.noise_variance)
                
            cov = K_test_sub - beta_product_sub
            cov += reg_val * np.eye(len(X_test), dtype=cov.dtype)
            diagnostics['cond_cov_matrix'] = float(np.linalg.cond(cov.astype(np.float64)))

            # =====================================
            # ATOMIC OPERATION 10: chol_cond
            # =====================================
            if ops['chol_cond'] == 'single':
                cov_chol = cov.astype(np.float32)
            else:
                cov_chol = cov.astype(np.float64)
                
            L_cond = cholesky(cov_chol, lower=True)
            
            # =====================================
            # ATOMIC OPERATION 11: solve_cond (L_cond @ v = residuals)
            # =====================================
            # Prepare residuals first (always in working precision)
            residuals_dp = y_test_dp - mean.astype(np.float64)
            
            if ops['solve_cond'] == 'single':
                L_cond_solve = L_cond.astype(np.float32)
                residuals_solve = residuals_dp.astype(np.float32)
            else:
                L_cond_solve = L_cond.astype(np.float64)
                residuals_solve = residuals_dp.astype(np.float64)
                
            v = solve_triangular(L_cond_solve, residuals_solve, lower=True)
            
            # =====================================
            # ATOMIC OPERATION 12: inner_product (v.T @ v)
            # =====================================
            if ops['inner_product'] == 'single':
                v_qf = v.astype(np.float32)
            else:
                v_qf = v.astype(np.float64)
                
            quad_form = np.sum(v_qf**2)
            
            # =====================================
            # ATOMIC OPERATION 13: log_diag (log of diagonal elements)
            # =====================================
            if ops['log_diag'] == 'single':
                L_cond_logdet = L_cond.astype(np.float32)
            else:
                L_cond_logdet = L_cond.astype(np.float64)
                
            log_det = 2 * np.sum(np.log(np.diag(L_cond_logdet)))
            
            # Final log-likelihood computation (always in double precision)
            # n_test = len(X_test)
            log_likelihood = -0.5 * (float(quad_form) + float(log_det)) #+ n_test * np.log(2 * np.pi)
            
            # Store operation precisions used
            diagnostics['operation_precisions'] = ops.copy()
            diagnostics['quad_form'] = float(quad_form)
            diagnostics['log_det'] = float(log_det)
            
            return log_likelihood, diagnostics
            
        except (np.linalg.LinAlgError, ValueError, OverflowError) as e:
            # Return failure indicator
            diagnostics['error'] = str(e)
            diagnostics['operation_precisions'] = ops.copy()
            return np.nan, diagnostics

def generate_circle_points(n: int, n_points: int = 600) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate points in circle with inner/outer regions."""
    outer_radius = 1.0 / n
    inner_radius = 1.0 / (n * np.sqrt(3))
    
    points = []
    while len(points) < n_points:
        # Generate random points in square
        x = np.random.uniform(-outer_radius, outer_radius)
        y = np.random.uniform(-outer_radius, outer_radius)
        r = np.sqrt(x**2 + y**2)
        
        # Keep points within outer circle
        if r <= outer_radius:
            points.append([x, y])
    
    points = np.array(points[:n_points])
    
    # Classify points
    distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    inner_mask = distances <= inner_radius
    outer_mask = ~inner_mask
    
    inner_points = points[inner_mask]
    outer_points = points[outer_mask]
    points = np.concatenate([inner_points, outer_points], axis=0)
    
    return points, inner_points, outer_points

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
    np.random.seed(42 + n + int(nu*10))  
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
        results['baseline_ll'] = np.mean(baseline_ll_values)
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
            print(f"Trial {trial}, ll single: {ll}")
            if ll is np.nan:  # Failed
                failures += 1
            else:
                ll_values.append(ll)
                ll_errors.append(abs(ll - results['baseline_ll']))
        
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
        ax1.set_yticklabels(sorted_ops)
        ax1.set_xlabel('Average Log-Likelihood Error')
        ax1.set_title('Operations Ranked by Average Error Impact (All Parameter Combinations)', fontsize=14, fontweight='bold')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, error) in enumerate(zip(bars, errors)):
            ax1.text(error * 1.1, bar.get_y() + bar.get_height()/2, 
                    f'{error:.2e}', va='center', fontsize=8)
    
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
                    ax.set_yticklabels([op.replace('_', '\n') for op in local_sorted_ops], fontsize=7)
                    ax.set_xlabel('Log-Likelihood Error', fontsize=8)
                    ax.set_title(f'n={n}, ν={nu}', fontsize=10, fontweight='bold')
                    ax.set_xscale('log')
                    ax.grid(True, alpha=0.3, axis='x')
                    
                    # Add value labels on bars (only for top 3)
                    for k, (bar, error) in enumerate(zip(bars[:3], errors[:3])):
                        ax.text(error * 1.1, bar.get_y() + bar.get_height()/2, 
                               f'{error:.1e}', va='center', fontsize=6)
            
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
    
    ax.set_title('Operation Error Impact Rankings Across Parameter Combinations\n(1 = Highest Error Impact)', 
                fontsize=14, fontweight='bold')
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
    n_trials = 1
    
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
    fig1.suptitle('Operations Ranked by Average Error Impact - Detailed Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
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
        'chol_train', 'solve_train_forward', 'solve_cross_forward', 'train_gemm',
        'cond_cov_gemm', 'cond_cov_subtraction', 'chol_cond', 
        'solve_cond_forward', 'log_diag', 'quad_form_solve'
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
