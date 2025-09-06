#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as spdist
from scipy.special import gamma, kv
from scipy.linalg import cholesky, solve_triangular
import time
from typing import Tuple, Dict, Any, List
import warnings
import matplotlib
matplotlib.rcParams.update({'font.size': 14})

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
            
        K = self.variance * _item_poly * np.exp(-scaled_dists)    
        # Set diagonal to variance (avoid numerical issues)
        if X1.shape == X2.shape and np.allclose(X1.astype(np.float64), X2.astype(np.float64)):
            np.fill_diagonal(K, self.variance)
            
        return K.astype(dtype)

class MixedPrecisionGaussianProcess:
    """Gaussian Process with mixed precision capabilities."""
    
    def __init__(self, kernel: MaternKernel, noise_variance: float = 1e-6):
        self.kernel = kernel
        self.noise_variance = noise_variance
        
    def conditional_log_likelihood_mixed(self, X_train: np.ndarray, y_train: np.ndarray,
                                       X_test: np.ndarray, y_test: np.ndarray,
                                       operation_precision: Dict[str, str]) -> Tuple[float, Dict[str, Any]]:
        """
        Compute conditional log-likelihood with mixed precision operations.
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
            if ops['kernel_train_gen'] == 'single':
                K_train_dp = self.kernel(X_train_dp, precision='single')
                noise_val = np.float32(self.noise_variance)
            else:
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
            if ops['kernel_test_gen'] == 'single':
                K_test_dp = self.kernel(X_test_dp, precision='single')
            else:
                K_test_dp = self.kernel(X_test_dp, precision='double')
            K_test = K_test_dp.astype(np.float64)
            
            # =====================================
            # ATOMIC OPERATION 4: kernel_cross_gen
            # =====================================
            if ops['kernel_cross_gen'] == 'single':
                K_cross_dp = self.kernel(X_train_dp, X_test_dp, precision='single')
            else:
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
            log_likelihood = -0.5 * (float(quad_form) + float(log_det))
            
            # Store operation precisions used
            diagnostics['operation_precisions'] = ops.copy()
            diagnostics['quad_form'] = float(quad_form)
            diagnostics['log_det'] = float(log_det)
            
            return log_likelihood, diagnostics
            
        except (np.linalg.LinAlgError, ValueError, OverflowError) as e:
            # Return failure indicator
            diagnostics['error'] = str(e)
            diagnostics['operation_precisions'] = ops.copy()
            return -1e10, diagnostics

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

def run_mixed_precision_experiment(n_values: List[int], nu_values: List[float], n_trials: int = 10) -> Dict[str, Any]:
    """Run mixed precision comparison experiment."""
    print("Mixed Precision vs Full Double Precision Comparison")
    print("=" * 60)
    
    # Define the specific mixed precision configuration
    mixed_precision_config = {
        # Double precision operations (critical for accuracy)
        'kernel_train_gen': 'double',
        'kernel_cross_gen': 'double', 
        'kernel_test_gen': 'double',
        'chol_train': 'double',
        'solve_train_cross': 'double',
        'gemm_train': 'double',
        'cov_subtraction': 'double',
        'chol_cond': 'double', 
        # Single precision operations (less critical)
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
            gp = MixedPrecisionGaussianProcess(kernel, noise_variance=1e-6)
            
            # Storage for trial results
            double_ll_values = []
            mixed_ll_values = []
            ll_differences = []
            relative_errors = []
            
            # Generate data for trials
            np.random.seed(42 + n + int(nu*10))  # Reproducible
            
            for trial in range(n_trials):
                print(f"  Trial {trial+1}/{n_trials}", end="")
                
                # Generate spatial points and reference function
                all_points, inner_points, outer_points = generate_circle_points(n)
                
                # # Generate reference function values (always double precision)
                # y_true_all = np.random.multivariate_normal(
                #     np.zeros(len(all_points)), 
                #     kernel(all_points, precision='double')
                # )
                y_true_all = np.zeros(len(all_points))
                y_true_inner = y_true_all[:len(inner_points)]
                y_true_outer = y_true_all[len(inner_points):len(inner_points)+len(outer_points)]
                
                try:
                    # Compute full double precision log-likelihood
                    ll_double, diag_double = gp.conditional_log_likelihood_mixed(
                        outer_points, y_true_outer, inner_points, y_true_inner, full_double_config
                    )
                    
                    # Compute mixed precision log-likelihood
                    ll_mixed, diag_mixed = gp.conditional_log_likelihood_mixed(
                        outer_points, y_true_outer, inner_points, y_true_inner, mixed_precision_config
                    )
                    
                    if ll_double > -1e9 and ll_mixed > -1e9:  # Both succeeded
                        double_ll_values.append(ll_double)
                        mixed_ll_values.append(ll_mixed)
                        
                        ll_diff = abs(ll_mixed - ll_double)
                        ll_differences.append(ll_diff)
                        
                        rel_error = ll_diff / abs(ll_double) if abs(ll_double) > 1e-12 else 0
                        relative_errors.append(rel_error)
                        
                        print(f" ✓ (diff: {ll_diff:.2e})")
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
                'success_rate': len(double_ll_values) / n_trials * 100
            }
            
            results['experiments'].append(experiment_result)
            
            print(f"  Results: Mean LL diff = {experiment_result['mean_ll_difference']:.2e}, "
                  f"Mean rel error = {experiment_result['mean_relative_error']:.2e}")
    
    return results

def visualize_mixed_precision_results(results: Dict[str, Any]):
    """Create comprehensive visualizations of mixed precision comparison results."""
    
    experiments = results['experiments']
    n_values = results['n_values']
    nu_values = results['nu_values']
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Mixed Precision vs Full Double Precision Comparison', fontsize=16)
    
    # Prepare data matrices
    n_n = len(n_values)
    n_nu = len(nu_values)
    
    ll_diff_matrix = np.full((n_n, n_nu), np.nan)
    rel_error_matrix = np.full((n_n, n_nu), np.nan)
    success_matrix = np.full((n_n, n_nu), np.nan)
    max_error_matrix = np.full((n_n, n_nu), np.nan)
    
    for exp in experiments:
        i = n_values.index(exp['n'])
        j = nu_values.index(exp['nu'])
        ll_diff_matrix[i, j] = exp['mean_ll_difference']
        rel_error_matrix[i, j] = exp['mean_relative_error']
        success_matrix[i, j] = exp['success_rate']
        max_error_matrix[i, j] = exp['max_ll_difference']
    
    # Plot 1: Mean Log-Likelihood Difference (Line Plot)
    axes[0, 0].set_title('Mean Log-Likelihood Difference\n(|Mixed - Double|)')
    colors = ['red', 'blue', 'green']
    for j, nu in enumerate(nu_values):
        y_values = ll_diff_matrix[:, j]
        valid_mask = ~np.isnan(y_values)
        if np.any(valid_mask):
            axes[0, 0].plot(np.array(n_values)[valid_mask], y_values[valid_mask], 
                          'o-', color=colors[j], label=f'ν={nu}', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('n (problem size)')
    axes[0, 0].set_ylabel('Mean LL Difference')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Mean Relative Error (Line Plot)
    axes[0, 1].set_title('Mean Relative Error\n(|Mixed - Double| / |Double|)')
    for j, nu in enumerate(nu_values):
        y_values = rel_error_matrix[:, j]
        valid_mask = ~np.isnan(y_values)
        if np.any(valid_mask):
            axes[0, 1].plot(np.array(n_values)[valid_mask], y_values[valid_mask], 
                          'o-', color=colors[j], label=f'ν={nu}', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('n (problem size)')
    axes[0, 1].set_ylabel('Mean Relative Error')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Success Rate
    im3 = axes[1, 0].imshow(success_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    axes[1, 0].set_title('Success Rate (%)')
    axes[1, 0].set_xticks(range(n_nu))
    axes[1, 0].set_xticklabels([f'ν={nu}' for nu in nu_values])
    axes[1, 0].set_yticks(range(n_n))
    axes[1, 0].set_yticklabels([f'n={n}' for n in n_values])
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Add text annotations
    for i in range(n_n):
        for j in range(n_nu):
            if not np.isnan(success_matrix[i, j]):
                text = axes[1, 0].text(j, i, f'{success_matrix[i, j]:.0f}%',
                                     ha="center", va="center", 
                                     color="black" if success_matrix[i, j] > 50 else "white",
                                     fontsize=10)
    
    # Plot 4: Maximum Error
    im4 = axes[1, 1].imshow(max_error_matrix, cmap='Reds', aspect='auto')
    axes[1, 1].set_title('Maximum Log-Likelihood Difference')
    axes[1, 1].set_xticks(range(n_nu))
    axes[1, 1].set_xticklabels([f'ν={nu}' for nu in nu_values])
    axes[1, 1].set_yticks(range(n_n))
    axes[1, 1].set_yticklabels([f'n={n}' for n in n_values])
    plt.colorbar(im4, ax=axes[1, 1])
    
    # Add text annotations
    for i in range(n_n):
        for j in range(n_nu):
            if not np.isnan(max_error_matrix[i, j]):
                text = axes[1, 1].text(j, i, f'{max_error_matrix[i, j]:.1e}',
                                     ha="center", va="center", color="white", fontsize=10)
    
    plt.tight_layout()
    return fig

def create_summary_table(results: Dict[str, Any]):
    """Create a summary table of the results."""
    experiments = results['experiments']
    
    print("\n" + "=" * 80)
    print("MIXED PRECISION COMPARISON SUMMARY")
    print("=" * 80)
    
    print("\nMixed Precision Configuration:")
    for op, precision in results['mixed_precision_config'].items():
        print(f"  {op}: {precision}")
    
    print(f"\n{'n':>4} {'ν':>4} {'Success%':>8} {'Mean LL Diff':>12} {'Max LL Diff':>12} {'Mean Rel Err':>12} {'Max Rel Err':>12}")
    print("-" * 80)
    
    for exp in experiments:
        print(f"{exp['n']:>4} {exp['nu']:>4.1f} {exp['success_rate']:>7.0f}% "
              f"{exp['mean_ll_difference']:>11.2e} {exp['max_ll_difference']:>11.2e} "
              f"{exp['mean_relative_error']:>11.2e} {exp['max_relative_error']:>11.2e}")
    
    # Overall statistics
    all_mean_diffs = [exp['mean_ll_difference'] for exp in experiments if not np.isnan(exp['mean_ll_difference'])]
    all_rel_errors = [exp['mean_relative_error'] for exp in experiments if not np.isnan(exp['mean_relative_error'])]
    all_success_rates = [exp['success_rate'] for exp in experiments]
    
    print("-" * 80)
    print(f"Overall Statistics:")
    print(f"  Average Success Rate: {np.mean(all_success_rates):.1f}%")
    print(f"  Average Mean LL Difference: {np.mean(all_mean_diffs):.2e}")
    print(f"  Average Mean Relative Error: {np.mean(all_rel_errors):.2e}")
    print(f"  Maximum LL Difference: {np.max([exp['max_ll_difference'] for exp in experiments if not np.isnan(exp['max_ll_difference'])]):.2e}")

def main():
    """Main function to run mixed precision comparison."""
    # Experimental parameters
    n_values = [10, 20, 50, 80, 100, 200, 500, 1000]
    nu_values = [0.5, 1.5, 2.5]
    n_trials = 1
    
    # Run experiments
    results = run_mixed_precision_experiment(n_values, nu_values, n_trials)
    
    # Create visualizations
    print("\nCreating visualizations...")
    fig = visualize_mixed_precision_results(results)
    plt.savefig('/home/v-qilongpan/dev/MixedPrecisionSBV/ablation_study/mixed_precision_comparison.pdf', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create summary table
    create_summary_table(results)
    
    return results

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    results = main()
