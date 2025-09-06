#!/usr/bin/env python3
"""
Mixed Precision Multivariate Gaussian Conditional Distribution Simulation

This script explores mixed precision arithmetic in multivariate Gaussian 
conditional distributions using a 2D Gaussian process with Matern kernels.

The setup involves:
- Generate 600 points within a circle with radius 1/n
- Inner circle (radius 1/(n*sqrt(3))) as conditioned part
- Ring part as conditioning part
- Different Matern kernel smoothness: 0.5, 1.5, 2.5
- Different scaling parameters n: 10, 100, 1000
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as spdist
from scipy.special import gamma, kv
from scipy.linalg import cholesky, solve_triangular
import time
from typing import Tuple, Dict, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class MaternKernel:
    """Matern kernel implementation with different smoothness parameters."""
    
    def __init__(self, nu: float, length_scale: float = 1.0, variance: float = 1.0):
        self.nu = nu
        self.length_scale = length_scale
        self.variance = variance
        
    def __call__(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        """Compute Matern kernel matrix."""
        if X2 is None:
            X2 = X1
            
        # Compute pairwise distances
        dists = spdist.cdist(X1, X2, metric='euclidean')
        
        # Avoid division by zero
        dists = np.maximum(dists, 1e-12)
        
        # Scaled distances
        scaled_dists = np.sqrt(2 * self.nu) * dists / self.length_scale
        
        if self.nu == 0.5:
            # Exponential kernel (Matern with nu=0.5)
            K = self.variance * np.exp(-scaled_dists)
        elif self.nu == 1.5:
            K = self.variance * (1 + scaled_dists) * np.exp(-scaled_dists)
        elif self.nu == 2.5:
            K = self.variance * (1 + scaled_dists + scaled_dists**2 / 3) * np.exp(-scaled_dists)
        else:
            # General Matern kernel
            scaled_dists = np.maximum(scaled_dists, 1e-8)
            temp = (2**(1-self.nu)) / gamma(self.nu)
            K = self.variance * temp * (scaled_dists**self.nu) * kv(self.nu, scaled_dists)
            
        # Set diagonal to variance (avoid numerical issues)
        if X1.shape == X2.shape and np.allclose(X1, X2):
            np.fill_diagonal(K, self.variance)
            
        return K

class GaussianProcess:
    """Gaussian Process with mixed precision support."""
    
    def __init__(self, kernel: MaternKernel, noise_variance: float = 1e-6):
        self.kernel = kernel
        self.noise_variance = noise_variance
        
    def sample_prior(self, X: np.ndarray, n_samples: int = 1, 
                    precision: str = 'double') -> np.ndarray:
        """Sample from GP prior with specified precision."""
        # Set precision
        dtype = np.float64 if precision == 'double' else np.float32
        X = X.astype(dtype)
        
        # Compute covariance matrix
        K = self.kernel(X).astype(dtype)
        K += self.noise_variance * np.eye(len(X), dtype=dtype)
        
        # Cholesky decomposition
        try:
            L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            # Add more noise if matrix is not positive definite
            K += 1e-3 * np.eye(len(X), dtype=dtype)
            L = cholesky(K, lower=True)
        
        # Sample
        samples = []
        for _ in range(n_samples):
            z = np.random.randn(len(X)).astype(dtype)
            sample = L @ z
            samples.append(sample)
            
        return np.array(samples).T if n_samples > 1 else samples[0]
    
    def conditional_distribution(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, precision: str = 'double') -> Tuple[np.ndarray, np.ndarray]:
        """Compute conditional mean and covariance with specified precision."""
        # Set precision
        dtype = np.float64 if precision == 'double' else np.float32
        X_train = X_train.astype(dtype)
        y_train = y_train.astype(dtype)
        X_test = X_test.astype(dtype)
        
        # Compute kernel matrices
        K_train = self.kernel(X_train).astype(dtype)
        K_train += self.noise_variance * np.eye(len(X_train), dtype=dtype)
        K_test = self.kernel(X_test).astype(dtype)
        K_cross = self.kernel(X_train, X_test).astype(dtype)
        
        # Solve for conditional mean
        try:
            L = cholesky(K_train, lower=True)
            alpha = solve_triangular(L, y_train, lower=True)
            beta = solve_triangular(L, K_cross, lower=True)
            
            mean = beta.T @ alpha
            
            # Conditional covariance
            cov = K_test - beta.T @ beta
            
        except np.linalg.LinAlgError:
            # Fallback to direct solve
            K_train += 1e-3 * np.eye(len(X_train), dtype=dtype)
            alpha = np.linalg.solve(K_train, y_train)
            mean = K_cross.T @ alpha
            
            temp = np.linalg.solve(K_train, K_cross)
            cov = K_test - K_cross.T @ temp
        
        return mean, cov
    
    def conditional_log_likelihood(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_test: np.ndarray, y_test: np.ndarray, 
                                 precision: str = 'double') -> float:
        """Compute conditional log-likelihood with specified precision."""
        # Set precision
        dtype = np.float64 if precision == 'double' else np.float32
        X_train = X_train.astype(dtype)
        y_train = y_train.astype(dtype)
        X_test = X_test.astype(dtype)
        y_test = y_test.astype(dtype)
        
        try:
            # Get conditional distribution
            mean, cov = self.conditional_distribution(X_train, y_train, X_test, precision)
            
            # Add small regularization for numerical stability
            cov += 1e-6 * np.eye(len(X_test), dtype=dtype)
            
            # Compute log-likelihood components
            L_cond = cholesky(cov, lower=True)
            
            # Residuals
            residuals = (y_test - mean).astype(dtype)
            
            # Solve L @ v = residuals
            v = solve_triangular(L_cond, residuals, lower=True)
            
            # Log-likelihood = -0.5 * (residuals^T @ cov^-1 @ residuals + log|cov| + n*log(2π))
            quad_form = np.sum(v**2)
            log_det = 2 * np.sum(np.log(np.diag(L_cond)))
            n_test = len(X_test)
            
            log_likelihood = -0.5 * (quad_form + log_det + n_test * np.log(2 * np.pi))
            
            return float(log_likelihood)
            
        except (np.linalg.LinAlgError, ValueError) as e:
            # Return very negative log-likelihood for numerical failures
            return -1e10

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

def run_mixed_precision_experiment(n: int, nu: float, n_samples: int = 10) -> Dict[str, Any]:
    """Run mixed precision experiment for given parameters with focus on log-likelihood."""
    print(f"Running experiment: n={n}, nu={nu}")
    
    # Create kernel and GP
    kernel = MaternKernel(nu=nu, length_scale=0.1, variance=1.0)
    gp = GaussianProcess(kernel, noise_variance=1e-6)
    
    all_points = []
    inner_points = []
    outer_points = []
    y_true_all = []
    y_true_outer = []
    y_true_inner = []

    # Generate spatial points and reference function
    np.random.seed(42 + n + int(nu*10))  # Reproducible
    for trial in range(n_samples):
        _all_points, _inner_points, _outer_points = generate_circle_points(n)
    
        # Generate reference function values (always double precision)
        # _y_true_all = np.random.multivariate_normal(
        #     np.zeros(len(_all_points)), 
        #     kernel(_all_points) # no jitter
        #     # kernel(_all_points, precision='double') + 1e-6 * np.eye(len(_all_points))
        # )
        _y_true_all = np.zeros(len(_all_points))
        _y_true_inner = _y_true_all[:len(_inner_points)]
        _y_true_outer = _y_true_all[len(_inner_points):len(_inner_points)+len(_outer_points)]

        all_points.append(_all_points)
        inner_points.append(_inner_points)
        outer_points.append(_outer_points)
        y_true_all.append(_y_true_all)
        y_true_outer.append(_y_true_outer)
        y_true_inner.append(_y_true_inner)
    results = {
        'n': n,
        'nu': nu,
        'n_inner': np.mean([len(inner_points[i]) for i in range(n_samples)]),
        'n_outer': np.mean([len(outer_points[i]) for i in range(n_samples)]),
        'precision_comparison': {}
    }
    # Test both precisions
    for precision in ['double', 'single']:
        print(f"  Testing {precision} precision...")
        
        precision_results = {
            'computation_times': [],
            'condition_numbers': [],
            'log_likelihoods': [],
            'memory_usage': [],
            'numerical_failures': 0
        }
        
        for sample_idx in range(n_samples):
            start_time = time.time()
            
            # Compute conditional log-likelihood
            log_likelihood = gp.conditional_log_likelihood(
                outer_points[sample_idx], y_true_outer[sample_idx], inner_points[sample_idx], y_true_inner[sample_idx], precision=precision
            )
            # print(f"Log-likelihood: {log_likelihood}")
            
            computation_time = time.time() - start_time
            
            # Check for numerical failures
            if log_likelihood <= -1e9:
                precision_results['numerical_failures'] += 1
                continue
            
            # Compute condition number of conditioning matrix
            try:
                K_outer = kernel(outer_points[sample_idx])
                if precision == 'single':
                    K_outer = K_outer.astype(np.float32)
                K_outer += gp.noise_variance * np.eye(len(outer_points[sample_idx]))
                cond_num = np.linalg.cond(K_outer)
            except:
                cond_num = np.inf
            
            # Memory usage (approximate)
            dtype_size = 8 if precision == 'double' else 4
            memory_kb = (len(outer_points[sample_idx])**2 + len(inner_points[sample_idx])**2) * dtype_size / 1024
            
            precision_results['computation_times'].append(computation_time)
            precision_results['condition_numbers'].append(cond_num)
            precision_results['log_likelihoods'].append(log_likelihood)
            precision_results['memory_usage'].append(memory_kb)
        
        # Aggregate results
        for key in ['computation_times', 'condition_numbers', 'log_likelihoods', 'memory_usage']:
            values = np.array(precision_results[key])
            valid_values = values[~np.isnan(values) & ~np.isinf(values)]
            if len(valid_values) > 0:
                precision_results[key] = {
                    'mean': np.mean(valid_values),
                    'std': np.std(valid_values),
                    'min': np.min(valid_values),
                    'max': np.max(valid_values),
                    'median': np.median(valid_values)
                }
            else:
                precision_results[key] = {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan, 'median': np.nan}
        
        results['precision_comparison'][precision] = precision_results
    
    return results

def visualize_results(all_results: list):
    """Create visualizations of the experimental results focused on log-likelihood."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Mixed Precision Analysis: Conditional Log-Likelihood in Gaussian Processes', fontsize=16)
    
    # Extract data for plotting
    n_values = sorted(set(r['n'] for r in all_results))
    nu_values = sorted(set(r['nu'] for r in all_results))
    
    # Plot 1: Log-likelihood comparison
    ax = axes[0, 0]
    for nu in nu_values:
        ll_double = []
        ll_single = []
        ns = []
        for n in n_values:
            result = next((r for r in all_results if r['n'] == n and r['nu'] == nu), None)
            if result:
                ll_d = result['precision_comparison']['double']['log_likelihoods']['mean']
                ll_s = result['precision_comparison']['single']['log_likelihoods']['mean']
                if not np.isnan(ll_d) and not np.isnan(ll_s):
                    ll_double.append(ll_d)
                    ll_single.append(ll_s)
                    ns.append(n)
        
        ax.plot(ns, ll_double, 'o-', label=f'Double (ν={nu})', alpha=0.7)
        ax.plot(ns, ll_single, 's--', label=f'Single (ν={nu})', alpha=0.7)
    
    ax.set_xlabel('n (Scale Parameter)')
    ax.set_ylabel('Conditional Log-Likelihood')
    ax.set_title('Conditional Log-Likelihood vs Scale')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Log-likelihood difference (precision loss)
    ax = axes[0, 1]
    for nu in nu_values:
        ll_diff = []
        ns = []
        for n in n_values:
            result = next((r for r in all_results if r['n'] == n and r['nu'] == nu), None)
            if result:
                ll_d = result['precision_comparison']['double']['log_likelihoods']['mean']
                ll_s = result['precision_comparison']['single']['log_likelihoods']['mean']
                if not np.isnan(ll_d) and not np.isnan(ll_s):
                    diff = ll_d - ll_s  # Positive means double precision is better
                    ll_diff.append(diff)
                    ns.append(n)
        
        ax.plot(ns, ll_diff, 'o-', label=f'ν={nu}', alpha=0.7)
    
    ax.set_xlabel('n (Scale Parameter)')
    ax.set_ylabel('Log-Likelihood Difference (Double - Single)')
    ax.set_title('Precision Loss in Log-Likelihood')
    ax.set_xscale('log')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='No difference')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # # Plot 3: Computation time comparison
    # ax = axes[0, 2]
    # for nu in nu_values:
    #     double_times = []
    #     single_times = []
    #     ns = []
    #     for n in n_values:
    #         result = next((r for r in all_results if r['n'] == n and r['nu'] == nu), None)
    #         if result:
    #             double_time = result['precision_comparison']['double']['computation_times']['mean']
    #             single_time = result['precision_comparison']['single']['computation_times']['mean']
    #             if not np.isnan(double_time) and not np.isnan(single_time):
    #                 double_times.append(double_time)
    #                 single_times.append(single_time)
    #                 ns.append(n)
        
    #     ax.plot(ns, double_times, 'o-', label=f'Double (ν={nu})', alpha=0.7)
    #     ax.plot(ns, single_times, 's--', label=f'Single (ν={nu})', alpha=0.7)
    
    # ax.set_xlabel('n (Scale Parameter)')
    # ax.set_ylabel('Computation Time (s)')
    # ax.set_title('Computation Time vs Scale')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.legend()
    # ax.grid(True, alpha=0.3)
    
    # Plot 4: Numerical failure rate
    ax = axes[1, 0]
    n_samples = 10  # This should match the experiment parameter
    for nu in nu_values:
        failure_rates = []
        ns = []
        for n in n_values:
            result = next((r for r in all_results if r['n'] == n and r['nu'] == nu), None)
            if result:
                failures_double = result['precision_comparison']['double']['numerical_failures']
                failures_single = result['precision_comparison']['single']['numerical_failures']
                
                rate_double = failures_double / n_samples * 100
                rate_single = failures_single / n_samples * 100
                
                # Only plot if there are failures to show
                if rate_double > 0 or rate_single > 0:
                    ax.bar([f"{n}_D"], [rate_double], alpha=0.7, color=f'C{nu_values.index(nu)}', 
                          label=f'Double (ν={nu})' if n == n_values[0] else "")
                    ax.bar([f"{n}_S"], [rate_single], alpha=0.7, color=f'C{nu_values.index(nu)}', 
                          hatch='//', label=f'Single (ν={nu})' if n == n_values[0] else "")
                    ns.append(n)
    
    ax.set_xlabel('n (Scale Parameter) and Precision')
    ax.set_ylabel('Numerical Failure Rate (%)')
    ax.set_title('Numerical Failure Rates')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Speedup ratio
    ax = axes[1, 1]
    for nu in nu_values:
        speedups = []
        ns = []
        for n in n_values:
            result = next((r for r in all_results if r['n'] == n and r['nu'] == nu), None)
            if result:
                time_double = result['precision_comparison']['double']['computation_times']['mean']
                time_single = result['precision_comparison']['single']['computation_times']['mean']
                if not np.isnan(time_double) and not np.isnan(time_single) and time_single > 0:
                    speedup = time_double / time_single
                    speedups.append(speedup)
                    ns.append(n)
        
        ax.plot(ns, speedups, 'o-', label=f'ν={nu}', alpha=0.7)
    
    ax.set_xlabel('n (Scale Parameter)')
    ax.set_ylabel('Speedup (Double/Single)')
    ax.set_title('Single Precision Speedup')
    ax.set_xscale('log')
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='No speedup')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Sample geometry
    ax = axes[1, 2]
    # Show example geometry for n=100
    points, inner_points, outer_points = generate_circle_points(100)
    ax.scatter(outer_points[:, 0], outer_points[:, 1], c='blue', s=10, alpha=0.6, label='Conditioning')
    ax.scatter(inner_points[:, 0], inner_points[:, 1], c='red', s=10, alpha=0.8, label='Conditioned')
    
    # Draw circles
    theta = np.linspace(0, 2*np.pi, 100)
    outer_radius = 1.0 / 100
    inner_radius = 1.0 / (100 * np.sqrt(3))
    ax.plot(outer_radius * np.cos(theta), outer_radius * np.sin(theta), 'b-', alpha=0.5)
    ax.plot(inner_radius * np.cos(theta), inner_radius * np.sin(theta), 'r-', alpha=0.5)
    
    ax.set_aspect('equal')
    ax.set_title('Sample Geometry (n=100)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    """Main function to run all experiments."""
    print("Mixed Precision Gaussian Conditional Distribution Simulation")
    print("=" * 60)
    
    # Experimental parameters
    # n_values = [10, 100, 1000]
    n_values = [10, 20, 50, 80]
    nu_values = [0.5, 1.5, 2.5]
    n_samples = 10  # Reduced for faster execution
    
    all_results = []
    
    # Run experiments
    total_experiments = len(n_values) * len(nu_values)
    exp_count = 0
    
    for n in n_values:
        for nu in nu_values:
            exp_count += 1
            print(f"\nExperiment {exp_count}/{total_experiments}")
            
            try:
                result = run_mixed_precision_experiment(n, nu, n_samples)
                all_results.append(result)
            except Exception as e:
                print(f"Error in experiment n={n}, nu={nu}: {e}")
                continue
    
    # Create visualizations
    print("\nCreating visualizations...")
    fig = visualize_results(all_results)
    plt.savefig('/home/v-qilongpan/dev/MixedPrecisionSBV/simulation/mixed_precision_results.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY - CONDITIONAL LOG-LIKELIHOOD ANALYSIS")
    print("=" * 60)
    
    for result in all_results:
        n, nu = result['n'], result['nu']
        n_inner, n_outer = result['n_inner'], result['n_outer']
        
        print(f"\nn={n}, ν={nu} (inner: {n_inner}, outer: {n_outer} points)")
        print("-" * 50)
        
        for precision in ['double', 'single']:
            stats = result['precision_comparison'][precision]
            time_mean = stats['computation_times']['mean']
            cond_mean = stats['condition_numbers']['mean']
            memory_mean = stats['memory_usage']['mean']
            ll_mean = stats['log_likelihoods']['mean']
            failures = stats['numerical_failures']
            
            print(f"{precision.capitalize()} precision:")
            print(f"  Computation time: {time_mean:.4f} ± {stats['computation_times']['std']:.4f} s")
            print(f"  Condition number: {cond_mean:.2e}")
            print(f"  Memory usage: {memory_mean:.2f} KB")
            print(f"  Log-likelihood: {ll_mean:.4f} ± {stats['log_likelihoods']['std']:.4f}")
            print(f"  Numerical failures: {failures}/10")
        
        # Compute log-likelihood difference and speedup
        ll_double = result['precision_comparison']['double']['log_likelihoods']['mean']
        ll_single = result['precision_comparison']['single']['log_likelihoods']['mean']
        time_double = result['precision_comparison']['double']['computation_times']['mean']
        time_single = result['precision_comparison']['single']['computation_times']['mean']
        
        if not np.isnan(ll_double) and not np.isnan(ll_single):
            ll_diff = ll_double - ll_single
            print(f"Log-likelihood difference (double - single): {ll_diff:.4f}")
        
        if not np.isnan(time_double) and not np.isnan(time_single) and time_single > 0:
            speedup = time_double / time_single
            print(f"Speedup (single vs double): {speedup:.2f}x")
    
    print(f"\nResults saved to: /home/v-qilongpan/dev/MixedPrecisionSBV/simulation/mixed_precision_results.png")
    
    return all_results

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    results = main()
