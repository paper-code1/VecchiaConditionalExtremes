import numpy as np
import scipy.spatial.distance as spdist
import warnings
import matplotlib
from scipy.special import gamma, kv
from scipy.linalg import cholesky, solve_triangular

from typing import Tuple, Dict, Any, List
matplotlib.rcParams.update({'font.size': 18})

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

# mimic the worst approximation, the first block in the Vecchia approximation
def generate_circle_points(n: int, n_points: int = 600) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate points in circle with inner/outer regions."""
    outer_radius = 1.0 / n
    inner_radius = 1.0 / (n * np.sqrt(3))
    n_inner = int(n_points / 3)
    n_outer = n_points - n_inner

    points_inner = []
    while len(points_inner) < n_inner:
        # Generate random points in square
        x = np.random.uniform(-inner_radius, inner_radius)
        y = np.random.uniform(-inner_radius, inner_radius)
        r = np.sqrt(x**2 + y**2)
        
        # Keep points within outer circle
        if r <= inner_radius:
            points_inner.append([x, y])
    
    points_inner = np.array(points_inner)
    
    # outer points
    points_outer = []
    while len(points_outer) < n_outer:
        # Generate random points in square
        x = np.random.uniform(0.5 - outer_radius, 0.5 + outer_radius)
        y = np.random.uniform(0.5 - outer_radius, 0.5 + outer_radius)
        r = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
        
        # Keep points within outer circle
        if r <= outer_radius:
            points_outer.append([x, y])
    
    points_outer = np.array(points_outer)
    points = np.concatenate([points_inner, points_outer], axis=0)
    
    return points, points_inner, points_outer

# # mimic the best approximation, the end block in the Vecchia approximation
# def generate_circle_points(n: int, n_points: int = 600) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """Generate points in circle with inner/outer regions."""
#     outer_radius = 1.0 / n
#     inner_radius = 1.0 / (n * np.sqrt(3))
    
#     points = []
#     while len(points) < n_points:
#         # Generate random points in square
#         x = np.random.uniform(-outer_radius, outer_radius)
#         y = np.random.uniform(-outer_radius, outer_radius)
#         r = np.sqrt(x**2 + y**2)
        
#         # Keep points within outer circle
#         if r <= outer_radius:
#             points.append([x, y])
    
#     points = np.array(points[:n_points])
    
#     # Classify points
#     distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
#     inner_mask = distances <= inner_radius
#     outer_mask = ~inner_mask
    
#     inner_points = points[inner_mask]
#     outer_points = points[outer_mask]
#     points = np.concatenate([inner_points, outer_points], axis=0)
    
#     return points, inner_points, outer_points
