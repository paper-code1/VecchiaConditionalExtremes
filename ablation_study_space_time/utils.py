import numpy as np
import scipy.spatial.distance as spdist
import warnings
import matplotlib
import matplotlib.pyplot as plt
import json
from datetime import datetime
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from scipy.special import gamma, kv
from scipy.linalg import cholesky, solve_triangular
from scipy.interpolate import griddata

from typing import Tuple, Dict, Any, List, Optional, Union
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


class MaternKernelHighDimTime:
    """
    Matern kernel implementation with different 
    smoothness parameters for high-dimensional time-series data.
    """
    
    def __init__(self, nu_space: float, nu_time: float, variance: float = 1.0,
                 length_scale: float = [0.05, 0.05, 0.05, 5, 5, 5, 5, 5, 5, 5],
                 length_dim: int = 10,
                 time_scale: float = 1.0,
                 time_lag: int = 2,
                 seperablity: float = 0.0):
        assert len(length_scale) == length_dim, "length_scale must be a list of length length_dim"

        self.nu_space = nu_space
        self.nu_time = nu_time
        self.length_scale = np.array(length_scale)
        self.variance = variance
        self.length_dim = length_dim
        self.time_scale = time_scale
        self.time_lag = time_lag
        self.seperablity = seperablity
        
    def __call__(self, 
                 X1: np.ndarray, 
                 X2: np.ndarray = None, 
                 precision: str = 'double') -> np.ndarray:
        """Compute Matern kernel matrix with specified precision."""
        if X2 is None:
            X2 = X1
            
        # Set precision for computation
        dtype = np.float64 if precision == 'double' else np.float32
        X1_space = X1.astype(dtype)[:, :self.length_dim] / self.length_scale
        X2_space = X2.astype(dtype)[:, :self.length_dim] / self.length_scale
        X1_time = X1.astype(dtype)[:, self.length_dim].reshape(-1, 1)
        X2_time = X2.astype(dtype)[:, self.length_dim].reshape(-1, 1)
        
        # Compute pairwise distances
        dists_space = spdist.cdist(X1_space, X2_space, metric='euclidean').astype(dtype)
        dists_time = spdist.cdist(X1_time, X2_time, metric='euclidean').astype(dtype)
        
        # Avoid division by zero
        # dists = np.maximum(dists, np.finfo(dtype).eps)
        
        # Scaled distances
        scaled_dists_space = dists_space
        scaled_dists_time = np.power(np.abs(dists_time), self.nu_time * 2) \
            / self.time_scale + 1
        scaled_combined_dist = scaled_dists_space / \
            np.power(scaled_dists_time, self.seperablity/2)
        
        if self.nu_space == 0.5:
            # Exponential kernel (Matern with nu=0.5)
            _item_poly = 1.0
        elif self.nu_space == 1.5:
            _item_poly = 1.0 + scaled_combined_dist
        elif self.nu_space == 2.5:
            _item_poly = 1.0 + scaled_combined_dist + scaled_combined_dist**2 / 3
        elif self.nu_space == 3.5:
            _item_poly = 1.0 + scaled_combined_dist + 2 * scaled_combined_dist**2 / 5 +\
                scaled_combined_dist**3 / 15
        else:
            raise ValueError(f"Matern kernel with nu={self.nu_space} is not supported")
            # # General Matern kernel
            # scaled_dists = np.maximum(scaled_dists, np.finfo(dtype).eps)
            # temp = (2**(1-self.nu)) / gamma(self.nu)
            # K = self.variance * temp * (scaled_dists**self.nu) * kv(self.nu, scaled_dists)
        # matern kernel
        K = self.variance * _item_poly * np.exp(-scaled_combined_dist)    
        # time fixed
        K = K / scaled_dists_time
        
        # Set diagonal to variance (avoid numerical issues)
        if X1.shape == X2.shape and np.allclose(X1.astype(np.float64), X2.astype(np.float64)):
            np.fill_diagonal(K, self.variance)
            
        return K.astype(dtype)


class AblationGaussianProcess:
    """Gaussian Process with ablation study capabilities for precision effects."""
    
    def __init__(self, kernel: MaternKernel, noise_variance: float = 1e-5):
        self.kernel = kernel
        self.noise_variance = noise_variance
        
    def conditional_log_likelihood_ablation(self, 
                                            X_train: np.ndarray, y_train: np.ndarray,
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


# mimic the best/good/worst approximation, the end block in the Vecchia approximation
def generate_circle_points(n: int, n_points: int = 300, 
                           quality: str = 'best', 
                           time_lag: int = 2,
                           dim_length: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate points in circle with inner/outer regions."""
    outer_radius = 1.0 / n
    inner_radius = 1.0 / (n * np.power(3, 1/dim_length))
    
    points = []
    while len(points) < n_points:
        # Generate random points in square
        x = np.zeros(dim_length)
        for i in range(dim_length):
            x[i] = np.random.uniform(-outer_radius, outer_radius)
        
        r = np.sqrt(np.sum(x**2))
        
        # Keep points within outer circle
        if r <= outer_radius:
            points.append(x)
    
    points = np.array(points[:n_points])

    # Classify points
    distances = np.sqrt(np.sum(points**2, axis=1))
    inner_mask = distances <= inner_radius
    outer_mask = ~inner_mask
    
    # inner and outer points are only spatial points; need to add time as the last column.
    # time_lag: how many time tags, each time has the same spatial points

    inner_points_spatial = points[inner_mask]
    outer_points_spatial = points[outer_mask]
    
    # print(f"Generated {len(points)} total points:")
    # print(f"  - Inner region: {len(inner_points_spatial)} points")
    # print(f"  - Outer region: {len(outer_points_spatial)} points")

    if quality == 'good':
        # mimic the good approximation, to expand the outer nearest points
        outer_points_spatial *= np.sqrt(n)
    elif quality == 'worst':
        # mimic the worst approximation, to shrink the outer nearest points, do a shift
        outer_points_spatial *= n
        outer_points_spatial += 0.5

    # For each time tag, repeat the spatial points and append the time as the last column
    def add_time_tags(spatial_points, time_lag):
        n_points = spatial_points.shape[0]
        # Repeat for each time tag
        all_points = []
        for t in range(time_lag):
            time_col = np.full((n_points, 1), t, dtype=np.float64)
            points_with_time = np.hstack([spatial_points, time_col])
            all_points.append(points_with_time)
        return np.vstack(all_points)

    # Only keep inner_points with t = time_lag - 1, rest go to outer_points
    all_inner_points = add_time_tags(inner_points_spatial, time_lag)
    all_outer_points = add_time_tags(outer_points_spatial, time_lag)
    # Split inner_points: keep only those with t = time_lag - 1
    inner_points = all_inner_points[all_inner_points[:, -1] == (time_lag - 1)]
    # The rest of inner_points (t < time_lag - 1) are added to outer_points
    extra_outer_points = all_inner_points[all_inner_points[:, -1] < (time_lag - 1)]
    # Combine original outer_points and extra_outer_points
    outer_points = np.vstack([all_outer_points, extra_outer_points])

    # # Find indices for outer points at the last time step
    # last_time_mask = (outer_points[:, -1] == (time_lag - 1))
    # if quality == 'good':
    #     # mimic the good approximation, to expand the outer nearest points
    #     outer_points[np.ix_(last_time_mask, np.arange(dim_length))] *= np.sqrt(n)
    # elif quality == 'worst':
    #     # mimic the worst approximation, to shrink the outer nearest points, do a shift
    #     outer_points[np.ix_(last_time_mask, np.arange(dim_length))] += 0.5

    # Concatenate all points (inner first, then outer)
    points = np.concatenate([inner_points, outer_points], axis=0)
    
    # print(f"Generated {len(points)} total points:")
    # print(f"  - Inner region: {len(inner_points)} points")
    # print(f"  - Outer region: {len(outer_points)} points")

    return points, inner_points, outer_points

PathLike = Union[str, Path]
DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent / "results"


def _to_serialisable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _to_serialisable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_serialisable(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.generic,)):
        return obj.item()
    return obj


def sanitise_filename_component(value: Any) -> str:
    if isinstance(value, float):
        value_str = f"{value:.3g}"
    elif isinstance(value, (list, tuple)):
        value_str = "-".join(sanitise_filename_component(v) for v in value[:4])
        if len(value) > 4:
            value_str += "-etc"
    else:
        value_str = str(value)

    cleaned = [ch.lower() if ch.isalnum() else '_' for ch in value_str]
    cleaned_str = ''.join(cleaned).strip('_')
    return cleaned_str or "value"


def format_experiment_filename(prefix: str, params: Dict[str, Any], suffix: str = ".json") -> str:
    parts = [prefix]
    for key in sorted(params.keys()):
        value = params[key]
        if value is None:
            continue
        parts.append(f"{key}-{sanitise_filename_component(value)}")
    filename = "_".join(parts)
    return f"{filename}{suffix}"


def prepare_results_dir(
    root: Optional[PathLike] = None,
    timestamp: bool = True,
    subdirs: Optional[List[str]] = None,
) -> Dict[str, Path]:
    root_path = Path(root).expanduser() if root is not None else DEFAULT_RESULTS_ROOT
    experiment_root = root_path

    run_dir = experiment_root
    if timestamp:
        run_dir = experiment_root / datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {'root': run_dir}
    for subdir in subdirs or []:
        sub_path = run_dir / subdir
        sub_path.mkdir(parents=True, exist_ok=True)
        paths[subdir] = sub_path

    return paths


def save_json(data: Any, destination: PathLike) -> None:
    destination_path = Path(destination).expanduser()
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    with destination_path.open('w', encoding='utf-8') as fp:
        json.dump(_to_serialisable(data), fp, indent=2, sort_keys=True)


def write_text_report(lines: List[str], destination: PathLike) -> None:
    destination_path = Path(destination).expanduser()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text("\n".join(lines), encoding='utf-8')


def load_json(source: PathLike) -> Any:
    """Load JSON data from a file."""
    source_path = Path(source).expanduser()
    with source_path.open('r', encoding='utf-8') as fp:
        return json.load(fp)


def load_all_experiment_results(results_dir: PathLike) -> List[Dict[str, Any]]:
    """
    Load all experiment result files from a results directory.
    
    Args:
        results_dir: Path to the directory containing JSON result files
    
    Returns:
        List of all loaded experiment results
    """
    results_path = Path(results_dir).expanduser()
    all_results = []
    
    # Look for JSON files in data subdirectory first, then in root
    search_dirs = [results_path / 'data', results_path]
    
    for search_dir in search_dirs:
        if search_dir.exists():
            json_files = list(search_dir.glob('*.json'))
            for json_file in json_files:
                if 'llh_error_ablation' in json_file.name:
                    try:
                        result = load_json(json_file)
                        if isinstance(result, dict) and 'n' in result and 'nu' in result:
                            all_results.append(result)
                        elif isinstance(result, list):
                            # Handle case where JSON contains a list of results
                            all_results.extend([r for r in result if isinstance(r, dict) and 'n' in r and 'nu' in r])
                    except Exception as e:
                        print(f"Warning: Could not load {json_file}: {e}")
    
    print(f"Loaded {len(all_results)} experiment results from {results_dir}")
    return all_results
