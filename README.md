# Distributed Scaled Block Vecchia Approximation for Gaussian Processes (SBV-GP)

This repository contains the implementation of the Distributed Scaled Block Vecchia (SBV) method for large-scale Gaussian Process approximation. The code provides a highly efficient distributed implementation optimized for GPU acceleration that can handle massive datasets.

## Overview

SBV-GP is a scalable method for approximating Gaussian Processes that:

- Uses a block-based approach to partition data
- Leverages GPU acceleration for covariance computations
- Distributes computation across multiple nodes with MPI
- Supports different covariance kernels including Matérn family

The implementation is optimized for modern high-performance computing environments with multi-GPU support.

## Features

- **Scalable**: Handles millions of data points with efficient block-based partitioning
- **GPU-Accelerated**: Optimized for NVIDIA GPUs with MAGMA linear algebra acceleration
- **Distributed**: MPI-based implementation for multi-node execution
- **Flexible Covariance Functions**: Supports multiple kernels including PowerExponential and Matérn family (1/2, 3/2, 5/2, 7/2)
- **Parameter Estimation**: Implements efficient likelihood-based parameter optimization
- **Prediction**: Fast prediction at new locations by leveraging the approximation structure

## Prerequisites

The following libraries and tools are required:

- C++ compiler GCC 13.3.0
- CUDA 11.8/12/9
- MAGMA 2.7.2 - GPU-accelerated linear algebra library
- NLopt 2.7.1 - Nonlinear optimization library
- OpenMPI 5.0.5 - MPI implementation

## Installation

1. Clone the repository:
   ```bash
   cd DSBV-GPs
   ```

2. Build the code:
   ```bash
   # For systems using OpenMPI
   make SYSTEM=OPENMPI
   
   # For Cray systems 
   make SYSTEM=CRAY
   ```

The build system will automatically detect available GPUs and configure the compilation accordingly.

## Usage

### Basic Examples

To run a simple example using a single GPU:

```bash
# For systems with SLURM (specify your node in batch files)
srun ./bin/dbv --num_total_points 2000000 --num_total_blocks 10000 -m 800 --dim 10 --mode estimation --maxeval 100 --theta_init 1.5,0.0 --distance_scale 0.05,0.5,0.05,1,1,1,1,1,1,1 --kernel_type Matern72 --nn_multiplier 10 --seed 7

# For systems without SLURM (direct MPI execution)
mpirun -np 4 ./bin/dbv --num_total_points 8000000 --num_total_blocks 80000 -m 400 --dim 10 --mode estimation --maxeval 100 --theta_init 1.5,0.0 --distance_scale 1,1,1,1,1,1,1,1,1,1 --kernel_type Matern72 --nn_multiplier 10 --seed 8
```

### Command Line Options

SBV-GP provides many configuration options:

#### Data Configuration
- `--num_total_points <N>`: Total number of points in the dataset (default: 20000)
- `--num_total_blocks <B>`: Total number of blocks for partitioning (default: 1000)
- `--num_total_points_test <N>`: Total number of points for testing (default: 2000)
- `--num_total_blocks_test <B>`: Total number of blocks for testing (default: 100)
- `--dim <D>`: Dimension of the input space (default: 8)
- `--train_metadata_path <PATH>`: Path to the training metadata file (default: "")
- `--test_metadata_path <PATH>`: Path to the testing metadata file (default: "")

#### Method Parameters
- `-m <M>`: Number of nearest neighbors to consider (default: 200)
- `--m_test <M>`: Number of nearest neighbors for testing (default: 120)
- `--nn_multiplier <M>`: Number of nearest neighbors multiplier ($\alpha$) (default: 400)
- `--distance_scale <S>`: Distance scales for each dimension (default: all 1.0)
- `--distance_scale_init <S>`: Initial distance scales for optimization (default: same as distance_scale)

#### Kernel Configuration
- `--kernel_type <TYPE>`: Covariance kernel type (default: "Matern72")
  - Options: "Matern12", "Matern32", "Matern52", "Matern72"
- `--theta_init <θ>`: Initial parameter values (comma-separated, defaults depend on kernel)
  - For PowerExponential kernel: variance, smoothness, nugget (default: 1.0, 0.5, 0.0)
  - For Matérn kernels: variance, nugget (default: 1.0, 0.0)

#### Optimization Settings
- `--mode <MODE>`: Operation mode (default: "estimation")
  - Options: "estimation", "prediction", "full"
- `--maxeval <N>`: Maximum number of function evaluations (default: 500)
- `--xtol_rel <T>`: Relative tolerance for optimization (default: 1e-5)
- `--ftol_rel <T>`: Relative tolerance of function for optimization (default: 1e-5)
- `--lower_bounds <B>`: Lower bounds for optimization (default: theta_init * 0.001)
- `--upper_bounds <B>`: Upper bounds for optimization (default: theta_init * 10)

#### Execution Configuration
- `--omp_num_threads <T>`: Number of OpenMP threads per MPI process (default: 20)
- `--seed <S>`: Seed for random number generator (default: 0)
- `--print`: Print additional information (default: true)
- `--log_append <S>`: Append string to log file names (default: "")

#### Advanced Configuration
- `--partition <TYPE>`: Partition type (default: "linear")
  - Options: "linear", "none"
- `--clustering <TYPE>`: Clustering method for large datasets (default: "random")
  - Options: "random", "kmeans++"
- `--kmeans_max_iter <N>`: Maximum iterations for k-means++ (default: 10)
- `--num_simulations <N>`: Number of simulations for evaluation (default: 1000)

Run `./bin/dbv --help` to see all available options.

### Example Workflows

#### Parameter Estimation

Parameter estimation finds optimal hyperparameters for the Gaussian process model:

```
mpirun -n 4 ./bin/dbv \
    --num_total_points 1000000 \
    --num_total_blocks 10000 \
    --distance_scale 0.05,0.05,0.1,1.0,1.0,1.0,1.0,1.0 \
    --theta_init 1.0,0.001 \
    -m 200 \
    --dim 8 \
    --mode estimation \
    --maxeval 500 \
    --xtol_rel 1e-6 \
    --ftol_rel 1e-6 \
    --kernel_type Matern72 \
    --seed 42 \
    --nn_multiplier 500 \
    --omp_num_threads 20 \
    --log_append "estimation_run1"
```

This performs maximum likelihood estimation for the Matérn 7/2 kernel parameters. The results are saved to log files with the suffix "estimation_run1".

#### Prediction at New Locations

After parameter estimation, you can make predictions at new test locations:

```
mpirun -n 4 ./bin/dbv \
    --num_total_points 1000000 \
    --num_total_blocks 10000 \
    --num_total_points_test 10000 \
    --num_total_blocks_test 1000 \
    --distance_scale 0.05,0.05,0.1,1.0,1.0,1.0,1.0,1.0 \
    --theta_init 1.0,0.001 \
    -m 200 \
    --m_test 60 \
    --dim 8 \
    --mode prediction \
    --kernel_type Matern72 \
    --nn_multiplier 500 \
    --omp_num_threads 20 \
    --log_append "prediction_run1"
```

#### Full Workflow (Estimation + Prediction)

You can perform both estimation and prediction in a single run:

```
mpirun -n 4 ./bin/dbv \
    --num_total_points 1000000 \
    --num_total_blocks 10000 \
    --num_total_points_test 10000 \
    --num_total_blocks_test 1000 \
    --distance_scale 0.05,0.05,0.1,1.0,1.0,1.0,1.0,1.0 \
    --theta_init 1.0,0.001 \
    -m 200 \
    --m_test 120 \
    --dim 8 \
    --mode full \
    --maxeval 10 \
    --kernel_type Matern72 \
    --log_append "full_run1"
```

### Using Real Datasets

For real datasets, data should be formatted with one point per line:

```
x1,x2,...,xD,y
```

where `x1, x2, ..., xD` are the D-dimensional input coordinates and `y` is the observation value.

Use the `--train_metadata_path` and `--test_metadata_path` options to specify the paths to your data files.

## Performance Optimization

For optimal performance:

1. Choose the number of blocks based on your GPU memory
2. Set the number of nearest neighbors (-m) according to desired accuracy and computational budget
3. Adjust `--omp_num_threads` based on your CPU resources
4. Use `--nn_multiplier` to control the coarseness of nearest neighbor search
5. For higher-dimensional problems, tune `--distance_scale` parameters to give appropriate weights to different dimensions

## Reproducity 

See `experiments`

## Citation

If you use this code in your research, please cite:

```
@article{...},
```

## License

[License information]

