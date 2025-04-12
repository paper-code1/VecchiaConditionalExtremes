# BV-MPI Experiments

This directory contains various experiments to evaluate and demonstrate the Scaled Block Vecchia (SBV) implementation for large-scale Gaussian Process modeling. Each subdirectory contains specific experiments with their own README files and reproduction instructions.

## Overview of Experiments

### [metaRVM](./metaRVM/)

Experiments focused on metaRVM simulator applications:
- Parameter estimation and prediction on large-scale real datasets
- Uses 45M training points and 5M test points
- Demonstrates SBV capability on high-dimensional data (10D)

### [satellite](./satellite/)

Benchmark experiments for satellite drag modeling:
- Cross-validation tests across 6 chemical species (N2, N, O, He, O2, H)
- Each species has 10-fold cross-validation for robust evaluation
- Demonstrates SBV effectiveness for real-world scientific applications

### [scale](./scale/)

GPU scaling experiments to evaluate performance on different hardware:
- Compares A100 and H100/GH200 GPU performance
- Tests both strong scaling (fixed problem size, more GPUs) and weak scaling (larger problems with more GPUs)
- Evaluates power efficiency and computational throughput
- Includes single and multi-GPU configurations (1, 2, 4, 8, 16, 32, 64 GPUs)

### [simulations](./simulations/)

Controlled simulation experiments to evaluate approximation quality:
- Mean Squared Perdiction Error (MSPE) evaluation
- Kullback-Leibler divergence measurements
- Comparison of 4 Vecchia-based GPs
- Effects of different block counts, block sizes, and nearest neighbor configurations

## Prerequisites for All Experiments

- Compiled BV-MPI binary (`dbv`) in the project's `bin/` directory
- Python environment with Jupyter Notebook support
- R environment for data generation and certain analyses
- Access to a compute cluster with SLURM scheduler and GPU resources

## General Workflow

Most experiments follow this general workflow:
0. All sbatch jobs should be submitted at the home directory `./BSV-GP/`
1. Data preparation (generation or preprocessing)
2. Parameter estimation runs
3. Prediction runs using estimated parameters
4. Analysis and visualization of results

<!-- ## Directory Structure

```
experiments/
├── metaRVM/           
├── satellite/         # Satellite drag modeling benchmarks
├── scale/             # GPU scaling experiments
├── simulations/       # Controlled simulation experiments
``` -->

## Output Structure

Most experiments save outputs to `./log/` directories with subdirectories for specific experiment types. Visualization notebooks typically read from these logs and generate figures in the `fig/` directory.

<!-- ## Reproducing All Experiments

Each subdirectory contains its own README with specific instructions for reproducing the experiments. For a complete reproduction of all experiments, follow these steps in order:

1. First run the simulation experiments to validate the methodology
2. Then run the GPU scaling experiments to determine the optimal hardware configuration
3. Finally, run the application-specific experiments (metaRVM and satellite)

See individual README files in each subdirectory for detailed instructions.  -->