# Gaussian Process Simulation Experiments

This directory contains scripts and notebooks for running Gaussian Process (GP) simulation experiments to evaluate Scaled Block Vecchia (SBV) approximation methods.

## Directory Structure

- `GPsimu.ipynb`: Jupyter notebook for running and visualizing GP simulations
- `GPsimu-mspe.sh`: Script for evaluating Mean Squared Prediction Error (MSPE) across different configurations
- `GPsimu-kl.sh`: Script for Kullback-Leibler divergence evaluation
- `GP-simu-kl_bc_bs.sh`: Script for evaluating KL divergence with different block counts and block sizes

## Reproducing the Experiments

### Step 1: Data preparation

Run the first three blocks in `GPsimu-mspe.sh` to prepare the simulated GPs data in high dimensional settings.

### Step 2: MSPE Evaluation

Run the MSPE evaluation script:

```bash
bash GPsimu-mspe.sh
```

<!-- This script evaluates the Mean Squared Prediction Error for:
- Different nearest neighbor configurations (m_test): 5, 10, 20, 30, 50, ..., 500
- Both scaled and unscaled distance metrics
- Block Vecchia (with block_test=200) and classic Vecchia (with block_test=2000) -->

Key parameters:
- Training points: 5,000
- Test points: 2,000
- Training blocks: 500
- Dimension: 8
- Using true hyperparameters from `hyperparameters.csv`

Results will be saved to `./log/mspe-matern72-simu/`.

### Step 3: KL Divergence Evaluation

Run the KL divergence evaluation script:

```bash
bash GPsimu-kl.sh
```

This script evaluates the Kullback-Leibler divergence between the approximate posterior and the true posterior for various configurations.

### Step 3: Block Count/Size Evaluation

For evaluating the impact of different block counts and sizes:

```bash
bash GP-simu-kl_bc_bs.sh
```

### Step 4: Comprehensive Analysis

Run the Jupyter notebook for comprehensive analysis and visualization:

```bash
jupyter notebook GPsimu.ipynb
```

This notebook:
- Generates synthetic data if needed
- Analyzes MSPE and KL divergence results
- Creates visualizations for comparison
- Provides detailed analysis of the results

## Output Files

- MSPE evaluation results: `./log/mspe-matern72-simu/*.csv`
- KL divergence results: Stored in respective output directories

## Notes

- Easy to reproduce within a few minutes.