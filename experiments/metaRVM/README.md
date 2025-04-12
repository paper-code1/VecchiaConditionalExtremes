# metaRVM Experiments

This directory contains scripts and notebooks for running and visualizing the metaRVM (Meta Relevance Vector Machine) experiments on real datasets.

## Directory Structure

- `RealDataset_full_estimation_directly.sh`: Script for parameter estimation on full real dataset
- `RealDataset_full_prediction_directly.sh`: Script for prediction on full real dataset
- `RealDatasetVisu.ipynb`: Notebook for visualizing results from real dataset experiments
- `metaRVM_dataGen.R`: R script for data generation
- `fig/`: Directory containing generated figures

## Prerequisites

- R environment with required packages (see `metaRVM_dataGen.R` for dependencies)
- SLURM system

## Data Requirements

The scripts expect training and test datasets is generated from `metaRVM_dataGen.R`, where you need simultaneously generate 50 subsets and then combine them together using first block in `RealDatasetVisu.ipynb`

## Reproducing the Experiments

### Step 1: Data Generation (if needed)

```bash
Rscript metaRVM_dataGen.R [seed]
```

### Step 2: Parameter Estimation

Submit the parameter estimation job to the SLURM scheduler:

```bash
sbatch RealDataset_full_estimation_directly.sh
```

<!-- The script sets up an MPI job on 3 nodes with 4 tasks per node, each task with 1 GPU. It runs parameter estimation with different block counts and nearest neighbor configurations. -->

Key parameters:
- Total points: 45,000,000
- Block count: 450,000
- Nearest neighbors for estimation: 100, 200, 400
- Dimension: 10
- Kernel type: Matern72

Results will be saved to `./log/RealDataset/` and it takes around 6 hours.

### Step 3: Prediction

After parameter estimation is complete, submit the prediction job and it takes around 1 hours:

```bash
sbatch RealDataset_full_prediction_directly.sh
```

This script uses the estimated parameters from Step 2 and performs prediction on the test dataset using various nearest neighbor configurations (100, 200, 400, 600).

Results will be saved to `./log/RealDataset/`.

### Step 4: Visualization

Run the visualization notebook to analyze the results:

```bash
jupyter notebook RealDatasetVisu.ipynb
```

The notebook loads the results from `./log/RealDataset/` and generates visualizations that will be saved to the `fig/` directory.

## Output Files

- Parameter estimation results: `./log/RealDataset/theta_numPointsTotal*_numBlocksTotal*_m*_seed*_isScaled1_RealDataset.csv`
- Prediction results: `./log/RealDataset/*RealDataset_prediction.csv`

## Notes

- The esitmation will take around 6 hours.
- The prediction will takes around 1 hours.