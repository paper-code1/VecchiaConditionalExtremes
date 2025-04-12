# Satellite Drag Experiments

This directory contains scripts and notebooks for running benchmarks on satellite drag modeling using Block Vecchia approximation methods.

## Directory Structure

- `benchmarkRun.sh`: Main script to run the benchmark experiments for different species
- `benchmarkDataPreprocessing.ipynb`: Notebook for data preprocessing
- `benchmarkSummaryAndVisu.ipynb`: Notebook for visualizing benchmark results
- `satelliteDrag_comparison.R`: R script for comparison analysis
- `vecchia_scaled.R`: Implementation of scaled Vecchia methods in R
- `fig/`: Directory containing generated figures

## Prerequisites

- R environment with required packages (see `satelliteDrag_comparison.R` for dependencies)
- SLURM system on GH200/A100

## Data Requirements

`benchmarkDataPreprocessing.ipynb` helps to download the satellite drag benchmark dataset and preprocessing the dataset automatically.

## Reproducing the Experiments

### Step 1: Data Preprocessing

Run the data preprocessing notebook:

```bash
jupyter notebook benchmarkDataPreprocessing.ipynb
```

### Step 2: Run Benchmark Experiments

Submit the benchmark job to the SLURM scheduler:

```bash
sbatch benchmarkRun.sh
```

The script performs:
1. Parameter estimation for each species across 10 cross-validation folds
2. Prediction using the estimated parameters

Key parameters:
- Training points: 1,800,000
- Test points: 200,000
- Block count (train): 18,000
- Block count (test): 40,000
- Nearest neighbors for estimation: 100, 200, 400
- Nearest neighbors for prediction: 200, 400, 600
- Dimension: 8
- Kernel type: Matern72

Results will be saved to `./log/satellite/{species}/`.

### Step 3: Run R Comparison (Optional)

For comparison with traditional Vecchia methods:

```bash
Rscript satelliteDrag_comparison.R
```

### Step 4: Visualization and Analysis

Run the visualization notebook to analyze the benchmark results:

```bash
jupyter notebook benchmarkSummaryAndVisu.ipynb
```

This notebook loads results from `./log/satellite/` and generates comparative visualizations saved to the `fig/` directory.

## Output Files

- Parameter estimation results: `./log/satellite/{species}/theta_numPointsTotal*_numBlocksTotal*_m*_seed*_isScaled1_{species}.csv`
- Prediction results: `./log/satellite/{species}/*_{species}_pred.csv`

## Notes

- The estimation and prediction will takes around 8 hours in total