#!/bin/bash
#SBATCH -N 3
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH -J RealDataset_full_estimation
#SBATCH -o RealDataset_full_estimation.%J.out
#SBATCH -e RealDataset_full_estimation.%J.err
#SBATCH --time=4:00:00
#SBATCH -A jureap137

N=45000000
N_sub=1000000
N_sub_bc=$((N_sub/100))
N_sub_bc_est=200
seed=42
# N_TEST=100000
BlockCount=(450000)
# BlockCount_TEST=(20000)
NN_est=(100 200 400)
# NN_pred=(200 400 600)
DIM=10
kernel_type=Matern72
maxeval=(1000 1000 1000)

DATA_DIR="./log"

train_metadata_path="/p/fscratch/jureap137/data/train_combined.txt"
test_metadata_path="/p/fscratch/jureap137/data/test_combined.txt"

params_path="$DATA_DIR/RealDataset/theta_numPointsTotal${N_sub}_numBlocksTotal${N_sub_bc}_m${N_sub_bc_est}_seed${seed}_isScaled1_RealDataset.csv"
echo "params_path: $params_path"
# Read the first line of the CSV file
line=$(head -n 1 $params_path)

# Extract the first two values
theta_init=$(echo "$line" | cut -d',' -f1-2)

# Extract the rest of the values
distance_scale=$(echo "$line" | cut -d',' -f3-)
distance_scale_init=$distance_scale
echo "theta_init: $theta_init"
echo "distance_scale: $distance_scale"
echo "distance_scale_init: $distance_scale_init"

for bc_est in ${BlockCount[@]}
do
    for index in $(seq 0 2); do
        nn_est=${NN_est[$index]} 
        current_maxeval=${maxeval[$index]}
        echo "bc_est: $bc_est, nn_est: $nn_est, current_maxeval: $current_maxeval"

        srun --exclusive ./bin/dbv --num_total_points "$N" \
            --num_total_blocks "$bc_est" \
            -m "$nn_est" \
            --omp_num_threads 72 \
            --theta_init "$theta_init" \
            --distance_scale "$distance_scale" \
            --distance_scale_init "$distance_scale_init" \
            --dim "$DIM" \
            --mode estimation \
            --xtol_rel 1e-3 \
            --ftol_rel 1e-5 \
            --maxeval "$current_maxeval" \
            --train_metadata_path "$train_metadata_path" \
            --test_metadata_path "$test_metadata_path" \
            --kernel_type "$kernel_type"\
            --seed "$seed" \
            --log_append RealDataset
            params_path="$DATA_DIR/theta_numPointsTotal${N}_numBlocksTotal${bc_est}_m${nn_est}_seed${seed}_isScaled1_RealDataset.csv"
            echo "params_path: $params_path"

            # Read the first line of the CSV file
            line=$(head -n 1 $params_path)

            # Extract the first two values
            theta_init=$(echo "$line" | cut -d',' -f1-2)

            # Extract the rest of the values
            distance_scale=$(echo "$line" | cut -d',' -f3-)
            distance_scale_init=$distance_scale
            echo "theta_init: $theta_init"
            echo "distance_scale: $distance_scale"
            echo "distance_scale_init: $distance_scale_init"
    done
done

mkdir -p ./log/RealDataset
mv ./log/*RealDataset.csv ./log/RealDataset/
