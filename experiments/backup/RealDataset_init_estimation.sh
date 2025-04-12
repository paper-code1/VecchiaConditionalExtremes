#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH -J RealDataset_init_estimation
#SBATCH -o RealDataset_init_estimation.%J.out
#SBATCH -e RealDataset_init_estimation.%J.err
#SBATCH --time=1:00:00
#SBATCH -A jureap137

N=45000000
N_sub=1000000
N_sub_bc=$((N_sub/100))
N_sub_bc_ests=(100 200)
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

theta_init="1,0"
distance_scale_init="0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2"
distance_scale="0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2"

echo "theta_init: $theta_init"
echo "distance_scale: $distance_scale"
echo "distance_scale_init: $distance_scale_init"

for N_sub_bc_est in ${N_sub_bc_ests[@]}
do
    srun --exclusive ./bin/dbv --num_total_points "$N_sub" \
        --num_total_blocks "$N_sub_bc" \
        -m "$N_sub_bc_est" \
        --omp_num_threads 72 \
        --theta_init "$theta_init" \
        --distance_scale "$distance_scale" \
        --distance_scale_init "$distance_scale_init" \
        --dim "$DIM" \
        --mode estimation \
        --xtol_rel 1e-4 \
        --ftol_rel 1e-6 \
        --maxeval 2000 \
        --train_metadata_path "$train_metadata_path" \
        --test_metadata_path "$test_metadata_path" \
        --kernel_type "$kernel_type"\
        --nn_multiplier 600 \
        --seed "$seed" \
        --log_append RealDataset

    # Read the first line of the CSV file
    params_path="$DATA_DIR/theta_numPointsTotal${N_sub}_numBlocksTotal${N_sub_bc}_m${N_sub_bc_est}_seed${seed}_isScaled1_RealDataset.csv"
    line=$(head -n 1 $params_path)

    # Extract the first two values
    theta_init=$(echo "$line" | cut -d',' -f1-2)
    # Extract the rest of the values
    distance_scale_init=$(echo "$line" | cut -d',' -f3-)
    distance_scale=$distance_scale_init
done
        
mkdir -p ./log/RealDataset
mv ./log/*RealDataset.csv ./log/RealDataset/
