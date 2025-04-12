# !/bin/bash

N=5000
bc=(500)
m_bv=(5 10 20 30 60 90 120 150 180 210 240 270)
DATA_DIR="./experiments/simulations/maternSimuData"
train_metadata_path="$DATA_DIR/training_data_kl.csv"
params_path="$DATA_DIR/hyperparameters.csv"
DIM=8

# read params from params_path
params=($(cat $params_path))
theta_init=$(IFS=,; echo "${params[0]},${params[1]}")
distance_scale=$(IFS=,; echo "${params[@]:2:$DIM}" | tr ' ' ',')
distance_scale_init=$distance_scale
nodistance_scale=$(printf '1%.0s,' {1..8} | sed 's/,$//')
nodistance_scale_init=$distance_scale

for b in ${bc[@]}; do
    for m in ${m_bv[@]}; do

        ./bin/dbv --num_total_points $N \
        --num_total_blocks $b \
        -m $m \
        --dim $DIM \
        --mode estimation \
        --maxeval 1 \
        --theta_init $theta_init \
        --distance_scale $distance_scale \
        --distance_scale_init $distance_scale \
        --train_metadata_path $train_metadata_path \
        --kernel_type Matern72 \
        --nn_multiplier 999999 \
        --log_append kl-matern72-simu
        # no scaled distance
        ./bin/dbv --num_total_points $N \
        --num_total_blocks $b \
        -m $m \
        --dim $DIM \
        --mode estimation \
        --maxeval 1 \
        --theta_init $theta_init \
        --distance_scale $nodistance_scale \
        --distance_scale_init $nodistance_scale_init \
        --train_metadata_path $train_metadata_path \
        --kernel_type Matern72 \
        --nn_multiplier 999999 \
        --log_append kl-matern72-simu
    done
done


# classic Vecchia
N=5000
bc=(5000)
m_bv=(5 10 20 30 40 50 60 70 80 90 100)
DATA_DIR="./experiments/simulations/maternSimuData"
train_metadata_path="$DATA_DIR/training_data_kl.csv"
params_path="$DATA_DIR/hyperparameters.csv"
THRESHOLD=999
DIM=8

# read params from params_path
params=($(cat $params_path))
theta_init=$(IFS=,; echo "${params[0]},${params[1]}")
distance_scale=$(IFS=,; echo "${params[@]:2:$DIM}" | tr ' ' ',')
distance_scale_init=$distance_scale
nodistance_scale=$(printf '1%.0s,' {1..8} | sed 's/,$//')
nodistance_scale_init=$distance_scale

for b in ${bc[@]}; do
    for m in ${m_bv[@]}; do
        # scaled distance
        ./bin/dbv --num_total_points $N \
        --num_total_blocks $b \
        -m $m \
        --dim $DIM \
        --mode estimation \
        --maxeval 1 \
        --theta_init $theta_init \
        --distance_scale $distance_scale \
        --distance_scale_init $distance_scale_init \
        --train_metadata_path $train_metadata_path \
        --nn_multiplier 999999 \
        --kernel_type Matern72 \
        --log_append kl-matern72-simu
        # no scaled distance
        ./bin/dbv --num_total_points $N \
        --num_total_blocks $b \
        -m $m \
        --dim $DIM \
        --mode estimation \
        --maxeval 1 \
        --theta_init $theta_init \
        --distance_scale $nodistance_scale \
        --distance_scale_init $nodistance_scale_init \
        --train_metadata_path $train_metadata_path \
        --nn_multiplier 999999 \
        --kernel_type Matern72 \
        --log_append kl-matern72-simu
    done
done

mkdir -p ./log/kl-matern72-simu
mv ./log/*.csv ./log/kl-matern72-simu/