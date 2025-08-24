# !/bin/bash

N=5000
N_TEST=2000
bc=500
bc_test=(200)
m_bv=1 # no meaning here for prediction, we plug in the true theta
m_bv_test=(5 10 20 30 50 70 90 110 130 150 170 190 210 230 250 270 290 310 330 350 370 390 410 430 450 470 490 500)
# m_bv_test=(90)
precision=(float double)

DATA_DIR="./experiments/simulations/maternSimuData"
train_metadata_path="$DATA_DIR/training_data.csv"
test_metadata_path="$DATA_DIR/test_data.csv"
params_path="$DATA_DIR/hyperparameters.csv"
DIM=8

# read params from params_path
params=($(cat $params_path))
theta_init=$(IFS=,; echo "${params[0]},${params[1]}")
distance_scale=$(IFS=,; echo "${params[@]:2:$DIM}" | tr ' ' ',')
distance_scale_init=$distance_scale
nodistance_scale=$(printf '1%.0s,' {1..8} | sed 's/,$//')
nodistance_scale_init=$distance_scale

for B_TEST in ${bc_test[@]}; do
    for M_TEST in ${m_bv_test[@]}; do
        # scale the distance
        for p in ${precision[@]}; do
            ./bin/dbv --num_total_points $N --num_total_points_test $N_TEST --num_total_blocks $bc --num_total_blocks_test $B_TEST -m $m_bv --m_test $M_TEST --dim $DIM --mode prediction --maxeval 1 --theta_init $theta_init --distance_scale $distance_scale --distance_scale_init $distance_scale_init --train_metadata_path $train_metadata_path --test_metadata_path $test_metadata_path --nn_multiplier 99999 --log_append mspe-matern72-simu-$p --kernel_type Matern72 --precision $p
        done
    done
done


mkdir -p ./log/mspe-matern72-simu
mv ./log/*.csv ./log/mspe-matern72-simu/