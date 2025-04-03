#!/bin/bash -x
#SBATCH --account=rcfd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=A100_single1.%j
#SBATCH --error=A100_single1-err.%j
#SBATCH --time=10:00:00 
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:1

N_all=(10000 30000 70000 100000 300000 500000 700000)
N_bs=(100)
M_ests=(100 200 400)
nn_multipliers=(300 300 500)

DIM=10
theta_init=1.0,0.001
distance_scale=0.05,0.01,0.05,5.0,5.0,5.0,5.0,5.0,5.0,5.0
distance_scale_init=$distance_scale

for index in {0..6}; do
    N=${N_all[$index]}
    # Scaled block Vecchia
    for index_est in {0..2}; do
        m_bv=${M_ests[$index_est]}
        nn_multiplier=${nn_multipliers[$index_est]}
        for N_b in ${N_bs[@]}; do
            for i in {1..3}; do
                bc=$((N/N_b))
                echo "N: $N, bc: $bc, m_bv: $m_bv, seed: $i, nn_multiplier: $nn_multiplier"
                if [ \( $N -le 2000000 -a $m_bv -eq 400 \) -o \( $N -le 4000000 -a $m_bv -eq 200 \) -o \( $N -le 7000000 -a $m_bv -eq 100 \) ]; then
                    ./bin/dbv \
                        --num_total_points $N \
                        --num_total_blocks $bc \
                        --distance_scale $distance_scale \
                        --distance_scale_init $distance_scale_init \
                        --theta_init $theta_init \
                        -m $m_bv \
                        --dim $DIM \
                        --mode estimation \
                        --maxeval 500 \
                        --xtol_rel 1e-8 \
                        --ftol_rel 1e-8 \
                        --kernel_type Matern72 \
                        --seed $i \
                        --nn_multiplier $nn_multiplier \
                        --log_append A100_single \
                        --omp_num_threads 32 \
                        --print=false
                fi
            done
        done
    done

    # Scaled Vecchia
    if [ $N -le 500000 ]; then # more than 500k takes a lot of time, not worth it, just show a trend
        for i in {1..3}; do
           ./bin/dbv \
                --num_total_points $N \
                --num_total_blocks $N \
                --distance_scale $distance_scale \
                --distance_scale_init $distance_scale_init \
                --theta_init $theta_init \
                -m 60 \
                --dim $DIM \
                --mode estimation \
                --maxeval 500 \
                --xtol_rel 1e-8 \
                --ftol_rel 1e-8 \
                --kernel_type Matern72 \
                --seed $i \
                --nn_multiplier 50 \
                --log_append A100_single \
                --omp_num_threads 32 \
                --print=false
        done
    fi
done

mkdir -p ./log/A100_single
mv ./log/*_A100_single.csv ./log/A100_single/
