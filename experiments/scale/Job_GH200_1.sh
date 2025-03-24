#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -J scaling_GH200_1
#SBATCH -o scaling_GH200_1.%J.out
#SBATCH -e scaling_GH200_1.%J.err
#SBATCH --time=1:20:00
#SBATCH -A jureap137

N_base_strong=(5000000) # larger problem BSV 100/400 GH200

M_ests=(400)
N_bs=(100)

DIM=10
theta_init=1.0,0.001
distance_scale=0.05,0.01,0.05,5.0,5.0,5.0,5.0,5.0,5.0,5.0
distance_scale_init=$distance_scale

# base scaling
for N in ${N_base_strong[@]}; do
    # Scaled block Vecchia
    for M_est in ${M_ests[@]}; do
        for N_b in ${N_bs[@]}; do
            for i in {1..3}; do
                bc=$((N/N_b))
                m_bv=$M_est
                echo "N: $N, bc: $bc, m_bv: $m_bv"
                srun --exclusive ./bin/dbv \
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
                    --nn_multiplier 500 \
                    --log_append GH200_scaling_base\
                    --omp_num_threads 72 \
                    # --print=false
            done
        done
    done
done

mkdir -p ./log/GH200_scaling
mv ./log/*_GH200_scaling_base.csv ./log/GH200_scaling/
