#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -J scaling_GH200_1
#SBATCH -o scaling_GH200_1.%J.out
#SBATCH -e scaling_GH200_1.%J.err
#SBATCH --time=0:30:00
#SBATCH -A jureap137

N_base_strong=(5000000 5000000 5000000) # larger problem BSV 100/400 GH200
M_ests=(100 200 400)
nn_multipliers=(300 300 500)
N_bs=(100 100 100)
num_runs=1
DIM=10
theta_init=1.0,0.001
distance_scale=0.05,0.01,0.05,5.0,5.0,5.0,5.0,5.0,5.0,5.0
distance_scale_init=$distance_scale

# base scaling
for index in {0..1}; do
    # Scaled block Vecchia
    N=${N_base_strong[$index]}
    m_bv=${M_ests[$index]}
    nn_multiplier=${nn_multipliers[$index]}
    N_b=${N_bs[$index]}
    for i in $(seq 1 $num_runs); do
        echo "Running $i"
        bc=$((N/N_b))
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
                --nn_multiplier $nn_multiplier \
                --log_append GH200_scaling_base\
                --omp_num_threads 72 \
                --print=false
        done
    done
done

mkdir -p ./log/GH200_scaling
mv ./log/*_GH200_scaling_base.csv ./log/GH200_scaling/
