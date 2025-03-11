#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=batch
#SBATCH -J benchmark_data_vbatched
#SBATCH -o benchmark_data_vbatched.%J.out
#SBATCH -e benchmark_data_vbatched.%J.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200G # try larger memory

# make clean && make -j

N=1800000
N_TEST=200000
BlockCount=(20000)
BlockCount_TEST=(40000)
NN_est=(100 200 400)
NN_pred=(400 600)
DIM=8
kernel_type=Matern72
maxeval=(500 1000 1000)

# export OMP_DISPLAY_AFFINITY=TRUE

SPECIES=("O2" "N2" "H" "N" "O" "He") # 

for species in ${SPECIES[@]}
do
    for fold in {1..10}
    do
        train_metadata_path="./HST/${species}/hst${species}_fold${fold}_train.csv"
        test_metadata_path="./HST/${species}/hst${species}_fold${fold}_val.csv"
        theta_init="1,0"
        distance_scale_init="0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2"
        distance_scale="0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2"
        # theta_init="1.96492,0."
        # distance_scale="0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2"
        # distance_scale_init="2,2,2,0.814359,1.21417,0.0490129,0.0852691,0.097571"

        for bc_est in ${BlockCount[@]}
        do
            # Add an index to track position in NN_est array
            index=0
            for nn_est in ${NN_est[@]}
            do
                echo "nn_est: $nn_est"
                # Get corresponding maxeval value using index
                current_maxeval=${maxeval[$index]}
                for bc_pred in ${BlockCount_TEST[@]}
                do
                    echo "bc_pred: $bc_pred"
                    for nn_pred in ${NN_pred[@]}
                    do
                        echo "nn_pred: $nn_pred"
                        # Skip first iteration, run others
                        if [ $index -ne -1 ]; then
                            # Update maxeval parameter to use current_maxeval
                            srun --cpus-per-task=40 ./bin/dbv --num_total_points "$N" \
                            --num_total_points_test "$N_TEST" \
                            --num_total_blocks "$bc_est" \
                            --num_total_blocks_test "$bc_pred" \
                            -m "$nn_est" \
                            --m_test "$nn_pred" \
                            --omp_num_threads 40 \
                            --theta_init "$theta_init" \
                            --distance_scale "$distance_scale" \
                            --distance_scale_init "$distance_scale_init" \
                            --xtol_rel 1e-3 \
                            --ftol_rel 1e-5 \
                            --dim "$DIM" \
                            --mode prediction \
                            --maxeval "$current_maxeval" \
                            --train_metadata_path "$train_metadata_path" \
                            --test_metadata_path "$test_metadata_path" \
                            --kernel_type "$kernel_type"\
                            --seed "$fold"\
                            --nn_multiplier 500
                        fi
                        current_maxeval=1
                    done
                done    
    params_path="./log/theta_numPointsTotal1800000_numBlocksTotal${bc_est}_m${nn_est}_seed${fold}_isScaled1_.csv"

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
                echo "species: $species"
                echo "fold: $fold"
                # Increment index for next iteration
                ((index++))
            done
        done
    done
    mkdir -p ./log/benchmark/$species
    mv ./log/*.csv ./log/benchmark/$species
done