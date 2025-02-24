#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=batch
#SBATCH -J benchmark_increasing_cs
#SBATCH -o benchmark_increasing_cs.%J.out
#SBATCH -e benchmark_increasing_cs.%J.err
#SBATCH --time=7:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200G # try larger memory


make clean && make -j

N=1800000
N_TEST=200000
BlockCount=(20000)
BlockCount_TEST=(40000)
NN_est=(200) # 300 400
NN_pred=(200)
DIM=8
kernel_type=Matern72

DATA_DIR="./log"
SPECIES=("O2" "N2" "H" "N" "O" "He") # 

for species in ${SPECIES[@]}
do
    echo "species: $species"
    for fold in {1..10}
    do
        train_metadata_path="./HST/${species}/hst${species}_fold${fold}_train.csv"
        test_metadata_path="./HST/${species}/hst${species}_fold${fold}_val.csv"
        echo "fold: $fold"
        for bc_est in ${BlockCount[@]}
        do
            for nn_est in ${NN_est[@]}
            do
                params_path="$DATA_DIR/${species}/theta_numPointsTotal1800000_numBlocksPerProcess${bc_est}_m${nn_est}_seed${fold}_isScaled1.csv"
                
                # Read the last line of the CSV file (2nd data line)
                line_scaled=$(tail -n 2 $params_path)
                line_estimated=$(tail -n 1 $params_path)

                # Extract the first two values
                theta_scaled=$(echo "$line_scaled" | cut -d',' -f1-2)
                theta_estimated=$(echo "$line_estimated" | cut -d',' -f1-2)

                # Extract the rest of the values
                distance_scaled=$(echo "$line_scaled" | cut -d',' -f3-)
                distance_estimated=$(echo "$line_estimated" | cut -d',' -f3-)
                echo "theta_scaled: $theta_scaled"
                echo "theta_estimated: $theta_estimated"
                echo "distance_scaled: $distance_scaled"
                echo "distance_estimated: $distance_estimated"

                for bc_pred in ${BlockCount_TEST[@]}
                do
                    echo "bc_pred: $bc_pred"
                    for nn_pred in ${NN_pred[@]}
                    do
                        echo "nn_pred: $nn_pred"
                        ./bin/dbv --num_total_points "$N" \
                        --num_total_points_test "$N_TEST" \
                        --num_total_blocks "$bc_est" \
                        --num_total_blocks_test "$bc_pred" \
                        -m "$nn_est" \
                        --m_test "$nn_pred" \
                        --omp_num_threads 40 \
                        --theta_init "$theta_estimated" \
                        --distance_scale "$distance_scaled" \
                        --distance_scale_init "$distance_estimated" \
                        --dim "$DIM" \
                        --mode prediction \
                        --maxeval 1 \
                        --train_metadata_path "$train_metadata_path" \
                        --test_metadata_path "$test_metadata_path" \
                        --kernel_type "$kernel_type"\
                        --seed "$fold" \
                        --log_append "${species}_increasing_cs"
                    done
                done    
            done
        done
    done
    mkdir -p $DATA_DIR/"${species}_increasing_cs"
    mv $DATA_DIR/"*_${species}_increasing_cs.csv" $DATA_DIR/"${species}_increasing_cs"
done