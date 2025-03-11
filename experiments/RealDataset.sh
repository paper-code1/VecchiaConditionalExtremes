#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=batch
#SBATCH -J RealDataset
#SBATCH -o RealDataset.%J.out
#SBATCH -e RealDataset.%J.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200G # try larger memory

make clean && make -j

N=900000
N_TEST=100000
BlockCount=(10000)
BlockCount_TEST=(20000)
NN_est=(100 200 300 400)
NN_pred=(200 400 600)
DIM=10
kernel_type=Matern72
maxeval=(1000 2000 1000 1000)
fold=1

DATA_DIR="./log"

train_metadata_path="./metaRVMdata/train_data.txt"
test_metadata_path="./metaRVMdata/test_data.txt"
theta_init="1,0"
distance_scale_init="0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2"
distance_scale="0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2"

for bc_est in ${BlockCount[@]}
do
    # Add an index to track position in NN_est array
    index=0
    for nn_est in ${NN_est[@]}
    do
        echo "nn_est: $nn_est"
        # Get corresponding maxeval value using index
        current_maxeval=${maxeval[$index]}
        echo "current_maxeval: $current_maxeval"
        for bc_pred in ${BlockCount_TEST[@]}
        do
            echo "bc_pred: $bc_pred"
            for nn_pred in ${NN_pred[@]}
            do
                echo "nn_pred: $nn_pred"
                # Skip first iteration, run others
                if [ $index -ne -1 ]; then
                    # Update maxeval parameter to use current_maxeval
                    ./bin/dbv --num_total_points "$N" \
                    --num_total_points_test "$N_TEST" \
                    --num_total_blocks "$bc_est" \
                    --num_total_blocks_test "$bc_pred" \
                    -m "$nn_est" \
                    --m_test "$nn_pred" \
                    --omp_num_threads 40 \
                    --theta_init "$theta_init" \
                    --distance_scale "$distance_scale" \
                    --distance_scale_init "$distance_scale_init" \
                    --dim "$DIM" \
                    --mode prediction \
                    --maxeval "$current_maxeval" \
                    --train_metadata_path "$train_metadata_path" \
                    --test_metadata_path "$test_metadata_path" \
                    --kernel_type "$kernel_type"\
                    --seed "$fold" \
                    --log_append RealDataset
                fi
                    current_maxeval=1    params_path="$DATA_DIR/theta_numPointsTotal${N}_numBlocksTotal${bc_est}_m${nn_est}_seed${fold}_isScaled1_RealDataset.csv"

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
                echo "fold: $fold"
            done
        done
        # Increment index for next iteration
        ((index++))
    done
done

mkdir -p ./log/RealDataset
mv ./log/*RealDataset.csv ./log/RealDataset/
