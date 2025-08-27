# !/bin/bash
range=(0.026270 0.017512 0.014290 0.078809 0.052537 0.042869 0.210158 0.140098 0.114318)
nu=(0.5 1.5 2.5 0.5 1.5 2.5 0.5 1.5 2.5)
n_array=(1 2 3 4 5 6 7 8 9)
precision=(float double)

N=40000
bc=(400)
m_bv=(5 10 20 30 60 90 120 150 180 210 240 270)

DIM=(1 2 4 8)

for p in ${precision[@]}; do
    for n in ${n_array[@]}; do 
        for b in ${bc[@]}; do
            for m in ${m_bv[@]}; do  
                for d in ${DIM[@]}; do
                    range_init="${range[$((n-1))]}"
                    nu_init="${nu[$((n-1))]}"
                    if (( $(echo "$nu_init == 0.5" | bc -l) )); then
                        kernel_type="Matern12"
                    elif (( $(echo "$nu_init == 1.5" | bc -l) )); then
                        kernel_type="Matern32"
                    elif (( $(echo "$nu_init == 2.5" | bc -l) )); then
                        kernel_type="Matern52"
                    fi
                    _distance_scale=$(printf "%s," $(yes $range_init | head -n $d) | sed 's/,$//')
                    ./bin/dbv --num_total_points $N \
                    --num_total_blocks $b \
                    -m $m \
                    --dim $d \
                    --mode estimation \
                    --maxeval 1 \
                    --theta_init 1.0,0.0 \
                    --distance_scale $_distance_scale \
                    --distance_scale_init $_distance_scale \
                    --kernel_type $kernel_type \
                    --nn_multiplier 999999 \
                    --log_append "kl-$kernel_type-simu-$p-$n-dim$d" \
                    --precision $p
                done
            done
        done
    done
done

mkdir -p ./log/kl-matern-simu
mv ./log/*.csv ./log/kl-matern-simu