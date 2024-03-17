#!/bin/bash
zeros=(FLAT ZERO ZEROX)

if [[ $1 == "S" ]]; then
    lZero=3
    lUct=5
elif [[ $1 == "M" ]]; then
    lZero=6
    lUct=4
elif [[ $1 == "L" ]]; then
    lZero=5
    lUct=3
fi

for ((i=1; i<$lZero+1; i++)); do
    # time-invariant, zero vs zero
    for ((zi=1; zi<${#zeros[@]}+1; zi++)); do
        for ((zj=$zi+1; zj<${#zeros[@]}+1; zj++)); do
            sbatch fit.sh eval --game_size=$1 ${zeros[$zi-1]},$i ${zeros[$zj-1]},$i
        done
    done

    # time-invariant, zero vs uct
    for zero in "${zeros[@]}"; do
        for ((j=1; j<$lUct+1; j++)); do
            sbatch fit.sh eval --game_size=$1 $zero,$i UCT,$j
        done
    done

    # time-variant, zero vs zero
    for ((j=$i+1; j<$lZero+1; j++)); do
        for zero in "${zeros[@]}"; do
            sbatch fit.sh eval --game_size=$1 $zero,$i $zero,$j
        done
    done
done



