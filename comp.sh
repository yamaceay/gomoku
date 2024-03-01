#!/bin/bash
lZero=3
lUct=5
zeros=(FLAT ZERO ZEROX)

for ((i=1; i<$lZero+1; i++)); do
    # time-invariant
    for zero in "${zeros[@]}"; do
        for ((j=1; j<$lUct+1; j++)); do
            sbatch fit.sh eval $zero,$i UCT,$j
        done
    done

    # time-variant
    for ((j=$i+1; j<$lZero+1; j++)); do
        for zero in "${zeros[@]}"; do
            sbatch fit.sh eval $zero,$i $zero,$j
        done
    done
done


