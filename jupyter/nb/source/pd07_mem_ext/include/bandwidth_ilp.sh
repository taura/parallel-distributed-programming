#!/bin/bash
set -e
export OMP_TARGET_OFFLOAD=DISABLED
#export OMP_TARGET_OFFLOAD=MANDATORY
export OMP_NUM_TEAMS=1
export OMP_NUM_THREADS=1
m=$((1 << 23))
n=$((1 << 26))

for C in 1 2 3 4 6 8 10 12 14 16 20 24 28 32; do
    echo "==== C=${C} ===="
    echo ./exe/latency_clang++_omp_ilp.exe --n-elements ${m} --min-accesses ${n} --n-cycles ${C} --n-conc-cycles ${C}
done
      
