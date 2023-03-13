#!/bin/bash

echo "----[ BENCHMARKS ]" > output.txt ;


for i in {1..12}
do
    OMP_NUM_THREADS=$i ./omp-sph 5000 20 2>&1 | tee -a output.txt ;
    #mpirun -n $i ./mpi-sph 5000 20 2>&1 | tee -a output.txt ;
done

