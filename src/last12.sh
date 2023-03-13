#!/bin/bash

echo "----[ 10k 50 ]"
echo "--------[ OpenMP ]"
OMP_NUM_THREADS=12 ./omp-sph 10000 50
echo "--------[ MPI ]"
mpirun -n 12 ./mpi-sph 10000 50
echo -e " "
echo "----[ 17320 20 ]"
echo "--------[ OpenMP ]"
OMP_NUM_THREADS=12 ./omp-sph 17320 20
echo "--------[ MPI ]"
mpirun -n 12 ./mpi-sph 17320 20