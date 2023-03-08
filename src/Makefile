# Makefile for the High Performance Computing programming project,
# Academic Year 2022/2023.
#
# Available targets:
#
# - sph: builds serial version
#
# - omp-sph: builds OpenMP version
#
# - mpi-sph: builds MPI version
#
# - all: builds all three versions (default)
#
# - clean: clean up
#
# Last modified on 2023-03-08 by Alessandro Sciarrillo

EXE:=sph omp-sph mpi-sph
CFLAGS+=-std=c99 -Wall -Wpedantic
LDLIBS=-lm

.PHONY: clean

all: $(EXE) hpc.h

sph: sph.c hpc.h
	$(CC) $(CFLAGS) $< $(LDLIBS) -o $@

omp-sph: omp-sph.c hpc.h
	$(CC) $(CFLAGS) -fopenmp $< $(LDLIBS) -o $@

mpi-sph: mpi-sph.c hpc.h
	mpicc $(CFLAGS) $< $(LDLIBS) -o $@

clean:
	\rm -f $(EXE) *.o *~