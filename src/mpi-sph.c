/****************************************************************************
 *
 * [ Alessandro Sciarrillo  ]
 * [ 0000970435             ]
 * 
 * sph.c -- Smoothed Particle Hydrodynamics
 *
 * https://github.com/cerrno/mueller-sph
 *
 * Copyright (C) 2016 Lucas V. Schuermann
 * Copyright (C) 2022 Moreno Marzolla
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 ****************************************************************************/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* "Particle-Based Fluid Simulation for Interactive Applications" by
   Müller et al. solver parameters */

const float Gx = 0.0, Gy = -10.0;   // external (gravitational) forces
const float REST_DENS = 300;    // rest density
const float GAS_CONST = 2000;   // const for equation of state
const float H = 16;             // kernel radius
const float EPS = 16;           // equal to H
const float MASS = 2.5;         // assume all particles have the same mass
const float VISC = 200;         // viscosity constant
const float DT = 0.0007;        // integration timestep
const float BOUND_DAMPING = -0.5;

const int MAX_PARTICLES = 20000;

// Larger window size to accommodate more particles
#define WINDOW_WIDTH 3000
#define WINDOW_HEIGHT 2000
const float VIEW_WIDTH = 1.5 * WINDOW_WIDTH;
const float VIEW_HEIGHT = 1.5 * WINDOW_HEIGHT;

const int DAM_PARTICLES = 500;

/**
 * Particles data structure (SoA); stores arrays of position, velocity, and force for
 * integration stores density (rho) and pressure values for SPH.
 */
typedef struct {
    float *x, *y;         // position
    float *vx, *vy;       // velocity
    float *fx, *fy;       // force
    float *rho, *p;       // density, pressure
} particles_t;

int n_particles = 0;    // number of currently active particles

int my_rank, comm_sz;
int my_start, my_end;
int local_n;
int *recvcounts, *displs;

/**
 * Return a random value in [a, b]
 */
float randab(float a, float b)
{
    return a + (b-a)*rand() / (float)(RAND_MAX);
}

/**
 * Set the initial position of the particle at index 'index'
 * in the structure of '*p' to (x, y); initialize all
 * other attributes to default values (zeros).
 */
void init_particle( particles_t *p, int index, float x, float y )
{
    p->x[index] = x;
    p->y[index] = y;
    p->vx[index] = p->vy[index] = 0.0;
    p->fx[index] = p->fy[index] = 0.0;
    p->rho[index] = 0.0;
    p->p[index] = 0.0;
}

/**
 * Return nonzero iff (x, y) is within the frame
 */
int is_in_domain( float x, float y )
{
    return ((x < VIEW_WIDTH - EPS) &&
            (x > EPS) &&
            (y < VIEW_HEIGHT - EPS) &&
            (y > EPS));
}

/**
 * Initialize the SPH model with `n` particles. The caller is
 * responsible for allocating an array for each variable in
 * the particles structure of size `MAX_PARTICLES`.
 */
void init_sph( int n, particles_t *particles )
{
    n_particles = 0;
    printf("Initializing with %d particles\n", n);

    for (float y = EPS; y < VIEW_HEIGHT - EPS; y += H) {
        for (float x = EPS; x <= VIEW_WIDTH * 0.8f; x += H) {
            if (n_particles < n) {
                float jitter = rand() / (float)RAND_MAX;
                init_particle(particles, n_particles, x+jitter, y);
                n_particles++;
            } else {
                return;
            }
        }
    }
    assert(n_particles == n);
}

void compute_density_pressure( particles_t *p )
{
    const float HSQ = H * H;    // radius^2 for optimization

    /* Smoothing kernels defined in Muller and their gradients adapted
       to 2D per "SPH Based Shallow Water Simulation" by Solenthaler
       et al. */ 
    const float POLY6 = 4.0 / (M_PI * pow(H, 8)); 

    for (int i=my_start; i<my_end; i++) {
        p->rho[i] = 0.0;
        for (int j=0; j<n_particles; j++) {
            const float dx = p->x[j] - p->x[i];
            const float dy = p->y[j] - p->y[i];
            const float d2 = dx*dx + dy*dy;

            if (d2 < HSQ) {
                p->rho[i] += MASS * POLY6 * pow(HSQ - d2, 3.0); 
            }
        }
        p->p[i] = GAS_CONST * (p->rho[i] - REST_DENS);
    }
}

void compute_forces( particles_t *p )
{
    /* Smoothing kernels defined in Muller and their gradients adapted
       to 2D per "SPH Based Shallow Water Simulation" by Solenthaler
       et al. */
    const float SPIKY_GRAD = -10.0 / (M_PI * pow(H, 5));
    const float VISC_LAP = 40.0 / (M_PI * pow(H, 5));
    const float EPS = 1e-6;

    for (int i=my_start; i<my_end; i++) {
        float fpress_x = 0.0, fpress_y = 0.0;
        float fvisc_x = 0.0, fvisc_y = 0.0;

        for (int j=0; j<n_particles; j++) {
            if (i == j)
                continue;

            const float dx = p->x[j] - p->x[i];
            const float dy = p->y[j] - p->y[i];
            const float dist = hypotf(dx, dy) + EPS; // avoids division by zero later on

            if (dist < H) {
                const float norm_dx = dx / dist;
                const float norm_dy = dy / dist;
                // compute pressure force contribution
                fpress_x += -norm_dx * MASS * (p->p[i] + p->p[j]) / (2 * p->rho[j]) * SPIKY_GRAD * pow(H - dist, 3);
                fpress_y += -norm_dy * MASS * (p->p[i] + p->p[j]) / (2 * p->rho[j]) * SPIKY_GRAD * pow(H - dist, 3);
                // compute viscosity force contribution
                fvisc_x += VISC * MASS * (p->vx[j] - p->vx[i]) / p->rho[j] * VISC_LAP * (H - dist);
                fvisc_y += VISC * MASS * (p->vy[j] - p->vy[i]) / p->rho[j] * VISC_LAP * (H - dist);
            }
        }
        const float fgrav_x = Gx * MASS / p->rho[i];
        const float fgrav_y = Gy * MASS / p->rho[i];
        p->fx[i] = fpress_x + fvisc_x + fgrav_x;
        p->fy[i] = fpress_y + fvisc_y + fgrav_y;
    }
}

void integrate( particles_t *p )
{   
    for (int i=my_start; i<my_end; i++) {
        // forward Euler integration
        p->vx[i] += DT * p->fx[i] / p->rho[i];
        p->vy[i] += DT * p->fy[i] / p->rho[i];
        p->x[i] += DT * p->vx[i];
        p->y[i] += DT * p->vy[i];

        // enforce boundary conditions
        if (p->x[i] - EPS < 0.0) {
            p->vx[i] *= BOUND_DAMPING;
            p->x[i] = EPS;
        }
        if (p->x[i] + EPS > VIEW_WIDTH) {
            p->vx[i] *= BOUND_DAMPING;
            p->x[i] = VIEW_WIDTH - EPS;
        }
        if (p->y[i] - EPS < 0.0) {
            p->vy[i] *= BOUND_DAMPING;
            p->y[i] = EPS;
        }
        if (p->y[i] + EPS > VIEW_HEIGHT) {
            p->vy[i] *= BOUND_DAMPING;
            p->y[i] = VIEW_HEIGHT - EPS;
        }
    }
}

float avg_velocities( particles_t *p )
{
    double result, my_result = 0.0;
    for (int i=my_start; i<my_end; i++) {
        /* the hypot(x,y) function is equivalent to sqrt(x*x +
           y*y); */
        my_result += hypot(p->vx[i], p->vy[i]) / n_particles;
    }

    MPI_Reduce(
        &my_result,     //const void *sendbuf
        &result,        //void *recvbuf
        1,              //int count
        MPI_DOUBLE,     //MPI_Datatype datatype
        MPI_SUM,        //MPI_Op op
        0,              //int root
        MPI_COMM_WORLD  //MPI_Comm comm
        );

    return result;
}

/**
 * Call the MPI_Allgather() function to update the
 * input array across all processes compounding
 * the partial work done by each.
 */
void allgatherv_pv_array( float *pv_array)
{
    MPI_Allgatherv(
        &(pv_array[my_start]),  //const void *sendbuf
        local_n,                //int sendcount
        MPI_FLOAT,              //MPI_Datatype sendtype
        pv_array,               //void *recvbuf
        recvcounts,             //const int *recvcounts
        displs,                 //const int *displs
        MPI_FLOAT,              //MPI_Datatype recvtype
        MPI_COMM_WORLD          //MPI_Comm comm
        );
}


void update( particles_t *particles )
{
    compute_density_pressure( particles ); 

    allgatherv_pv_array( particles->rho );
    allgatherv_pv_array( particles->p );

    compute_forces( particles );

    allgatherv_pv_array( particles->fx );
    allgatherv_pv_array( particles->fy );

    integrate( particles );

    allgatherv_pv_array( particles->x );
    allgatherv_pv_array( particles->y );
    allgatherv_pv_array( particles->vx );
    allgatherv_pv_array( particles->vy );
}

/**
 * Broadcast an array of float of length 'n_particles'
 * from process 0 to all processes using MPI_Bcast().
 */
void bcast_array_of_float( float *bc_array)
{
    MPI_Bcast(
        bc_array, 
        n_particles, 
        MPI_FLOAT,
        0, 
        MPI_COMM_WORLD ); 
}

/**
 * Pass the x,y,vx,vy values of the particles
 * from process 0 to all processes using MPI_Bcast().
 */
void bcast_initial_values(particles_t *particles)
{
    bcast_array_of_float(particles->x);
    bcast_array_of_float(particles->y);
    bcast_array_of_float(particles->vx);
    bcast_array_of_float(particles->vy);
}

/**
 * Calculate the split by number of particles between processes.
 * Assign 'local_n' the number of particles the process is
 * responsible for, assign 'my_start' and 'my_end' start and end
 * indexes to work on arrays.
 * Create respective 'recvcounts' and 'displs' for MPI functions.
 */
void compute_blocks()
{
    recvcounts = (int*)malloc(comm_sz * sizeof(*recvcounts)); assert(recvcounts != NULL);
    displs = (int*)malloc(comm_sz * sizeof(*displs)); assert(displs != NULL);
    for (int i=0; i<comm_sz; i++) {
        const int istart = n_particles*i/comm_sz;
        const int iend = n_particles*(i+1)/comm_sz;
        const int blklen = iend - istart;
        recvcounts[i] = blklen;
        displs[i] = istart;
    }
    local_n = recvcounts[my_rank];
    my_start = displs[my_rank];
    my_end = my_start + recvcounts[my_rank];
}

/**
 * Allocates an array of float of size `MAX_PARTICLES`.
 * Return the pointer.
 */
float* alloc_maxp_length_array( void )
{
    float *a = (float*)malloc(MAX_PARTICLES * sizeof(float));
    assert( a != NULL );
    return a;
}

/**
 * Allocates an array for each variable in
 * the particles structure of size `MAX_PARTICLES`.
 */
void alloc_particles(particles_t *particles )
{
    particles->x = alloc_maxp_length_array();
    particles->y = alloc_maxp_length_array();
    particles->vx = alloc_maxp_length_array();
    particles->vy = alloc_maxp_length_array();
    particles->fx = alloc_maxp_length_array();
    particles->fy = alloc_maxp_length_array();
    particles->rho = alloc_maxp_length_array();
    particles->p = alloc_maxp_length_array();
}

/**
 * Free the memory of each array in the 
 * particles structure.
 */
void free_particles( particles_t *particles)
{
    free(particles->x);
    free(particles->y);
    free(particles->vx);
    free(particles->vy);
    free(particles->fx);
    free(particles->fy);
    free(particles->rho);
    free(particles->p);
}

int main(int argc, char **argv)
{
    srand(1234);
    particles_t particles;

    int n = DAM_PARTICLES;
    int nsteps = 50;
    double tstart, tfinish;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    alloc_particles(&particles);

    if(0 == my_rank){
        if (argc > 3) {
            fprintf(stderr, "Usage: %s [nparticles [nsteps]]\n", argv[0]);
            return EXIT_FAILURE;
        }

        if (argc > 1) {
            n = atoi(argv[1]);
        }

        if (argc > 2) {
            nsteps = atoi(argv[2]);
        }

        if (n > MAX_PARTICLES) {
            fprintf(stderr, "FATAL: the maximum number of particles is %d\n", MAX_PARTICLES);
            return EXIT_FAILURE;
        }

        init_sph(n, &particles);
    }

    MPI_Bcast( &nsteps,      1, MPI_INT, 0, MPI_COMM_WORLD ); 
    MPI_Bcast( &n_particles, 1, MPI_INT, 0, MPI_COMM_WORLD ); 

    if(0 == my_rank){
        tstart = hpc_gettime();
    }

    bcast_initial_values(&particles);
    compute_blocks();

    for (int s=0; s<nsteps; s++) {
        update(&particles);

        /* the average velocities MUST be computed at each step, even
           if it is not shown (to ensure constant workload per
           iteration) */
        const float avg = avg_velocities(&particles);
        //if (s % 10 == 0 && 0 == my_rank)
          //  printf("step %5d, avgV=%f\n", s, avg);
    }
  
    if(0 == my_rank){
        tfinish = hpc_gettime();
        printf("Elapsed time: %e seconds\n", tfinish - tstart);
    }

    free_particles(&particles);
    free(recvcounts);
    free(displs);
    MPI_Finalize();

    return EXIT_SUCCESS;
}