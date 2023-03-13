/* Mussoni Niccolò 0000970381 */

/****************************************************************************
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
#ifdef GUI
#if __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
#include "hpc.h"

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

// rendering projection parameters
// (the following ought to be "const float", but then the compiler
// would give an error because VIEW_WIDTH and VIEW_HEIGHT are
// initialized with non-literal expressions)
#ifdef GUI

const int MAX_PARTICLES = 5000;
#define WINDOW_WIDTH 1024
#define WINDOW_HEIGHT 768

#else

const int MAX_PARTICLES = 20000;
// Larger window size to accommodate more particles
#define WINDOW_WIDTH 3000
#define WINDOW_HEIGHT 2000

#endif

const int DAM_PARTICLES = 500;

const float VIEW_WIDTH = 1.5 * WINDOW_WIDTH;
const float VIEW_HEIGHT = 1.5 * WINDOW_HEIGHT;

/* Particle data structure; stores position, velocity, and force for
   integration stores density (rho) and pressure values of all the SPHs.
*/
typedef struct {
    float *x, *y;         // position
    float *vx, *vy;       // velocity
    float *fx, *fy;       // force
    float *rho, *p;       // density, pressure
} particle_t;

int n_particles = 0;    // number of currently active particles
int my_rank, comm_sz;
int my_start, my_end, local_n;
int *sendcounts, *displs;

/**
 * Return a random value in [a, b]
 */
float randab(float a, float b)
{
    return a + (b-a)*rand() / (float)(RAND_MAX);
}

/**
 * Set initial position of the particle corresponding to the index to (x, y); initialize all
 * other attributes to default values (zeros).
 */
void init_particle( particle_t *part, int index, float x, float y )
{
    part->x[index] = x;
    part->y[index] = y;
    part->vx[index] = part->vy[index] = 0.0;
    part->fx[index] = part->fy[index] = 0.0;
    part->rho[index] = 0.0;
    part->p[index] = 0.0;
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
 * Initialize the SPH model with `n` particles.
 */
void init_sph(particle_t *part, int n )
{
    n_particles = 0;
    printf("Initializing with %d particles\n", n);

    for (float y = EPS; y < VIEW_HEIGHT - EPS; y += H) {
        for (float x = EPS; x <= VIEW_WIDTH * 0.8f; x += H) {
            if (n_particles < n) {
                float jitter = rand() / (float)RAND_MAX;
                init_particle(part, n_particles, x+jitter, y);
                n_particles++;
            } else {
                return;
            }
        }
    }
    assert(n_particles == n);
}

/**
 * Allocate memory for every attribute of the n particles.
 */
void malloc_particles( particle_t *part, int n ) {
    part->x = (float *)malloc(n * sizeof(*part->x));
    assert(part->x != NULL);

    part->y = (float *)malloc(n * sizeof(*part->y));
    assert(part->y != NULL);

    part->vx = (float *)malloc(n * sizeof(*part->vx));
    assert(part->vx != NULL);

    part->vy = (float *)malloc(n * sizeof(*part->vy));
    assert(part->vy != NULL);

    part->fx = (float *)malloc(n * sizeof(*part->fx));
    assert(part->fx != NULL);

    part->fy = (float *)malloc(n * sizeof(*part->fy));
    assert(part->fy != NULL);

    part->rho = (float *)malloc(n * sizeof(*part->rho));
    assert(part->rho != NULL);
    
    part->p = (float *)malloc(n * sizeof(*part->p));
    assert(part->p != NULL);
}

/**
 * Deallocate memory for every attribute of the n particles.
*/
void free_particles(particle_t *part) {
    free(part->x);
    free(part->y);
    free(part->fx);
    free(part->fy);
    free(part->vx);
    free(part->vy);
    free(part->p);
    free(part->rho);
}

void compute_density_pressure( particle_t *part )
{               
    const float HSQ = H * H;    // radius^2 for optimization

    /* Smoothing kernels defined in Muller and their gradients adapted
       to 2D per "SPH Based Shallow Water Simulation" by Solenthaler
       et al. */
    const float POLY6 = 4.0 / (M_PI * pow(H, 8));
    int local_n = sendcounts[my_rank];
    int local_i = 0;

    float *local_p = (float *)malloc(local_n * sizeof(*local_p));
    float *local_rho = (float *)malloc(local_n * sizeof(*local_rho));

    for (int i = my_start; i < my_end; i++) {
        float pi_rho = 0.0;
        for (int j = 0; j < n_particles; j++) {
            const float dx = part->x[j] - part->x[i];
            const float dy = part->y[j] - part->y[i];
            const float d2 = dx*dx + dy*dy;

            if (d2 < HSQ) {
                pi_rho += MASS * POLY6 * pow(HSQ - d2, 3.0);
            }
        }

        local_rho[local_i] = pi_rho;
        local_p[local_i] = GAS_CONST * (pi_rho - REST_DENS); 
        local_i++;
    }

    MPI_Allgatherv( local_rho, 
                    local_n,
                    MPI_FLOAT,
                    part->rho,
                    sendcounts,
                    displs,
                    MPI_FLOAT,
                    MPI_COMM_WORLD);
    
    MPI_Allgatherv( local_p, 
                    local_n,
                    MPI_FLOAT,
                    part->p,
                    sendcounts,
                    displs,
                    MPI_FLOAT,
                    MPI_COMM_WORLD);

    free(local_rho);
    free(local_p);
}

void compute_forces(particle_t *part)
{
    /* Smoothing kernels defined in Muller and their gradients adapted
       to 2D per "SPH Based Shallow Water Simulation" by Solenthaler
       et al. */
    const float SPIKY_GRAD = -10.0 / (M_PI * pow(H, 5));
    const float VISC_LAP = 40.0 / (M_PI * pow(H, 5));
    const float EPS = 1e-6;
    int local_i = 0;
    int local_n = sendcounts[my_rank];

    float *local_fx = (float *)malloc(local_n * sizeof(*local_fx));
    float *local_fy = (float *)malloc(local_n * sizeof(*local_fy));

    MPI_Bcast(part->vx, MAX_PARTICLES, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(part->vy, MAX_PARTICLES, MPI_FLOAT, 0, MPI_COMM_WORLD);

    for (int i = my_start; i < my_end; i++) {
        float fpress_x = 0.0, fpress_y = 0.0;
        float fvisc_x = 0.0, fvisc_y = 0.0;

        for (int j = 0; j < n_particles; j++) {
            if (i == j)
                continue;

            const float dx = part->x[j] - part->x[i];
            const float dy = part->y[j] - part->y[i];
            const float dist = hypotf(dx, dy) + EPS; // avoids division by zero later on

            if (dist < H) {
                const float norm_dx = dx / dist;
                const float norm_dy = dy / dist;
                // compute pressure force contribution
                fpress_x += -norm_dx * MASS * (part->p[i] + part->p[j]) / (2 * part->rho[j]) * SPIKY_GRAD * pow(H - dist, 3);
                fpress_y += -norm_dy * MASS * (part->p[i] + part->p[j]) / (2 * part->rho[j]) * SPIKY_GRAD * pow(H - dist, 3);
                // compute viscosity force contribution
                fvisc_x += VISC * MASS * (part->vx[j] - part->vx[i]) / part->rho[j] * VISC_LAP * (H - dist);
                fvisc_y += VISC * MASS * (part->vy[j] - part->vy[i]) / part->rho[j] * VISC_LAP * (H - dist);
            }
        }
        const float fgrav_x = Gx * MASS / part->rho[i];
        const float fgrav_y = Gy * MASS / part->rho[i];

        local_fx[local_i] = fpress_x + fvisc_x + fgrav_x;
        local_fy[local_i] = fpress_y + fvisc_y + fgrav_y;
        local_i++;
    }

    MPI_Allgatherv( local_fx, 
                    local_n,
                    MPI_FLOAT,
                    part->fx,
                    sendcounts,
                    displs,
                    MPI_FLOAT,
                    MPI_COMM_WORLD);

    MPI_Allgatherv( local_fy, 
                    local_n,
                    MPI_FLOAT,
                    part->fy,
                    sendcounts,
                    displs,
                    MPI_FLOAT,
                    MPI_COMM_WORLD);

    free(local_fx);
    free(local_fy);
}

void integrate(particle_t *part)
{
    int local_n = sendcounts[my_rank];
    int j = 0;
    
    float *local_x = (float *)malloc(local_n * sizeof(*local_x));
    float *local_y = (float *)malloc(local_n * sizeof(*local_y));
    float *local_vx = (float *)malloc(local_n * sizeof(*local_vx));
    float *local_vy = (float *)malloc(local_n * sizeof(*local_vy));

    MPI_Scatterv(part->x, 
                 sendcounts, 
                 displs, 
                 MPI_FLOAT, 
                 local_x, 
                 local_n, 
                 MPI_FLOAT, 
                 0, 
                 MPI_COMM_WORLD);
    
    MPI_Scatterv(part->y, 
                 sendcounts, 
                 displs, 
                 MPI_FLOAT, 
                 local_y, 
                 local_n, 
                 MPI_FLOAT, 
                 0, 
                 MPI_COMM_WORLD);

    MPI_Scatterv(part->vx, 
                 sendcounts, 
                 displs, 
                 MPI_FLOAT, 
                 local_vx, 
                 local_n, 
                 MPI_FLOAT, 
                 0, 
                 MPI_COMM_WORLD);

    MPI_Scatterv(part->vy, 
                 sendcounts, 
                 displs, 
                 MPI_FLOAT, 
                 local_vy, 
                 local_n, 
                 MPI_FLOAT, 
                 0, 
                 MPI_COMM_WORLD);

    for (int i = my_start; i < my_end; i++) {
        local_vx[j] += DT * part->fx[i] / part->rho[i];
        local_vy[j] += DT * part->fy[i] / part->rho[i];
        local_x[j] += DT * local_vx[j];
        local_y[j] += DT * local_vy[j];

        // enforce boundary conditions
        if (local_x[j] - EPS < 0.0) {
            local_vx[j] *= BOUND_DAMPING;
            local_x[j] = EPS;
        }
        if (local_x[j] + EPS > VIEW_WIDTH) {
            local_vx[j] *= BOUND_DAMPING;
            local_x[j] = VIEW_WIDTH - EPS;
        }
        if (local_y[j] - EPS < 0.0) {
            local_vy[j] *= BOUND_DAMPING;
            local_y[j] = EPS;
        }
        if (local_y[j] + EPS > VIEW_HEIGHT) {
            local_vy[j] *= BOUND_DAMPING;
            local_y[j] = VIEW_HEIGHT - EPS;
        }

        j++;
    }

    MPI_Allgatherv( local_x, 
                    local_n,
                    MPI_FLOAT,
                    part->x,
                    sendcounts,
                    displs,
                    MPI_FLOAT,
                    MPI_COMM_WORLD);
    
    MPI_Allgatherv( local_y, 
                    local_n,
                    MPI_FLOAT,
                    part->y,
                    sendcounts,
                    displs,
                    MPI_FLOAT,
                    MPI_COMM_WORLD);

    MPI_Allgatherv( local_vx, 
                    local_n,
                    MPI_FLOAT,
                    part->vx,
                    sendcounts,
                    displs,
                    MPI_FLOAT,
                    MPI_COMM_WORLD);
    
    MPI_Allgatherv( local_vy, 
                    local_n,
                    MPI_FLOAT,
                    part->vy,
                    sendcounts,
                    displs,
                    MPI_FLOAT,
                    MPI_COMM_WORLD);

    free(local_x);
    free(local_y);
    free(local_vx);
    free(local_vy);
}

float avg_velocities(particle_t *part)
{
    double result = 0.0;
    double my_result = 0.0;

    for (int i = my_start; i < my_end; i++) {
        /* the hypot(x,y) function is equivalent to sqrt(x*x +
           y*y); */
        my_result += hypot(part->vx[i], part->vy[i]) / n_particles;
    }

    MPI_Reduce( &my_result, 
                &result,
                1, 
                MPI_DOUBLE,
                MPI_SUM,
                0,
                MPI_COMM_WORLD
                );
    
    return result;
}

void update(particle_t *part)
{
    compute_density_pressure(part);
    compute_forces(part);
    integrate(part);
}

int main(int argc, char **argv)
{
    srand(1234);

    particle_t particles;
    int n = DAM_PARTICLES;
    int nsteps = 50;
    double start, end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    sendcounts = (int*)malloc(comm_sz * sizeof(*sendcounts)); assert(sendcounts != NULL);
    displs = (int*)malloc(comm_sz * sizeof(*displs)); assert(displs != NULL);

    malloc_particles(&particles, MAX_PARTICLES);

    if (0 == my_rank) {
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

        init_sph(&particles, n);
    }

    /* Send the number of steps and particles to all the processes */
    MPI_Bcast(&nsteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_particles, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /**
     * Compute the starting and ending position of each
     * block (iend is actually one element _past_ the ending position) 
     */
    for (int i = 0; i < comm_sz; i++) {
        const int istart = n_particles * i / comm_sz;
        const int iend = n_particles * (i + 1) / comm_sz;
        const int blklen = iend - istart;
        sendcounts[i] = blklen;
        displs[i] = istart;
    }

    my_start = (my_rank * n_particles) / comm_sz;
    my_end = ((my_rank + 1) * n_particles) / comm_sz;
    local_n = sendcounts[my_rank];

    start = hpc_gettime();
    /* Send the x and y attributes to all the processes, needed for the first iteration of update */
    MPI_Bcast((&particles)->x, n_particles, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast((&particles)->y, n_particles, MPI_FLOAT, 0, MPI_COMM_WORLD);

    for (int s = 0; s < nsteps; s++) {
        update(&particles);
        
        /* the average velocities MUST be computed at each step, even
        if it is not shown (to ensure constant workload per
        iteration) */
        const float avg = avg_velocities(&particles);

        /*if (0 == my_rank && s % 10 == 0)
            printf("step %5d, avgV=%f\n", s, avg);*/
    }

    if (0 == my_rank) {
        end = hpc_gettime(); 
        printf("Elapsed time: %f seconds\n", end - start);
    }

    free_particles(&particles);
    free(sendcounts);
    free(displs);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
