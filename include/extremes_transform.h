#ifndef EXTREMES_TRANSFORM_H
#define EXTREMES_TRANSFORM_H

#include <cuda_runtime.h>

struct SceTerms {
    double sum_log_fdl;
    double sum_log_b;
    double sum_y_sq;
};

template <typename Real>
void sce_transform(
    Real** d_locs_array, const int* d_lda_locs, const int stridex_locs,
    Real** d_locs_neighbors_array, const int* d_lda_locs_neighbors, const int stridex_neighbors,
    int batchCount, int dim,
    Real** d_obs_array, Real** d_obs_neighbors_array,
    Real s0x, Real s0y, Real x0,
    Real lambda_a, Real kappa_a, Real beta,
    Real mu, Real tau, Real delta1,
    Real* d_sum_log_fdl, Real* d_sum_log_b, Real* d_sum_y_sq,
    cudaStream_t stream);


// Device-side sum utility (implemented in .cu to allow Thrust reductions)
template <typename Real>
Real device_sum(const Real* d_array, int n, cudaStream_t stream);

#endif


