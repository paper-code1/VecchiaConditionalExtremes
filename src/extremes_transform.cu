#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math_constants.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include "extremes_transform.h"

// Gaussian pdf and cdf (approx for float/double)
template <typename Real>
__device__ inline Real phi_pdf(Real x) {
    const Real inv_sqrt_2pi = (Real)0.39894228040143267794; // 1/sqrt(2pi)
    return exp(-(Real)0.5 * x * x) * inv_sqrt_2pi;
}

template <typename Real>
__device__ inline Real phi_cdf(Real x) {
    // use erf
    return (Real)0.5 * (Real)(1.0 + erf(x / (Real)CUDART_SQRT_TWO));
}

// Delta-Laplace CDF and its inverse via simple Newton iterations on logit scale is heavy;
// we only need marginal map y = Phi^{-1}( F_DL(z) ). We'll compute u = F_DL(z) directly.
// f_DL(z) also needed for Jacobian terms. Parameters: mu, tau (>0), delta(|h|)=1+exp(-(||h||/delta1)^2)

template <typename Real>
__device__ inline Real delta_of_norm(Real norm_h, Real delta1) {
    return (Real)1.0 + exp(- (norm_h / delta1) * (norm_h / delta1));
}

template <typename Real>
__device__ inline Real f_dl_pdf(Real z, Real mu, Real sigma, Real delta) {
    Real t = fabs(z - mu) / sigma;
    Real norm_const = delta / ((Real)2.0 * sigma * tgamma((Real)1.0 / delta));
    return norm_const * exp(-pow(t, delta));
}

// Simpler version using cuda::std::tgamma with regularized lower incomplete gamma series approximation
template <typename Real>
__device__ inline Real F_dl_cdf_approx(Real z, Real mu, Real sigma, Real delta) {
    Real zdiff = z - mu;
    Real w = pow(fabs(zdiff) / sigma, delta);

    // Regularized lower incomplete gamma: P(a, x) = gammainc(x, a)
    Real a = (Real)1.0 / delta;
    Real x = w;
    // Series expansion for P(a,x) with correct initialization (Numerical Recipes gser)
    Real sum = (a > (Real)0.0) ? ((Real)1.0 / a) : (Real)0.0;
    Real term = sum;
    for (int k = 1; k < 200; ++k) {
        term *= x / (a + (Real)k);
        sum += term;
        if (fabs(term) < (Real)1e-12 * fabs(sum)) break;
    }
    Real pre = exp(-x + a * log(x) - lgamma(a));
    Real Psw = (x > 0) ? pre * sum : 0.0;
    Psw = fmin(fmax(Psw, (Real)0.0), (Real)1.0);
    // Match R's sign() behavior: sign(0) == 0
    Real sgn = (zdiff > (Real)0.0 ? (Real)1.0 : (zdiff < (Real)0.0 ? (Real)-1.0 : (Real)0.0));
    return (Real)0.5 + (Real)0.5 * sgn * Psw;
}

// number of threads per batch block
#ifndef SCE_THREADS_PER_BLOCK
#define SCE_THREADS_PER_BLOCK 128
#endif

template <typename Real>
__global__ void sce_kernel(
    Real** d_locs_array, const int* d_lda_locs, const int stridex_locs,
    Real** d_locs_neighbors_array, const int* d_lda_locs_neighbors, const int stridex_neighbors,
    int batchCount, int dim,
    Real** d_obs_array, Real** d_obs_neighbors_array,
    Real s0x, Real s0y, Real x0,
    Real lambda_a, Real kappa_a, Real beta,
    Real mu, Real tau, Real delta1,
    Real* d_sum_log_fdl, Real* d_sum_log_b, Real* d_sum_y_sq)
{
    int b = blockIdx.x; // block index
    if (b >= batchCount) return;
    int n = d_lda_locs[b];
    int m = d_lda_locs_neighbors[b];

    // pointers for first two dims (assume dim>=2 for SCE)
    Real* locs_x    = d_locs_array[b];
    Real* locs_y    = locs_x + stridex_locs;
    Real* locs_nn_x = d_locs_neighbors_array[b];
    Real* locs_nn_y = locs_nn_x + stridex_neighbors;
    Real* y    = d_obs_array[b];
    Real* y_nn = d_obs_neighbors_array[b];

    int tid = threadIdx.x;
    Real sum1 = 0; // log f_DL
    Real sum2 = 0; // log b
    Real sum3 = 0; // y^2

    // loop over block observations strided by threads
    for (int i = tid; i < n; i += blockDim.x) {
        Real sx = locs_x[i];
        Real sy = locs_y[i];
        Real hx = sx - s0x;
        Real hy = sy - s0y;
        Real hnorm = sqrt(hx*hx + hy*hy);
        Real delta = delta_of_norm(hnorm, delta1);
        Real a = x0 * exp( - pow(hnorm / lambda_a, kappa_a) );
        Real bfac = (Real)1.0 + pow(a, beta);
        Real z = (y[i] - a) / bfac;
        Real u = F_dl_cdf_approx(z, mu, tau, delta);
        u = fmin(fmax(u, (Real)1e-12), (Real)1.0 - (Real)1e-12);
        Real ylat = sqrt((Real)2.0) * erfinv((Real)2.0*u - (Real)1.0);
        Real fdl = f_dl_pdf(z, mu, tau, delta);
        // Real fdl = z*z;
        y[i] = ylat;
        // sum1 += log(fmax(fdl, (Real)1e-64));
        // sum2 += log(fmax(bfac, (Real)1e-64));
        sum1 += log(fdl);
        sum2 += log(bfac);
        sum3 += ylat * ylat;
        // sum3 += ylat * ylat;
    }

    // __syncthreads();

    // loop over neighbor observations strided by threads (transform only; do not contribute to sums)
    for (int j = tid; j < m; j += blockDim.x) {
        Real sx = locs_nn_x[j];
        Real sy = locs_nn_y[j];
        Real hx = sx - s0x;
        Real hy = sy - s0y;
        Real hnorm = sqrt(hx*hx + hy*hy);
        Real delta = delta_of_norm(hnorm, delta1);
        Real a = x0 * exp( - pow(hnorm / lambda_a, kappa_a) );
        Real bfac = (Real)1.0 + pow(a, beta);
        Real z = (y_nn[j] - a) / bfac;
        Real u = F_dl_cdf_approx(z, mu, tau, delta);
        u = fmin(fmax(u, (Real)1e-12), (Real)1.0 - (Real)1e-12);
        Real ylat = sqrt((Real)2.0) * erfinv((Real)2.0*u - (Real)1.0);
        y_nn[j] = ylat;
        // sum3 += ylat * ylat;
    }
    // __syncthreads();
    // shared-memory reduction within the CUDA block
    __shared__ Real shm1[SCE_THREADS_PER_BLOCK];
    __shared__ Real shm2[SCE_THREADS_PER_BLOCK];
    __shared__ Real shm3[SCE_THREADS_PER_BLOCK];
    shm1[tid] = sum1;
    shm2[tid] = sum2;
    shm3[tid] = sum3;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shm1[tid] += shm1[tid + offset];
            shm2[tid] += shm2[tid + offset];
            shm3[tid] += shm3[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        d_sum_log_fdl[b] = shm1[0];
        d_sum_log_b[b]   = shm2[0];
        d_sum_y_sq[b]    = shm3[0];
    }
}

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
    cudaStream_t stream)
{
    dim3 grid(batchCount);
    dim3 block(SCE_THREADS_PER_BLOCK);
    sce_kernel<Real><<<grid, block, 0, stream>>>(
        d_locs_array, d_lda_locs, stridex_locs,
        d_locs_neighbors_array, d_lda_locs_neighbors, stridex_neighbors,
        batchCount, dim,
        d_obs_array, d_obs_neighbors_array,
        s0x, s0y, x0,
        lambda_a, kappa_a, beta,
        mu, tau, delta1,
        d_sum_log_fdl, d_sum_log_b, d_sum_y_sq);
}

template void sce_transform<float>(float**, const int*, const int, float**, const int*, const int, int, int, float**, float**, float, float, float, float, float, float, float, float, float, float*, float*, float*, cudaStream_t);
template void sce_transform<double>(double**, const int*, const int, double**, const int*, const int, int, int, double**, double**, double, double, double, double, double, double, double, double, double, double*, double*, double*, cudaStream_t);


// Device reduction utility
template <typename Real>
Real device_sum(const Real* d_array, int n, cudaStream_t stream)
{
    thrust::device_ptr<const Real> p(d_array);
    return thrust::reduce(thrust::cuda::par.on(stream), p, p + n, (Real)0);
}

template float device_sum<float>(const float*, int, cudaStream_t);
template double device_sum<double>(const double*, int, cudaStream_t);


