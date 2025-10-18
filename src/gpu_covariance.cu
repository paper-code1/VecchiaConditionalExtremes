#include <cmath>
#include <iostream>
#include <curand_kernel.h>
#include <type_traits>
#include "gpu_covariance.h"
#include "block_info.h"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <mpi.h>
#define BATCHCOUNT_MAX 65536
#define THREADS_PER_BLOCK 64
#define MAXDIM 32

#define THREAD_X (16)
#define THREAD_Y (16)

// (coalesced memory access)
template <typename Real>
__global__ void RBF_matcov_kernel(const Real* X1, int ldx1, int incx1, int stridex1,
                                          const Real* X2, int ldx2, int incx2, int stridex2,
                                          Real* C, int ldc, int n, int dim, 
                                          Real sigma2, Real range, Real nugget, 
                                          bool nugget_tag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < ldx1 && j < ldx2 && i >= 0 && j >= 0) {
        if (X1 == X2 && j > i) return;
        Real dist_sqaure = 0;
        for (int k = 0; k < dim; k++) {
            Real x1 = X1[i * incx1 + k * stridex1];
            Real x2 = X2[j * incx2 + k * stridex2];
            dist_sqaure += (x1 - x2) * (x1 - x2);
        }
        Real scaled_distance = sqrt(dist_sqaure) / range;
        C[i + j * ldc] = sigma2 * exp( - scaled_distance );
    }
    // add nugget
    if (i == j && i < ldx1 && j < ldx2 && nugget_tag) {
        C[i + j * ldc] += nugget;
    }
}

// (coalesced memory access)
template <typename Real>
__global__ void PowerExp_matcov_scaled_kernel(const Real* __restrict__ X1, int ldx1, int incx1, int stridex1,
                                          const Real* X2, int ldx2, int incx2, int stridex2,
                                          Real* C, int ldc, int n, int dim, 
                                          Real sigma2, Real smoothness, Real nugget, 
                                          const Real* __restrict__ range, bool nugget_tag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < ldx1 && j < ldx2 && i >= 0 && j >= 0) {
        if (X1 == X2 && j > i) return;
        Real dist_sqaure = 0;
        #pragma unroll 4
        for (int k = 0; k < dim; k++) {
            Real x1 = X1[i * incx1 + k * stridex1];
            Real x2 = X2[j * incx2 + k * stridex2];
            Real diff = x1 - x2;
            dist_sqaure += diff * diff * range[k];
        }
        Real scaled_distance = sqrt(dist_sqaure);
        Real power_distance = pow(scaled_distance, smoothness);
        C[i + j * ldc] = sigma2 * exp( - power_distance );
    }
    // add nugget
    if (i == j && i < ldx1 && j < ldx2 && nugget_tag) {
        C[i + j * ldc] += nugget;
    }
}

template <typename Real>
__global__ void Matern12_scaled_matcov_kernel(const Real* __restrict__ X1, int ldx1, int incx1, int stridex1,
                                          const Real* __restrict__ X2, int ldx2, int incx2, int stridex2,
                                          Real* __restrict__ C, int ldc, int n, int dim, 
                                          Real sigma2, Real nugget, 
                                          const Real* __restrict__ range, bool nugget_tag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < ldx1 && j < ldx2 && i >= 0 && j >= 0) {
        if (X1 == X2 && j > i) return;
        Real dist_sqaure = 0;
        #pragma unroll 4
        for (int k = 0; k < dim; k++) {
            Real x1 = X1[i * incx1 + k * stridex1];
            Real x2 = X2[j * incx2 + k * stridex2];
            Real diff = x1 - x2;
            dist_sqaure += diff * diff * range[k];
        }
        Real scaled_distance = sqrt(dist_sqaure);
        Real a0 = 1.0;
        // double a1 = 1.0;
        // double a2 = 2.0 / 5.0;
        // double a3 = 1.0 / 15.0;
        Real item_poly = a0;
        // + a1 * scaled_distance + a2 * scaled_distance * scaled_distance + a3 * scaled_distance * scaled_distance * scaled_distance;
        C[i + j * ldc] = sigma2 * item_poly * exp( - scaled_distance );
    }
    // add nugget
    if (i == j && i < ldx1 && j < ldx2 && nugget_tag) {
        C[i + j * ldc] += nugget;
    }
}

template <typename Real>
__global__ void Matern32_scaled_matcov_kernel(const Real* __restrict__ X1, int ldx1, int incx1, int stridex1,
                                          const Real* __restrict__ X2, int ldx2, int incx2, int stridex2,
                                          Real* __restrict__ C, int ldc, int n, int dim, 
                                          Real sigma2, Real nugget, 
                                          const Real* __restrict__ range, bool nugget_tag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < ldx1 && j < ldx2 && i >= 0 && j >= 0) {
        if (X1 == X2 && j > i) return;
        Real dist_sqaure = 0;
        #pragma unroll 4
        for (int k = 0; k < dim; k++) {
            Real x1 = X1[i * incx1 + k * stridex1];
            Real x2 = X2[j * incx2 + k * stridex2];
            Real diff = x1 - x2;
            dist_sqaure += diff * diff * range[k];
        }
        Real scaled_distance = sqrt(dist_sqaure);
        Real a0 = 1.0;
        Real a1 = 1.0;
        // double a2 = 2.0 / 5.0;
        // double a3 = 1.0 / 15.0;
        Real item_poly = a0 + a1 * scaled_distance;
        //  + a2 * scaled_distance * scaled_distance + a3 * scaled_distance * scaled_distance * scaled_distance;
        C[i + j * ldc] = sigma2 * item_poly * exp( - scaled_distance );
    }
    // add nugget
    if (i == j && i < ldx1 && j < ldx2 && nugget_tag) {
        C[i + j * ldc] += nugget;
    }
}

template <typename Real>
__global__ void Matern52_scaled_matcov_kernel(const Real* __restrict__ X1, int ldx1, int incx1, int stridex1,
                                          const Real* __restrict__ X2, int ldx2, int incx2, int stridex2,
                                          Real* __restrict__ C, int ldc, int n, int dim, 
                                          Real sigma2, Real nugget, 
                                          const Real* __restrict__ range, bool nugget_tag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < ldx1 && j < ldx2 && i >= 0 && j >= 0) {
        Real dist_sqaure = 0;
        for (int k = 0; k < dim; k++) {
            Real x1 = X1[i * incx1 + k * stridex1];
            Real x2 = X2[j * incx2 + k * stridex2];
            Real diff = x1 - x2;
            dist_sqaure += diff * diff * range[k];
        }
        Real scaled_distance = sqrt(dist_sqaure);
        Real a0 = 1.0;
        Real a1 = 1.0;
        Real a2 = 1.0 / 3.0;
        Real item_poly = a0 + a1 * scaled_distance + a2 * scaled_distance * scaled_distance;
        C[i + j * ldc] = sigma2 * item_poly * exp( - scaled_distance );
    }
    // add nugget
    if (i == j && i < ldx1 && j < ldx2 && nugget_tag) {
        C[i + j * ldc] += nugget;
    }
}

template <typename Real>
__global__ void Matern72_scaled_matcov_kernel(
    const Real* __restrict__ X1, int ldx1, int incx1, int stridex1,
    const Real* __restrict__ X2, int ldx2, int incx2, int stridex2,
    Real* __restrict__ C, int ldc, int n, int dim, 
    Real sigma2, Real nugget, 
    const Real* __restrict__ range, bool nugget_tag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < ldx1 && j < ldx2 && i >= 0 && j >= 0) {
        Real dist_sqaure = 0;
        for (int k = 0; k < dim; k++) {
            Real x1 = X1[i * incx1 + k * stridex1];
            Real x2 = X2[j * incx2 + k * stridex2];
            Real diff = x1 - x2;
            dist_sqaure += diff * diff * range[k];
        }
        Real scaled_distance = sqrt(dist_sqaure);
        Real a0 = 1.0;
        Real a1 = 1.0;
        Real a2 = 2.0 / 5.0;
        Real a3 = 1.0 / 15.0;
        Real item_poly = a0 + a1 * scaled_distance + a2 * scaled_distance * scaled_distance + a3 * scaled_distance * scaled_distance * scaled_distance;
        C[i + j * ldc] = sigma2 * item_poly * exp( - scaled_distance );
    }
    // add nugget
    if (i == j && i < ldx1 && j < ldx2 && nugget_tag) {
        C[i + j * ldc] += nugget;
    }
}

// core kernel for batched matrix generation 
template <typename Real>
__device__ void Matern72_scaled_matcov_vbatched_kernel_device(
    const Real* __restrict__ d_X1, int ldx1, int incx1, int stridex1,
    const Real* __restrict__ d_X2, int ldx2, int incx2, int stridex2,
    Real* __restrict__ d_C, int ldc, int n, int dim, 
    Real sigma2, const Real* __restrict__ range, 
    Real nugget, bool nugget_tag,
    int gtx, int gty) {
    if (gtx < ldx1 && gty < ldx2 && gtx >= 0 && gty >= 0) {
        if (d_X1 == d_X2 && gty > gtx) return;
        Real dist_sqaure = 0;
        for (int k = 0; k < dim; k++) {
            Real x1 = d_X1[gtx * incx1 + k * stridex1];
            Real x2 = d_X2[gty * incx2 + k * stridex2];
            Real diff = x1 - x2;
            dist_sqaure += diff * diff * range[k];
        }
        Real scaled_distance = sqrt(dist_sqaure);
        Real a0 = 1.0;
        Real a1 = 1.0;
        Real a2 = 2.0 / 5.0;
        Real a3 = 1.0 / 15.0;
        Real item_poly = a0 + a1 * scaled_distance + a2 * scaled_distance * scaled_distance + a3 * scaled_distance * scaled_distance * scaled_distance;
        d_C[gtx + gty * ldc] = sigma2 * item_poly * exp( - scaled_distance );
    }
    // add nugget
    if (gtx == gty && gtx < ldx1 && gty < ldx2 && nugget_tag) {
        d_C[gtx + gty * ldc] += nugget;
    }
}

// core kernel for batched matrix generation 
template <typename Real>
__device__ void Matern52_scaled_matcov_vbatched_kernel_device(
    const Real* __restrict__ d_X1, int ldx1, int incx1, int stridex1,
    const Real* __restrict__ d_X2, int ldx2, int incx2, int stridex2,
    Real* __restrict__ d_C, int ldc, int n, int dim, 
    Real sigma2, const Real* __restrict__ range, 
    Real nugget, bool nugget_tag,
    int gtx, int gty) {
    if (gtx < ldx1 && gty < ldx2 && gtx >= 0 && gty >= 0) {
        if (d_X1 == d_X2 && gty > gtx) return;
        Real dist_sqaure = 0;
        for (int k = 0; k < dim; k++) {
            Real x1 = d_X1[gtx * incx1 + k * stridex1];
            Real x2 = d_X2[gty * incx2 + k * stridex2];
            Real diff = x1 - x2;
            dist_sqaure += diff * diff * range[k];
        }
        Real scaled_distance = sqrt(dist_sqaure);
        Real a0 = 1.0;
        Real a1 = 1.0;
        Real a2 = 1.0 / 3.0;
        Real item_poly = a0 + a1 * scaled_distance + a2 * scaled_distance * scaled_distance;
        d_C[gtx + gty * ldc] = sigma2 * item_poly * exp( - scaled_distance );
    }
    // add nugget
    if (gtx == gty && gtx < ldx1 && gty < ldx2 && nugget_tag) {
        d_C[gtx + gty * ldc] += nugget;
    }
}

// core kernel for batched matrix generation 
template <typename Real>
__device__ void Matern32_scaled_matcov_vbatched_kernel_device(
    const Real* __restrict__ d_X1, int ldx1, int incx1, int stridex1,
    const Real* __restrict__ d_X2, int ldx2, int incx2, int stridex2,
    Real* __restrict__ d_C, int ldc, int n, int dim, 
    Real sigma2, const Real* __restrict__ range, 
    Real nugget, bool nugget_tag,
    int gtx, int gty) {
    if (gtx < ldx1 && gty < ldx2 && gtx >= 0 && gty >= 0) {
        if (d_X1 == d_X2 && gty > gtx) return;
        Real dist_sqaure = 0;
        for (int k = 0; k < dim; k++) {
            Real x1 = d_X1[gtx * incx1 + k * stridex1];
            Real x2 = d_X2[gty * incx2 + k * stridex2];
            Real diff = x1 - x2;
            dist_sqaure += diff * diff * range[k];
        }
        Real scaled_distance = sqrt(dist_sqaure);
        Real a0 = 1.0;
        Real a1 = 1.0;
        // Real a2 = 1.0 / 2.0;
        Real item_poly = a0 + a1 * scaled_distance;
        d_C[gtx + gty * ldc] = sigma2 * item_poly * exp( - scaled_distance );
    }
    // add nugget
    if (gtx == gty && gtx < ldx1 && gty < ldx2 && nugget_tag) {
        d_C[gtx + gty * ldc] += nugget;
    }
}

// core kernel for batched matrix generation 
template <typename Real>
__device__ void Matern12_scaled_matcov_vbatched_kernel_device(
    const Real* __restrict__ d_X1, int ldx1, int incx1, int stridex1,
    const Real* __restrict__ d_X2, int ldx2, int incx2, int stridex2,
    Real* __restrict__ d_C, int ldc, int n, int dim, 
    Real sigma2, const Real* __restrict__ range, 
    Real nugget, bool nugget_tag,
    int gtx, int gty) {
    if (gtx < ldx1 && gty < ldx2 && gtx >= 0 && gty >= 0) {
        if (d_X1 == d_X2 && gty > gtx) return;
        Real dist_sqaure = 0;
        for (int k = 0; k < dim; k++) {
            Real x1 = d_X1[gtx * incx1 + k * stridex1];
            Real x2 = d_X2[gty * incx2 + k * stridex2];
            Real inv_r2 = range[k];
            Real diff = x1 - x2;
            dist_sqaure += diff * diff * inv_r2;
        }
        Real scaled_distance = sqrt(dist_sqaure);
        Real a0 = 1.0;
        // Real a1 = 1.0;
        Real item_poly = a0;
        d_C[gtx + gty * ldc] = sigma2 * item_poly * exp( - scaled_distance );
    }
    // add nugget
    if (gtx == gty && gtx < ldx1 && gty < ldx2 && nugget_tag) {
        d_C[gtx + gty * ldc] += nugget;
    }
}

// batched matrix generation 
template <typename Real>
__global__ void Matern72_scaled_matcov_vbatched_kernel(
    Real** d_X1, const int* ldx1, int incx1, int stridex1,
    Real** d_X2, const int* ldx2, int incx2, int stridex2,
    Real** d_C, const int* ldc, const int* n, int dim, 
    const Real sigma2, const Real nugget, const Real* range, bool nugget_tag) {
    // batched id
    const int batchid = blockIdx.z;
    const int gtx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gty = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ Real s_range[MAXDIM];
    if (dim <= MAXDIM) {
        if (threadIdx.x < dim) {
            s_range[threadIdx.x] = range[threadIdx.x];
        }
        __syncthreads();
    }
    Matern72_scaled_matcov_vbatched_kernel_device<Real>(d_X1[batchid], ldx1[batchid], incx1, stridex1, d_X2[batchid], ldx2[batchid], incx2, stridex2, d_C[batchid], ldc[batchid], n[batchid], dim, sigma2, s_range, nugget, nugget_tag, gtx, gty);
}

// batched matrix generation 
template <typename Real>
__global__ void Matern52_scaled_matcov_vbatched_kernel(
    Real** d_X1, const int* ldx1, int incx1, int stridex1,
    Real** d_X2, const int* ldx2, int incx2, int stridex2,
    Real** d_C, const int* ldc, const int* n, int dim, 
    const Real sigma2, const Real nugget, const Real* range, bool nugget_tag) {
    // batched id
    const int batchid = blockIdx.z;
    const int gtx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gty = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ Real s_range[MAXDIM];
    if (dim <= MAXDIM) {
        if (threadIdx.x < dim) {
            s_range[threadIdx.x] = range[threadIdx.x];
        }
        __syncthreads();
    }
    Matern52_scaled_matcov_vbatched_kernel_device<Real>(d_X1[batchid], ldx1[batchid], incx1, stridex1, d_X2[batchid], ldx2[batchid], incx2, stridex2, d_C[batchid], ldc[batchid], n[batchid], dim, sigma2, s_range, nugget, nugget_tag, gtx, gty);
}

// batched matrix generation 
template <typename Real>
__global__ void Matern32_scaled_matcov_vbatched_kernel(
    Real** d_X1, const int* ldx1, int incx1, int stridex1,
    Real** d_X2, const int* ldx2, int incx2, int stridex2,
    Real** d_C, const int* ldc, const int* n, int dim, 
    const Real sigma2, const Real nugget, const Real* range, bool nugget_tag) {
    // batched id
    const int batchid = blockIdx.z;
    const int gtx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gty = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ Real s_range[MAXDIM];
    if (dim <= MAXDIM) {
        if (threadIdx.x < dim) {
            s_range[threadIdx.x] = range[threadIdx.x];
        }
        __syncthreads();
    }
    Matern32_scaled_matcov_vbatched_kernel_device<Real>(d_X1[batchid], ldx1[batchid], incx1, stridex1, d_X2[batchid], ldx2[batchid], incx2, stridex2, d_C[batchid], ldc[batchid], n[batchid], dim, sigma2, s_range,   nugget, nugget_tag, gtx, gty);
}

// batched matrix generation 
template <typename Real>
__global__ void Matern12_scaled_matcov_vbatched_kernel(
    Real** d_X1, const int* ldx1, int incx1, int stridex1,
    Real** d_X2, const int* ldx2, int incx2, int stridex2,
    Real** d_C, const int* ldc, const int* n, int dim, 
    const Real sigma2, const Real nugget, const Real* range, bool nugget_tag) {
    // batched id
    const int batchid = blockIdx.z;
    const int gtx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gty = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ Real s_range[MAXDIM];
    if (dim <= MAXDIM) {
        if (threadIdx.x < dim) {
            s_range[threadIdx.x] = range[threadIdx.x];
        }
        __syncthreads();
    }
    Matern12_scaled_matcov_vbatched_kernel_device<Real>(d_X1[batchid], ldx1[batchid], incx1, stridex1, d_X2[batchid], ldx2[batchid], incx2, stridex2, d_C[batchid], ldc[batchid], n[batchid], dim, sigma2, s_range, nugget, nugget_tag, gtx, gty);
}


template <typename Real>
__global__ void Matern72_matcov_kernel(const Real* X1, int ldx1, int incx1, int stridex1,
                                          const Real* X2, int ldx2, int incx2, int stridex2,
                                          Real* C, int ldc, int n, int dim, 
                                          Real sigma2, Real range, 
                                          Real nugget, bool nugget_tag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < ldx1 && j < ldx2 && i >= 0 && j >= 0) {
        Real dist_sqaure = 0;
        for (int k = 0; k < dim; k++) {
            Real x1 = X1[i * incx1 + k * stridex1];
            Real x2 = X2[j * incx2 + k * stridex2];
            dist_sqaure += (x1 - x2) * (x1 - x2);
        }
        Real scaled_distance = sqrt(dist_sqaure) / range;
        Real item_poly = 1 + sqrt((Real)7.0) * scaled_distance + (Real)7.0 * scaled_distance * scaled_distance / (Real)3.0;  
        // + 7.0 * sqrt(7.0) * scaled_distance * scaled_distance * scaled_distance / 15.0; 
        C[i + j * ldc] = sigma2 * item_poly * exp( - sqrt((Real)7.0) * scaled_distance );
    }
    // add nugget
    if (i == j && i < ldx1 && j < ldx2 && nugget_tag) {
        C[i + j * ldc] += nugget;
    }
}

template <typename Real>
void Matern_matcov(const Real* d_X1, int ldx1, int incx1, int stridex1,
                const Real* d_X2, int ldx2, int incx2, int stridex2,
                Real* d_C, int ldc, int n, int dim, const std::vector<double> &theta, bool nugget_tag,
                cudaStream_t stream) {
    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((ldx1 + blockDim.x - 1) / blockDim.x, (ldx2 + blockDim.y - 1) / blockDim.y);
    // the smoothness = 3.5 theta[0]: variance, theta[1]: range, theta[3]: nugget
    Matern72_matcov_kernel<Real><<<gridDim, blockDim, 0, stream>>>(d_X1, ldx1, incx1, stridex1, d_X2, ldx2, incx2, stridex2, d_C, ldc, n, dim, (Real)theta[0], (Real)theta[1], (Real)theta[3], nugget_tag);
}

template <typename Real>
void PowerExp_scaled_matcov(const Real* d_X1, int ldx1, int incx1, int stridex1,
                const Real* d_X2, int ldx2, int incx2, int stridex2,
                Real* d_C, int ldc, int n, int dim, const std::vector<double> &theta,
                const Real* range, bool nugget_tag,
                cudaStream_t stream) {
    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((ldx1 + blockDim.x - 1) / blockDim.x, (ldx2 + blockDim.y - 1) / blockDim.y);
    // theta[0]: variance, theta[1]: smoothness, theta[2]: nugget, theta[3:]: range
    PowerExp_matcov_scaled_kernel<Real><<<gridDim, blockDim, 0, stream>>>(d_X1, ldx1, incx1, stridex1, d_X2, ldx2, incx2, stridex2, d_C, ldc, n, dim, (Real)theta[0], (Real)theta[1], (Real)theta[2], range, nugget_tag);
}

template <typename Real>
void Matern72_scaled_matcov(const Real* d_X1, int ldx1, int incx1, int stridex1,
                const Real* d_X2, int ldx2, int incx2, int stridex2,
                Real* d_C, int ldc, int n, int dim, const std::vector<double> &theta,
                const Real* range, bool nugget_tag,
                cudaStream_t stream) {
    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((ldx1 + blockDim.x - 1) / blockDim.x, (ldx2 + blockDim.y - 1) / blockDim.y);
    // theta[0]: variance, theta[1]: nugget, theta[2:]: range
    Matern72_scaled_matcov_kernel<Real><<<gridDim, blockDim, 0, stream>>>(d_X1, ldx1, incx1, stridex1, d_X2, ldx2, incx2, stridex2, d_C, ldc, n, dim, (Real)theta[0], (Real)theta[1], range, nugget_tag);
}

// batched version
template <typename Real>
void Matern72_scaled_matcov_vbatched(
    Real** d_X1, const int* ldx1, int incx1, int stridex1,
    Real** d_X2, const int* ldx2, int incx2, int stridex2,
    Real** d_C, const int* ldc, const int* n, int dim, const std::vector<double> &theta,
    const Real* range, bool nugget_tag, 
    int max_ldx1, int max_ldx2,
    int batchCount, cudaStream_t stream) {
    // Launch kernel for each single batch
    dim3 blockDim(THREAD_X, THREAD_Y, 1);
    const int gridx = ((max_ldx1 + blockDim.x - 1) / blockDim.x);
    const int gridy = ((max_ldx2 + blockDim.y - 1) / blockDim.y);
    // theta[0]: variance, theta[1]: nugget, theta[2:]: range
    // for loop over each batch
    const int max_batch = 65535;
    for (int i = 0; i < batchCount; i+= max_batch) {
        int gridz = min(max_batch, batchCount - i);
        dim3 gridDim(gridx, gridy, gridz);
        Matern72_scaled_matcov_vbatched_kernel<Real><<<gridDim, blockDim, 0, stream>>>(d_X1 + i, ldx1 + i, incx1, stridex1, d_X2 + i, ldx2 + i, incx2, stridex2, d_C + i, ldc + i, n + i, dim, (Real)theta[0], (Real)theta[1], range, nugget_tag);
    }
}

// batched version
template <typename Real>
void Matern52_scaled_matcov_vbatched(
    Real** d_X1, const int* ldx1, int incx1, int stridex1,
    Real** d_X2, const int* ldx2, int incx2, int stridex2,
    Real** d_C, const int* ldc, const int* n, int dim, const std::vector<double> &theta,
    const Real* range, bool nugget_tag, 
    int max_ldx1, int max_ldx2,
    int batchCount, cudaStream_t stream) {
    // Launch kernel for each single batch
    dim3 blockDim(THREAD_X, THREAD_Y, 1);
    const int gridx = ((max_ldx1 + blockDim.x - 1) / blockDim.x);
    const int gridy = ((max_ldx2 + blockDim.y - 1) / blockDim.y);
    // theta[0]: variance, theta[1]: nugget, theta[2:]: range
    // for loop over each batch
    const int max_batch = 65535;
    for (int i = 0; i < batchCount; i+= max_batch) {
        int gridz = min(max_batch, batchCount - i);
        dim3 gridDim(gridx, gridy, gridz);
        Matern52_scaled_matcov_vbatched_kernel<Real><<<gridDim, blockDim, 0, stream>>>(d_X1 + i, ldx1 + i, incx1, stridex1, d_X2 + i, ldx2 + i, incx2, stridex2, d_C + i, ldc + i, n + i, dim, (Real)theta[0], (Real)theta[1], range, nugget_tag);
    }
}

// batched version
template <typename Real>
void Matern32_scaled_matcov_vbatched(
    Real** d_X1, const int* ldx1, int incx1, int stridex1,
    Real** d_X2, const int* ldx2, int incx2, int stridex2,
    Real** d_C, const int* ldc, const int* n, int dim, const std::vector<double> &theta,
    const Real* range, bool nugget_tag, 
    int max_ldx1, int max_ldx2,
    int batchCount, cudaStream_t stream) {
    // Launch kernel for each single batch
    dim3 blockDim(THREAD_X, THREAD_Y, 1);
    const int gridx = ((max_ldx1 + blockDim.x - 1) / blockDim.x);
    const int gridy = ((max_ldx2 + blockDim.y - 1) / blockDim.y);
    // theta[0]: variance, theta[1]: nugget, theta[2:]: range
    // for loop over each batch
    const int max_batch = 65535;
    for (int i = 0; i < batchCount; i+= max_batch) {
        int gridz = min(max_batch, batchCount - i);
        dim3 gridDim(gridx, gridy, gridz);
        Matern32_scaled_matcov_vbatched_kernel<Real><<<gridDim, blockDim, 0, stream>>>(d_X1 + i, ldx1 + i, incx1, stridex1, d_X2 + i, ldx2 + i, incx2, stridex2, d_C + i, ldc + i, n + i, dim, (Real)theta[0], (Real)theta[1], range, nugget_tag);
    }
}

// batched version
template <typename Real>
void Matern12_scaled_matcov_vbatched(
    Real** d_X1, const int* ldx1, int incx1, int stridex1,
    Real** d_X2, const int* ldx2, int incx2, int stridex2,
    Real** d_C, const int* ldc, const int* n, int dim, const std::vector<double> &theta,
    const Real* range, bool nugget_tag, 
    int max_ldx1, int max_ldx2,
    int batchCount, cudaStream_t stream) {
    // Launch kernel for each single batch
    dim3 blockDim(THREAD_X, THREAD_Y, 1);
    const int gridx = ((max_ldx1 + blockDim.x - 1) / blockDim.x);
    const int gridy = ((max_ldx2 + blockDim.y - 1) / blockDim.y);
    // theta[0]: variance, theta[1]: nugget, theta[2:]: range
    // for loop over each batch
    const int max_batch = 65535;
    for (int i = 0; i < batchCount; i+= max_batch) {
        int gridz = min(max_batch, batchCount - i);
        dim3 gridDim(gridx, gridy, gridz);
        Matern12_scaled_matcov_vbatched_kernel<Real><<<gridDim, blockDim, 0, stream>>>(d_X1 + i, ldx1 + i, incx1, stridex1, d_X2 + i, ldx2 + i, incx2, stridex2, d_C + i, ldc + i, n + i, dim, (Real)theta[0], (Real)theta[1], range, nugget_tag);
    }
}

template <typename Real>
void Matern12_scaled_matcov(const Real* d_X1, int ldx1, int incx1, int stridex1,
                const Real* d_X2, int ldx2, int incx2, int stridex2,
                Real* d_C, int ldc, int n, int dim, const std::vector<double> &theta,
                const Real* range, bool nugget_tag,
                cudaStream_t stream) {
    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((ldx1 + blockDim.x - 1) / blockDim.x, (ldx2 + blockDim.y - 1) / blockDim.y);
    // theta[0]: variance, theta[1]: nugget, theta[2:]: range
    Matern12_scaled_matcov_kernel<Real><<<gridDim, blockDim, 0, stream>>>(d_X1, ldx1, incx1, stridex1, d_X2, ldx2, incx2, stridex2, d_C, ldc, n, dim, (Real)theta[0], (Real)theta[1], range, nugget_tag);
}

template <typename Real>
void Matern32_scaled_matcov(const Real* d_X1, int ldx1, int incx1, int stridex1,
                const Real* d_X2, int ldx2, int incx2, int stridex2,
                Real* d_C, int ldc, int n, int dim, const std::vector<double> &theta,
                const Real* range, bool nugget_tag,
                cudaStream_t stream) {
    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((ldx1 + blockDim.x - 1) / blockDim.x, (ldx2 + blockDim.y - 1) / blockDim.y);
    // theta[0]: variance, theta[1]: nugget, theta[2:]: range
    Matern32_scaled_matcov_kernel<Real><<<gridDim, blockDim, 0, stream>>>(d_X1, ldx1, incx1, stridex1, d_X2, ldx2, incx2, stridex2, d_C, ldc, n, dim, (Real)theta[0], (Real)theta[1], range, nugget_tag);
}   

template <typename Real>
void Matern52_scaled_matcov(const Real* d_X1, int ldx1, int incx1, int stridex1,
                const Real* d_X2, int ldx2, int incx2, int stridex2,
                Real* d_C, int ldc, int n, int dim, const std::vector<double> &theta,
                const Real* range, bool nugget_tag,
                cudaStream_t stream) {
    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((ldx1 + blockDim.x - 1) / blockDim.x, (ldx2 + blockDim.y - 1) / blockDim.y);
    // theta[0]: variance, theta[1]: nugget, theta[2:]: range
    Matern52_scaled_matcov_kernel<Real><<<gridDim, blockDim, 0, stream>>>(d_X1, ldx1, incx1, stridex1, d_X2, ldx2, incx2, stridex2, d_C, ldc, n, dim, (Real)theta[0], (Real)theta[1], range, nugget_tag);
}   

template <typename Real>
void RBF_matcov(const Real* d_X1, int ldx1, int incx1, int stridex1,
                const Real* d_X2, int ldx2, int incx2, int stridex2,
                Real* d_C, int ldc, int n, int dim, const std::vector<double> &theta, bool nugget_tag,
                cudaStream_t stream) {
    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((ldx1 + blockDim.x - 1) / blockDim.x, (ldx2 + blockDim.y - 1) / blockDim.y);
    RBF_matcov_kernel<Real><<<gridDim, blockDim, 0, stream>>>(d_X1, ldx1, incx1, stridex1, d_X2, ldx2, incx2, stridex2, d_C, ldc, n, dim, (Real)theta[0], (Real)theta[1], (Real)theta[2], nugget_tag);
}

template <typename Real>
__global__ void norm2_batch_kernel(const int* lda, const Real* const* d_A_array, const int* ldda, int batchCount, Real* norm2_results) {
    int batch_id = blockIdx.x;
    if (batch_id >= batchCount) return;

    int n = lda[batch_id];
    const Real* d_A = d_A_array[batch_id];

    __shared__ Real shared_sum[THREADS_PER_BLOCK];
    Real thread_sum = 0.0;

    // Each thread handles one element if possible
    if (threadIdx.x < n) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            Real val = d_A[i];
            thread_sum += val * val;
        }
    }
    shared_sum[threadIdx.x] = thread_sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = THREADS_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        norm2_results[batch_id] = shared_sum[0];
    }
}

template <typename Real>
Real norm2_batch(const int* d_lda, const Real* const* d_A_array, const int* d_ldda, int batchCount, cudaStream_t stream) {
    Real* d_norm2_results;
    cudaMalloc(&d_norm2_results, std::min(batchCount, BATCHCOUNT_MAX) * sizeof(Real));

    Real total_norm2 = 0.0;
    int remaining = batchCount;
    int offset = 0;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    while (remaining > 0) {
        int current_batch = std::min(remaining, BATCHCOUNT_MAX);

        dim3 gridDim(current_batch);
        dim3 blockDim(THREADS_PER_BLOCK);

        norm2_batch_kernel<Real><<<gridDim, blockDim, 0, stream>>>(
            d_lda + offset, 
            d_A_array + offset, 
            d_ldda + offset, 
            current_batch, 
            d_norm2_results
        );

        // Use thrust to sum up the results on the GPU
        thrust::device_ptr<Real> dev_ptr(d_norm2_results);
        Real batch_norm2 = thrust::reduce(thrust::cuda::par.on(stream), dev_ptr, dev_ptr + current_batch);
        total_norm2 += batch_norm2;

        remaining -= current_batch;
        offset += current_batch;
    }

    cudaFree(d_norm2_results);

    return total_norm2;
}

template <typename Real>
__global__ void log_det_batch_kernel(const int* lda, const Real* const* d_A_array, const int* ldda, int batchCount, Real* log_det_results) {
    int batch_id = blockIdx.x;
    if (batch_id >= batchCount) return;

    int n = lda[batch_id];
    int ldda_matrix = ldda[batch_id];
    const Real* d_A = d_A_array[batch_id];

    __shared__ Real shared_sum[THREADS_PER_BLOCK];
    Real thread_sum = 0.0;

    // Each thread handles one element if possible
    if (threadIdx.x < n) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            Real val = d_A[i * ldda_matrix + i];
            thread_sum += (Real)2 * log(val);
        }
    }

    shared_sum[threadIdx.x] = thread_sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = THREADS_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        log_det_results[batch_id] = shared_sum[0];
    }
}

template <typename Real>
Real log_det_batch(const int* d_lda, const Real* const* d_A_array, const int* d_ldda, int batchCount, cudaStream_t stream) {
    Real* d_log_det_results;
    cudaMalloc(&d_log_det_results, std::min(batchCount, BATCHCOUNT_MAX) * sizeof(Real));

    Real total_log_det = 0.0;
    int remaining = batchCount;
    int offset = 0;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    while (remaining > 0) {
        int current_batch = std::min(remaining, BATCHCOUNT_MAX);

        dim3 gridDim(current_batch);
        dim3 blockDim(THREADS_PER_BLOCK);
         
        // 2 * \sum_{i=1}^{#Bi} log(L_ii)
        log_det_batch_kernel<Real><<<gridDim, blockDim, 0, stream>>>(
            d_lda + offset, 
            d_A_array + offset, 
            d_ldda + offset, 
            current_batch, 
            d_log_det_results
        );

        // Use thrust to sum up the results on the GPU
        // \sum_{i=1}^{batchCount} log |\Sigma_i|
        thrust::device_ptr<Real> dev_ptr(d_log_det_results);
        Real batch_log_det = thrust::reduce(thrust::cuda::par.on(stream), dev_ptr, dev_ptr + current_batch);
        total_log_det += batch_log_det;

        remaining -= current_batch;
        offset += current_batch;
    }

    cudaFree(d_log_det_results);

    return total_log_det;
}

template <typename Real>
__global__ void generate_normal_kernel(Real *data, int n, Real mean, Real stddev, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize cuRAND state
    curandState state;
    curand_init(seed, idx, 0, &state);

    // Generate normally distributed random numbers
    if (idx < n) {
        if constexpr (std::is_same<Real, float>::value) {
            data[idx] = curand_normal(&state) * stddev + mean;
        } else {
            data[idx] = curand_normal_double(&state) * stddev + mean;
        }
    }
}

template <typename Real>
void generate_normal(Real *data, int n, Real mean, Real stddev, unsigned long long seed, cudaStream_t stream) {
    // Launch kernel
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    generate_normal_kernel<Real><<<gridDim, blockDim, 0, stream>>>(data, n, mean, stddev, seed);
}

// example of how to use kernel types
template <typename Real>
void compute_covariance(const Real* d_X1, int ldx1, int incx1, int stridex1,
                      const Real* d_X2, int ldx2, int incx2, int stridex2,
                      Real* d_C, int ldc, int n, int dim, 
                      const std::vector<double> &theta, const Real* range,
                      bool nugget_tag,
                      cudaStream_t stream, const Opts &opts) {
    switch (opts.kernel_type) {
        case KernelType::PowerExponential:
            PowerExp_scaled_matcov<Real>(
                d_X1, ldx1, incx1, stridex1,
                d_X2, ldx2, incx2, stridex2,
                d_C, ldc, n, dim, 
                theta, range, nugget_tag,
                stream);
            break;
        case KernelType::Matern72:
            Matern72_scaled_matcov<Real>(
                d_X1, ldx1, incx1, stridex1,
                d_X2, ldx2, incx2, stridex2,
                d_C, ldc, n, dim, 
                theta, range, nugget_tag,
                stream);
            break;
        case KernelType::Matern52:
            Matern52_scaled_matcov<Real>(
                d_X1, ldx1, incx1, stridex1,
                d_X2, ldx2, incx2, stridex2,
                d_C, ldc, n, dim, 
                theta, range, nugget_tag,
                stream);
            break;
        case KernelType::Matern12:
            Matern12_scaled_matcov<Real>(
                d_X1, ldx1, incx1, stridex1,
                d_X2, ldx2, incx2, stridex2,
                d_C, ldc, n, dim, 
                theta, range, nugget_tag,
                stream);
            break;
        case KernelType::Matern32:
            Matern32_scaled_matcov<Real>(
                d_X1, ldx1, incx1, stridex1,
                d_X2, ldx2, incx2, stridex2,
                d_C, ldc, n, dim, 
                theta, range, nugget_tag,
                stream);
            break;
        default:
            throw std::runtime_error("Unsupported kernel type");
            break;
    }
}

template <typename Real>
void compute_covariance_vbatched(Real** d_X1, const int* ldx1, int incx1, int stridex1,
                      Real** d_X2, const int* ldx2, int incx2, int stridex2,
                      Real** d_C, const int* ldc, const int* n, 
                      int batchCount,
                      int dim, const std::vector<double> &theta, const Real* range,
                      bool nugget_tag,
                      cudaStream_t stream, const Opts &opts) {
    // Find max of ldx1 and ldx2 using Thrust
    thrust::device_ptr<const int> d_ldx1(ldx1);
    thrust::device_ptr<const int> d_ldx2(ldx2);
    
    int max_ldx1 = thrust::reduce(thrust::cuda::par.on(stream), d_ldx1, d_ldx1 + batchCount, 0, thrust::maximum<int>());
    int max_ldx2 = thrust::reduce(thrust::cuda::par.on(stream), d_ldx2, d_ldx2 + batchCount, 0, thrust::maximum<int>());

    switch (opts.kernel_type) {
        case KernelType::Matern12:
            // std::cout << "Matern12 batched" << std::endl;
            Matern12_scaled_matcov_vbatched<Real>(
                d_X1, ldx1, incx1, stridex1,
                d_X2, ldx2, incx2, stridex2,
                d_C, ldc, n, dim, 
                theta, range, nugget_tag,
                max_ldx1, max_ldx2, batchCount,
                stream);
            break;
        case KernelType::Matern32:
            // std::cout << "Matern32 batched" << std::endl;
            Matern32_scaled_matcov_vbatched<Real>(
                d_X1, ldx1, incx1, stridex1,
                d_X2, ldx2, incx2, stridex2,
                d_C, ldc, n, dim, 
                theta, range, nugget_tag,
                max_ldx1, max_ldx2, batchCount,
                stream);
            break;
        case KernelType::Matern52:
            // std::cout << "Matern52 batched" << std::endl;
            Matern52_scaled_matcov_vbatched<Real>(
                d_X1, ldx1, incx1, stridex1,
                d_X2, ldx2, incx2, stridex2,
                d_C, ldc, n, dim, 
                theta, range, nugget_tag,
                max_ldx1, max_ldx2, batchCount,
                stream);
            break;
        case KernelType::Matern72:
            // std::cout << "Matern72 batched" << std::endl;
            Matern72_scaled_matcov_vbatched<Real>(
                d_X1, ldx1, incx1, stridex1,
                d_X2, ldx2, incx2, stridex2,
                d_C, ldc, n, dim, 
                theta, range, nugget_tag,
                max_ldx1, max_ldx2, batchCount,
                stream);
            break;
        default:
            throw std::runtime_error("Unsupported kernel type");
            break;
    }
}

// Batched matrix-matrix addition kernel
template <typename Real>
__global__ void batched_matrix_add_kernel(
    Real** d_A_array, const int* ldda_A,
    Real** d_B_array, const int* lda, const int* ldda_B,
    Real alpha, int batchCount) {
    
    int batch_id = blockIdx.z;
    if (batch_id >= batchCount) return;
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    int m = lda[batch_id];
    int n = lda[batch_id];
    int ldda_A_matrix = ldda_A[batch_id];
    int ldda_B_matrix = ldda_B[batch_id];
    
    Real* d_A = d_A_array[batch_id];
    Real* d_B = d_B_array[batch_id];
    
    if (i < m && j < n) {
        d_A[i + j * ldda_A_matrix] += alpha * d_B[i + j * ldda_B_matrix];
    }
}

// Batched matrix-vector addition kernel
template <typename Real>
__global__ void batched_vector_add_kernel(
    Real** d_A_array, const int* ldda_A,
    Real** d_B_array, const int* lda, const int* ldda_B,
    Real alpha, int batchCount) {
    
    int batch_id = blockIdx.x;
    if (batch_id >= batchCount) return;
    
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    int m = lda[batch_id];
    
    Real* d_A = d_A_array[batch_id];
    Real* d_B = d_B_array[batch_id];
    
    if (i < m) {
        d_A[i] += alpha * d_B[i];
    }
}

// Batched matrix-matrix addition wrapper function
template <typename Real>
void batched_matrix_add(
    Real** d_A_array, const int* ldda_A,
    Real** d_B_array, const int* lda, const int* ldda_B,
    Real alpha, int batchCount, cudaStream_t stream) {
    
    if (batchCount <= 0) return;
    
    // Find max dimensions for grid sizing
    thrust::device_ptr<const int> d_lda_ptr(lda);
    
    int max_lda = thrust::reduce(thrust::cuda::par.on(stream), d_lda_ptr, d_lda_ptr + batchCount, 0, thrust::maximum<int>());
  
    dim3 blockDim(THREAD_X, THREAD_Y, 1);
    const int gridx = (max_lda + blockDim.x - 1) / blockDim.x;
    const int gridy = (max_lda + blockDim.y - 1) / blockDim.y;
    
    // Process batches in chunks to respect the 65,535 grid size limit
    const int max_batch = 65535;
    for (int i = 0; i < batchCount; i += max_batch) {
        int gridz = min(max_batch, batchCount - i);
        dim3 gridDim(gridx, gridy, gridz);
        
        batched_matrix_add_kernel<Real><<<gridDim, blockDim, 0, stream>>>(
            d_A_array + i, ldda_A + i,
            d_B_array + i, lda + i, ldda_B + i,
            alpha, gridz);
    }
}

// Batched matrix-vector addition wrapper function
template <typename Real>
void batched_vector_add(
    Real** d_A_array, const int* ldda_A,
    Real** d_B_array, const int* lda, const int* ldda_B,
    Real alpha, int batchCount, cudaStream_t stream) {
    
    if (batchCount <= 0) return;
    
    // Find max dimension for grid sizing
    thrust::device_ptr<const int> d_lda_ptr(lda);
    int max_lda = thrust::reduce(thrust::cuda::par.on(stream), d_lda_ptr, d_lda_ptr + batchCount, 0, thrust::maximum<int>());
    
    dim3 blockDim(1, THREADS_PER_BLOCK, 1);
    const int gridy = (max_lda + blockDim.y - 1) / blockDim.y;
    
    // Process batches in chunks to respect the 65,535 grid size limit
    const int max_batch = 65535;
    for (int i = 0; i < batchCount; i += max_batch) {
        int gridx = min(max_batch, batchCount - i);
        dim3 gridDim(gridx, gridy, 1);
        
        batched_vector_add_kernel<Real><<<gridDim, blockDim, 0, stream>>>(
            d_A_array + i, ldda_A + i,
            d_B_array + i, lda + i, ldda_B + i,
            alpha, gridx);
    }
}

template void Matern_matcov<float>(const float*, int, int, int, const float*, int, int, int, float*, int, int, int, const std::vector<double>&, bool, cudaStream_t);
template void Matern_matcov<double>(const double*, int, int, int, const double*, int, int, int, double*, int, int, int, const std::vector<double>&, bool, cudaStream_t);
template void PowerExp_scaled_matcov<float>(const float*, int, int, int, const float*, int, int, int, float*, int, int, int, const std::vector<double>&, const float*, bool, cudaStream_t);
template void PowerExp_scaled_matcov<double>(const double*, int, int, int, const double*, int, int, int, double*, int, int, int, const std::vector<double>&, const double*, bool, cudaStream_t);
template void Matern72_scaled_matcov<float>(const float*, int, int, int, const float*, int, int, int, float*, int, int, int, const std::vector<double>&, const float*, bool, cudaStream_t);
template void Matern72_scaled_matcov<double>(const double*, int, int, int, const double*, int, int, int, double*, int, int, int, const std::vector<double>&, const double*, bool, cudaStream_t);
template void Matern12_scaled_matcov<float>(const float*, int, int, int, const float*, int, int, int, float*, int, int, int, const std::vector<double>&, const float*, bool, cudaStream_t);
template void Matern12_scaled_matcov<double>(const double*, int, int, int, const double*, int, int, int, double*, int, int, int, const std::vector<double>&, const double*, bool, cudaStream_t);
template void Matern32_scaled_matcov<float>(const float*, int, int, int, const float*, int, int, int, float*, int, int, int, const std::vector<double>&, const float*, bool, cudaStream_t);
template void Matern32_scaled_matcov<double>(const double*, int, int, int, const double*, int, int, int, double*, int, int, int, const std::vector<double>&, const double*, bool, cudaStream_t);
template void Matern52_scaled_matcov<float>(const float*, int, int, int, const float*, int, int, int, float*, int, int, int, const std::vector<double>&, const float*, bool, cudaStream_t);
template void Matern52_scaled_matcov<double>(const double*, int, int, int, const double*, int, int, int, double*, int, int, int, const std::vector<double>&, const double*, bool, cudaStream_t);
template void RBF_matcov<float>(const float*, int, int, int, const float*, int, int, int, float*, int, int, int, const std::vector<double>&, bool, cudaStream_t);
template void RBF_matcov<double>(const double*, int, int, int, const double*, int, int, int, double*, int, int, int, const std::vector<double>&, bool, cudaStream_t);
template float norm2_batch<float>(const int*, const float* const*, const int*, int, cudaStream_t);
template double norm2_batch<double>(const int*, const double* const*, const int*, int, cudaStream_t);
template float log_det_batch<float>(const int*, const float* const*, const int*, int, cudaStream_t);
template double log_det_batch<double>(const int*, const double* const*, const int*, int, cudaStream_t);
template void generate_normal<float>(float*, int, float, float, unsigned long long, cudaStream_t);
template void generate_normal<double>(double*, int, double, double, unsigned long long, cudaStream_t);
template void compute_covariance<float>(const float*, int, int, int, const float*, int, int, int, float*, int, int, int, const std::vector<double>&, const float*, bool, cudaStream_t, const Opts&);
template void compute_covariance<double>(const double*, int, int, int, const double*, int, int, int, double*, int, int, int, const std::vector<double>&, const double*, bool, cudaStream_t, const Opts&);
template void compute_covariance_vbatched<float>(float**, const int*, int, int, float**, const int*, int, int, float**, const int*, const int*, int, int, const std::vector<double>&, const float*, bool, cudaStream_t, const Opts&);
template void compute_covariance_vbatched<double>(double**, const int*, int, int, double**, const int*, int, int, double**, const int*, const int*, int, int, const std::vector<double>&, const double*, bool, cudaStream_t, const Opts&);
// Do not explicitly instantiate __global__ kernels with template syntax; only host wrappers need instantiation.
template void batched_matrix_add<float>(float**, const int*, float**, const int*, const int*, float, int, cudaStream_t);
template void batched_matrix_add<double>(double**, const int*, double**, const int*, const int*, double, int, cudaStream_t);
template void batched_vector_add<float>(float**, const int*, float**, const int*, const int*, float, int, cudaStream_t);
template void batched_vector_add<double>(double**, const int*, double**, const int*, const int*, double, int, cudaStream_t);
