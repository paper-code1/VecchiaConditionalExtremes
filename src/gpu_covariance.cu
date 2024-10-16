#include <cmath>
#include <iostream>
#include "gpu_covariance.h"
#include "block_info.h"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <mpi.h>
#define BATCHCOUNT_MAX 65536
#define THREADS_PER_BLOCK 64

// (coalesced memory access)
__global__ void RBF_matcov_kernel(const double* X1, int ldx1, int incx1, int stridex1,
                                          const double* X2, int ldx2, int incx2, int stridex2,
                                          double* C, int ldc, int n, int dim, double sigma2, double range, double nugget) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < ldx1 && j < ldx2 && i >= 0 && j >= 0) {
        double dist_sqaure = 0;
        for (int k = 0; k < dim; k++) {
            double x1 = X1[i * incx1 + k * stridex1];
            double x2 = X2[j * incx2 + k * stridex2];
            dist_sqaure += (x1 - x2) * (x1 - x2);
        }
        C[i + j * ldc] = sigma2 * exp( - sqrt(dist_sqaure) / range );
    }
    // add nugget
    if (i == j && i < ldx1 && j < ldx2) {
        C[i + j * ldc] += nugget;
    }
}

void RBF_matcov(const double* d_X1, int ldx1, int incx1, int stridex1,
                const double* d_X2, int ldx2, int incx2, int stridex2,
                double* d_C, int ldc, int n, int dim, const std::vector<double> &theta,
                cudaStream_t stream) {
    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((ldx1 + blockDim.x - 1) / blockDim.x, (ldx2 + blockDim.y - 1) / blockDim.y);
    RBF_matcov_kernel<<<gridDim, blockDim, 0, stream>>>(d_X1, ldx1, incx1, stridex1, d_X2, ldx2, incx2, stridex2, d_C, ldc, n, dim, theta[0], theta[1], theta[2]);
}

__global__ void norm2_batch_kernel(const int* lda, const double* const* d_A_array, const int* ldda, int batchCount, double* norm2_results) {
    int batch_id = blockIdx.x;
    if (batch_id >= batchCount) return;

    int n = lda[batch_id];
    const double* d_A = d_A_array[batch_id];

    __shared__ double shared_sum[THREADS_PER_BLOCK];
    double thread_sum = 0.0;

    // Each thread handles one element if possible
    if (threadIdx.x < n) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            double val = d_A[i];
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

double norm2_batch(const int* d_lda, const double* const* d_A_array, const int* d_ldda, int batchCount, cudaStream_t stream) {
    double* d_norm2_results;
    cudaMalloc(&d_norm2_results, std::min(batchCount, BATCHCOUNT_MAX) * sizeof(double));

    double total_norm2 = 0.0;
    int remaining = batchCount;
    int offset = 0;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    while (remaining > 0) {
        int current_batch = std::min(remaining, BATCHCOUNT_MAX);

        dim3 gridDim(current_batch);
        dim3 blockDim(THREADS_PER_BLOCK);

        norm2_batch_kernel<<<gridDim, blockDim, 0, stream>>>(
            d_lda + offset, 
            d_A_array + offset, 
            d_ldda + offset, 
            current_batch, 
            d_norm2_results
        );

        // Use thrust to sum up the results on the GPU
        thrust::device_ptr<double> dev_ptr(d_norm2_results);
        double batch_norm2 = thrust::reduce(thrust::cuda::par.on(stream), dev_ptr, dev_ptr + current_batch);
        total_norm2 += batch_norm2;

        remaining -= current_batch;
        offset += current_batch;
    }

    cudaFree(d_norm2_results);

    return total_norm2;
}

__global__ void log_det_batch_kernel(const int* lda, const double* const* d_A_array, const int* ldda, int batchCount, double* log_det_results) {
    int batch_id = blockIdx.x;
    if (batch_id >= batchCount) return;

    int n = lda[batch_id];
    int ldda_matrix = ldda[batch_id];
    const double* d_A = d_A_array[batch_id];

    __shared__ double shared_sum[THREADS_PER_BLOCK];
    double thread_sum = 0.0;

    // Each thread handles one element if possible
    if (threadIdx.x < n) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            double val = d_A[i * ldda_matrix + i];
            thread_sum += 2 * log(val);
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

double log_det_batch(const int* d_lda, const double* const* d_A_array, const int* d_ldda, int batchCount, cudaStream_t stream) {
    double* d_log_det_results;
    cudaMalloc(&d_log_det_results, std::min(batchCount, BATCHCOUNT_MAX) * sizeof(double));

    double total_log_det = 0.0;
    int remaining = batchCount;
    int offset = 0;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    while (remaining > 0) {
        int current_batch = std::min(remaining, BATCHCOUNT_MAX);

        dim3 gridDim(current_batch);
        dim3 blockDim(THREADS_PER_BLOCK);
         
        // 2 * \sum_{i=1}^{#Bi} log(L_ii)
        log_det_batch_kernel<<<gridDim, blockDim, 0, stream>>>(
            d_lda + offset, 
            d_A_array + offset, 
            d_ldda + offset, 
            current_batch, 
            d_log_det_results
        );

        // Use thrust to sum up the results on the GPU
        // \sum_{i=1}^{batchCount} log |\Sigma_i|
        thrust::device_ptr<double> dev_ptr(d_log_det_results);
        double batch_log_det = thrust::reduce(thrust::cuda::par.on(stream), dev_ptr, dev_ptr + current_batch);
        total_log_det += batch_log_det;

        remaining -= current_batch;
        offset += current_batch;
    }

    cudaFree(d_log_det_results);

    return total_log_det;
}