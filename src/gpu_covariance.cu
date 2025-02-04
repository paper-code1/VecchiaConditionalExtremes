#include <cmath>
#include <iostream>
#include <curand_kernel.h>
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
                                          double* C, int ldc, int n, int dim, 
                                          double sigma2, double range, double nugget, 
                                          bool nugget_tag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < ldx1 && j < ldx2 && i >= 0 && j >= 0) {
        double dist_sqaure = 0;
        for (int k = 0; k < dim; k++) {
            double x1 = X1[i * incx1 + k * stridex1];
            double x2 = X2[j * incx2 + k * stridex2];
            dist_sqaure += (x1 - x2) * (x1 - x2);
        }
        double scaled_distance = sqrt(dist_sqaure) / range;
        C[i + j * ldc] = sigma2 * exp( - scaled_distance );
    }
    // add nugget
    if (i == j && i < ldx1 && j < ldx2 && nugget_tag) {
        C[i + j * ldc] += nugget;
    }
}

// (coalesced memory access)
__global__ void PowerExp_matcov_scaled_kernel(const double* X1, int ldx1, int incx1, int stridex1,
                                          const double* X2, int ldx2, int incx2, int stridex2,
                                          double* C, int ldc, int n, int dim, 
                                          double sigma2, double smoothness, double nugget, 
                                          const double* range, bool nugget_tag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < ldx1 && j < ldx2 && i >= 0 && j >= 0) {
        double dist_sqaure = 0;
        for (int k = 0; k < dim; k++) {
            double x1 = X1[i * incx1 + k * stridex1];
            double x2 = X2[j * incx2 + k * stridex2];
            dist_sqaure += (x1 - x2) * (x1 - x2) / range[k] / range[k];
        }
        double scaled_distance = sqrt(dist_sqaure);
        double power_distance = pow(scaled_distance, smoothness);
        C[i + j * ldc] = sigma2 * exp( - power_distance );
    }
    // add nugget
    if (i == j && i < ldx1 && j < ldx2 && nugget_tag) {
        C[i + j * ldc] += nugget;
    }
}

__global__ void Matern72_scaled_matcov_kernel(const double* X1, int ldx1, int incx1, int stridex1,
                                          const double* X2, int ldx2, int incx2, int stridex2,
                                          double* C, int ldc, int n, int dim, 
                                          double sigma2, double nugget, 
                                          const double* range, bool nugget_tag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < ldx1 && j < ldx2 && i >= 0 && j >= 0) {
        double dist_sqaure = 0;
        for (int k = 0; k < dim; k++) {
            double x1 = X1[i * incx1 + k * stridex1];
            double x2 = X2[j * incx2 + k * stridex2];
            dist_sqaure += (x1 - x2) * (x1 - x2) / range[k] / range[k];
        }
        double scaled_distance = sqrt(dist_sqaure);
        double a0 = 1.0;
        double a1 = 1.0;
        double a2 = 2.0 / 5.0;
        double a3 = 1.0 / 15.0;
        double item_poly = a0 + a1 * scaled_distance + a2 * scaled_distance * scaled_distance + a3 * scaled_distance * scaled_distance * scaled_distance;
        C[i + j * ldc] = sigma2 * item_poly * exp( - scaled_distance );
    }
    // add nugget
    if (i == j && i < ldx1 && j < ldx2 && nugget_tag) {
        C[i + j * ldc] += nugget;
    }
}

__global__ void Matern12_scaled_matcov_kernel(const double* X1, int ldx1, int incx1, int stridex1,
                                          const double* X2, int ldx2, int incx2, int stridex2,
                                          double* C, int ldc, int n, int dim, 
                                          double sigma2, double nugget, 
                                          const double* range, bool nugget_tag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < ldx1 && j < ldx2 && i >= 0 && j >= 0) {
        double dist_sqaure = 0;
        for (int k = 0; k < dim; k++) {
            double x1 = X1[i * incx1 + k * stridex1];
            double x2 = X2[j * incx2 + k * stridex2];
            dist_sqaure += (x1 - x2) * (x1 - x2) / range[k] / range[k];
        }
        double scaled_distance = sqrt(dist_sqaure);
        double a0 = 1.0;
        // double a1 = 1.0;
        // double a2 = 2.0 / 5.0;
        // double a3 = 1.0 / 15.0;
        double item_poly = a0;
        // + a1 * scaled_distance + a2 * scaled_distance * scaled_distance + a3 * scaled_distance * scaled_distance * scaled_distance;
        C[i + j * ldc] = sigma2 * item_poly * exp( - scaled_distance );
    }
    // add nugget
    if (i == j && i < ldx1 && j < ldx2 && nugget_tag) {
        C[i + j * ldc] += nugget;
    }
}

__global__ void Matern32_scaled_matcov_kernel(const double* X1, int ldx1, int incx1, int stridex1,
                                          const double* X2, int ldx2, int incx2, int stridex2,
                                          double* C, int ldc, int n, int dim, 
                                          double sigma2, double nugget, 
                                          const double* range, bool nugget_tag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < ldx1 && j < ldx2 && i >= 0 && j >= 0) {
        double dist_sqaure = 0;
        for (int k = 0; k < dim; k++) {
            double x1 = X1[i * incx1 + k * stridex1];
            double x2 = X2[j * incx2 + k * stridex2];
            dist_sqaure += (x1 - x2) * (x1 - x2) / range[k] / range[k];
        }
        double scaled_distance = sqrt(dist_sqaure);
        double a0 = 1.0;
        double a1 = 1.0;
        // double a2 = 2.0 / 5.0;
        // double a3 = 1.0 / 15.0;
        double item_poly = a0 + a1 * scaled_distance;
        //  + a2 * scaled_distance * scaled_distance + a3 * scaled_distance * scaled_distance * scaled_distance;
        C[i + j * ldc] = sigma2 * item_poly * exp( - scaled_distance );
    }
    // add nugget
    if (i == j && i < ldx1 && j < ldx2 && nugget_tag) {
        C[i + j * ldc] += nugget;
    }
}

__global__ void Matern52_scaled_matcov_kernel(const double* X1, int ldx1, int incx1, int stridex1,
                                          const double* X2, int ldx2, int incx2, int stridex2,
                                          double* C, int ldc, int n, int dim, 
                                          double sigma2, double nugget, 
                                          const double* range, bool nugget_tag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < ldx1 && j < ldx2 && i >= 0 && j >= 0) {
        double dist_sqaure = 0;
        for (int k = 0; k < dim; k++) {
            double x1 = X1[i * incx1 + k * stridex1];
            double x2 = X2[j * incx2 + k * stridex2];
            dist_sqaure += (x1 - x2) * (x1 - x2) / range[k] / range[k];
        }
        double scaled_distance = sqrt(dist_sqaure);
        double a0 = 1.0;
        double a1 = 1.0;
        double a2 = 1.0 / 3.0;
        double item_poly = a0 + a1 * scaled_distance + a2 * scaled_distance * scaled_distance;
        C[i + j * ldc] = sigma2 * item_poly * exp( - scaled_distance );
    }
    // add nugget
    if (i == j && i < ldx1 && j < ldx2 && nugget_tag) {
        C[i + j * ldc] += nugget;
    }
}

__global__ void Matern72_matcov_kernel(const double* X1, int ldx1, int incx1, int stridex1,
                                          const double* X2, int ldx2, int incx2, int stridex2,
                                          double* C, int ldc, int n, int dim, 
                                          double sigma2, double range, 
                                          double nugget, bool nugget_tag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < ldx1 && j < ldx2 && i >= 0 && j >= 0) {
        double dist_sqaure = 0;
        for (int k = 0; k < dim; k++) {
            double x1 = X1[i * incx1 + k * stridex1];
            double x2 = X2[j * incx2 + k * stridex2];
            dist_sqaure += (x1 - x2) * (x1 - x2);
        }
        double scaled_distance = sqrt(dist_sqaure) / range;
        double item_poly = 1 + sqrt(7.0) * scaled_distance + 7.0 * scaled_distance * scaled_distance / 3.0;  
        // + 7.0 * sqrt(7.0) * scaled_distance * scaled_distance * scaled_distance / 15.0; 
        C[i + j * ldc] = sigma2 * item_poly * exp( - sqrt(7.0) * scaled_distance );
    }
    // add nugget
    if (i == j && i < ldx1 && j < ldx2 && nugget_tag) {
        C[i + j * ldc] += nugget;
    }
}

void Matern_matcov(const double* d_X1, int ldx1, int incx1, int stridex1,
                const double* d_X2, int ldx2, int incx2, int stridex2,
                double* d_C, int ldc, int n, int dim, const std::vector<double> &theta, bool nugget_tag,
                cudaStream_t stream) {
    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((ldx1 + blockDim.x - 1) / blockDim.x, (ldx2 + blockDim.y - 1) / blockDim.y);
    // the smoothness = 3.5 theta[0]: variance, theta[1]: range, theta[3]: nugget
    Matern72_matcov_kernel<<<gridDim, blockDim, 0, stream>>>(d_X1, ldx1, incx1, stridex1, d_X2, ldx2, incx2, stridex2, d_C, ldc, n, dim, theta[0], theta[1], theta[3], nugget_tag);
}

void PowerExp_scaled_matcov(const double* d_X1, int ldx1, int incx1, int stridex1,
                const double* d_X2, int ldx2, int incx2, int stridex2,
                double* d_C, int ldc, int n, int dim, const std::vector<double> &theta,
                const double* range, bool nugget_tag,
                cudaStream_t stream) {
    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((ldx1 + blockDim.x - 1) / blockDim.x, (ldx2 + blockDim.y - 1) / blockDim.y);
    // theta[0]: variance, theta[1]: smoothness, theta[2]: nugget, theta[3:]: range
    PowerExp_matcov_scaled_kernel<<<gridDim, blockDim, 0, stream>>>(d_X1, ldx1, incx1, stridex1, d_X2, ldx2, incx2, stridex2, d_C, ldc, n, dim, theta[0], theta[1], theta[2], range, nugget_tag);
}

void Matern72_scaled_matcov(const double* d_X1, int ldx1, int incx1, int stridex1,
                const double* d_X2, int ldx2, int incx2, int stridex2,
                double* d_C, int ldc, int n, int dim, const std::vector<double> &theta,
                const double* range, bool nugget_tag,
                cudaStream_t stream) {
    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((ldx1 + blockDim.x - 1) / blockDim.x, (ldx2 + blockDim.y - 1) / blockDim.y);
    // theta[0]: variance, theta[1]: nugget, theta[2:]: range
    Matern72_scaled_matcov_kernel<<<gridDim, blockDim, 0, stream>>>(d_X1, ldx1, incx1, stridex1, d_X2, ldx2, incx2, stridex2, d_C, ldc, n, dim, theta[0], theta[1], range, nugget_tag);
}

void Matern12_scaled_matcov(const double* d_X1, int ldx1, int incx1, int stridex1,
                const double* d_X2, int ldx2, int incx2, int stridex2,
                double* d_C, int ldc, int n, int dim, const std::vector<double> &theta,
                const double* range, bool nugget_tag,
                cudaStream_t stream) {
    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((ldx1 + blockDim.x - 1) / blockDim.x, (ldx2 + blockDim.y - 1) / blockDim.y);
    // theta[0]: variance, theta[1]: nugget, theta[2:]: range
    Matern12_scaled_matcov_kernel<<<gridDim, blockDim, 0, stream>>>(d_X1, ldx1, incx1, stridex1, d_X2, ldx2, incx2, stridex2, d_C, ldc, n, dim, theta[0], theta[1], range, nugget_tag);
}

void Matern32_scaled_matcov(const double* d_X1, int ldx1, int incx1, int stridex1,
                const double* d_X2, int ldx2, int incx2, int stridex2,
                double* d_C, int ldc, int n, int dim, const std::vector<double> &theta,
                const double* range, bool nugget_tag,
                cudaStream_t stream) {
    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((ldx1 + blockDim.x - 1) / blockDim.x, (ldx2 + blockDim.y - 1) / blockDim.y);
    // theta[0]: variance, theta[1]: nugget, theta[2:]: range
    Matern32_scaled_matcov_kernel<<<gridDim, blockDim, 0, stream>>>(d_X1, ldx1, incx1, stridex1, d_X2, ldx2, incx2, stridex2, d_C, ldc, n, dim, theta[0], theta[1], range, nugget_tag);
}   

void Matern52_scaled_matcov(const double* d_X1, int ldx1, int incx1, int stridex1,
                const double* d_X2, int ldx2, int incx2, int stridex2,
                double* d_C, int ldc, int n, int dim, const std::vector<double> &theta,
                const double* range, bool nugget_tag,
                cudaStream_t stream) {
    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((ldx1 + blockDim.x - 1) / blockDim.x, (ldx2 + blockDim.y - 1) / blockDim.y);
    // theta[0]: variance, theta[1]: nugget, theta[2:]: range
    Matern52_scaled_matcov_kernel<<<gridDim, blockDim, 0, stream>>>(d_X1, ldx1, incx1, stridex1, d_X2, ldx2, incx2, stridex2, d_C, ldc, n, dim, theta[0], theta[1], range, nugget_tag);
}   

void RBF_matcov(const double* d_X1, int ldx1, int incx1, int stridex1,
                const double* d_X2, int ldx2, int incx2, int stridex2,
                double* d_C, int ldc, int n, int dim, const std::vector<double> &theta, bool nugget_tag,
                cudaStream_t stream) {
    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((ldx1 + blockDim.x - 1) / blockDim.x, (ldx2 + blockDim.y - 1) / blockDim.y);
    RBF_matcov_kernel<<<gridDim, blockDim, 0, stream>>>(d_X1, ldx1, incx1, stridex1, d_X2, ldx2, incx2, stridex2, d_C, ldc, n, dim, theta[0], theta[1], theta[2], nugget_tag);
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

__global__ void generate_normal_kernel(double *data, int n, double mean, double stddev, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize cuRAND state
    curandState state;
    curand_init(seed, idx, 0, &state);

    // Generate normally distributed random numbers
    if (idx < n) {
        data[idx] = curand_normal(&state) * stddev + mean;  // Apply mean and stddev
    }
}

void generate_normal(double *data, int n, double mean, double stddev, unsigned long long seed, cudaStream_t stream) {
    // Launch kernel
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    generate_normal_kernel<<<gridDim, blockDim, 0, stream>>>(data, n, mean, stddev, seed);
}

// example of how to use kernel types
void compute_covariance(const double* d_X1, int ldx1, int incx1, int stridex1,
                      const double* d_X2, int ldx2, int incx2, int stridex2,
                      double* d_C, int ldc, int n, int dim, 
                      const std::vector<double> &theta, const double* range,
                      bool nugget_tag,
                      cudaStream_t stream, const Opts &opts) {
    switch (opts.kernel_type) {
        case KernelType::PowerExponential:
            PowerExp_scaled_matcov(
                d_X1, ldx1, incx1, stridex1,
                d_X2, ldx2, incx2, stridex2,
                d_C, ldc, n, dim, 
                theta, range, nugget_tag,
                stream);
            break;
        case KernelType::Matern72:
            Matern72_scaled_matcov(
                d_X1, ldx1, incx1, stridex1,
                d_X2, ldx2, incx2, stridex2,
                d_C, ldc, n, dim, 
                theta, range, nugget_tag,
                stream);
            break;
        case KernelType::Matern52:
            Matern52_scaled_matcov(
                d_X1, ldx1, incx1, stridex1,
                d_X2, ldx2, incx2, stridex2,
                d_C, ldc, n, dim, 
                theta, range, nugget_tag,
                stream);
            break;
        case KernelType::Matern12:
            Matern12_scaled_matcov(
                d_X1, ldx1, incx1, stridex1,
                d_X2, ldx2, incx2, stridex2,
                d_C, ldc, n, dim, 
                theta, range, nugget_tag,
                stream);
            break;
        case KernelType::Matern32:
            Matern32_scaled_matcov(
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
