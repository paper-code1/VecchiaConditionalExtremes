#ifndef GPU_COVARIANCE_H
#define GPU_COVARIANCE_H

#include <cuda_runtime.h>
#include <vector>
// Function declaration for GPU covariance matrix generation
__global__ void covarianceMatrixKernel(const double* X1, int ldx1, int incx1,
                                       const double* X2, int ldx2, int incx2,
                                       double* C, int ldc, int n, double sigma2, double range);

__global__ void covarianceMatrixKernel_v1(const double* X1, const double* Y1, int ldx1, int incx1,
                                         const double* X2, const double* Y2, int ldx2, int incx2,
                                         double* C, int ldc, int n, double sigma2, double range);

// Host function to launch the kernel
void covarianceMatern1_2(const double* d_X1, int ldx1, int incx1,
                         const double* d_X2, int ldx2, int incx2,
                         double* d_C, int ldc, int n, const std::vector<double> &theta,
                         cudaStream_t stream);

// (coalesced memory access)
void covarianceMatern1_2_v1(const double* d_X1, const double* d_Y1, int ldx1, int incx1,
                         const double* d_X2, const double* d_Y2, int ldx2, int incx2,
                         double* d_C, int ldc, int n, const std::vector<double> &theta,
                         cudaStream_t stream);


__global__ void norm2_batch_kernel(
    const int* d_lda, const double* const* d_A_array, 
    const int* d_ldda, int batchCount, double* d_norm2_results);

double norm2_batch(
    const int* d_lda, const double* const* d_A_array, 
    const int* d_ldda, int batchCount, cudaStream_t stream);

__global__ void log_det_batch_kernel(
    const int* d_lda, const double* const* d_A_array, 
    const int* d_ldda, int batchCount, double* d_log_det_results);

double log_det_batch(
    const int* d_lda, const double* const* d_A_array, 
    const int* d_ldda, int batchCount, cudaStream_t stream);

#endif // GPU_COVARIANCE_H