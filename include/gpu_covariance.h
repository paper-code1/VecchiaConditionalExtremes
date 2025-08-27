#ifndef GPU_COVARIANCE_H
#define GPU_COVARIANCE_H

#include <cuda_runtime.h>
#include <vector>
#include "input_parser_helper.h"
// Templated function declarations for GPU covariance matrix generation
template <typename Real>
__global__ void RBF_matcov_kernel(const Real* X1, int ldx1, int incx1, int stridex1,
                                         const Real* X2, int ldx2, int incx2, int stridex2,
                                         Real* C, int ldc, int n, int dim, 
                                         Real sigma2, Real range, Real nugget, bool nugget_tag);

template <typename Real>
__global__ void PowerExp_matcov_scaled_kernel(const Real* X1, int ldx1, int incx1, int stridex1,
                                         const Real* X2, int ldx2, int incx2, int stridex2,
                                         Real* C, int ldc, int n, int dim, 
                                         Real sigma2, Real smoothness, Real nugget, 
                                         const Real* range, bool nugget_tag);

template <typename Real>
__global__ void Matern72_matcov_kernel(const Real* X1, int ldx1, int incx1, int stridex1,
                                          const Real* X2, int ldx2, int incx2, int stridex2,
                                          Real* C, int ldc, int n, int dim, 
                                          Real sigma2, Real range, 
                                          Real nugget, bool nugget_tag);

template <typename Real>
__global__ void Matern52_matcov_kernel(const Real* X1, int ldx1, int incx1, int stridex1,
                                          const Real* X2, int ldx2, int incx2, int stridex2,
                                          Real* C, int ldc, int n, int dim, 
                                          Real sigma2, Real range, 
                                          Real nugget, bool nugget_tag);

template <typename Real>
__global__ void Matern32_matcov_kernel(const Real* X1, int ldx1, int incx1, int stridex1,
                                          const Real* X2, int ldx2, int incx2, int stridex2,
                                          Real* C, int ldc, int n, int dim, 
                                          Real sigma2, Real range, 
                                          Real nugget, bool nugget_tag);

template <typename Real>
__global__ void Matern12_matcov_kernel(const Real* X1, int ldx1, int incx1, int stridex1,
                                          const Real* X2, int ldx2, int incx2, int stridex2,
                                          Real* C, int ldc, int n, int dim, 
                                          Real sigma2, Real range, 
                                          Real nugget, bool nugget_tag);

template <typename Real>
void Matern_matcov(const Real* d_X1, int ldx1, int incx1, int stridex1,
                const Real* d_X2, int ldx2, int incx2, int stridex2,
                Real* d_C, int ldc, int n, int dim, 
                const std::vector<double> &theta, bool nugget_tag, 
                cudaStream_t stream);

template <typename Real>
void PowerExp_scaled_matcov(const Real* d_X1, int ldx1, int incx1, int stridex1,
                     const Real* d_X2, int ldx2, int incx2, int stridex2,
                     Real* d_C, int ldc, int n, int dim, 
                     const std::vector<double> &theta, const Real* range, 
                     bool nugget_tag,
                     cudaStream_t stream);

// Host function to launch the kernel
// (coalesced memory access)
template <typename Real>
void RBF_matcov(const Real* d_X1, int ldx1, int incx1, int stridex1,
                const Real* d_X2, int ldx2, int incx2, int stridex2,
                Real* d_C, int ldc, int n, int dim, const std::vector<double> &theta, bool nugget_tag, cudaStream_t stream);

// example of how to use kernel types
template <typename Real>
void compute_covariance(const Real* d_X1, int ldx1, int incx1, int stridex1,
                      const Real* d_X2, int ldx2, int incx2, int stridex2,
                      Real* d_C, int ldc, int n, int dim, 
                      const std::vector<double> &theta, const Real* range,
                      bool nugget_tag,
                      cudaStream_t stream, const Opts &opts);

template <typename Real>
void compute_covariance_vbatched(Real** d_X1, const int* ldx1, int incx1, int stridex1,
                      Real** d_X2, const int* ldx2, int incx2, int stridex2,
                      Real** d_C, const int* ldc, const int* n, 
                      int batchCount,
                      int dim, const std::vector<double> &theta, const Real* range,
                      bool nugget_tag,
                      cudaStream_t stream, const Opts &opts);


template <typename Real>
__global__ void norm2_batch_kernel(
    const int* d_lda, const Real* const* d_A_array, 
    const int* d_ldda, int batchCount, Real* d_norm2_results);

template <typename Real>
Real norm2_batch(
    const int* d_lda, const Real* const* d_A_array, 
    const int* d_ldda, int batchCount, cudaStream_t stream);

template <typename Real>
__global__ void log_det_batch_kernel(
    const int* d_lda, const Real* const* d_A_array, 
    const int* d_ldda, int batchCount, Real* d_log_det_results);

template <typename Real>
Real log_det_batch(
    const int* d_lda, const Real* const* d_A_array, 
    const int* d_ldda, int batchCount, cudaStream_t stream);

template <typename Real>
__global__ void generate_normal_kernel(Real *data, int n, Real mean, Real stddev, unsigned long long seed);

template <typename Real>
void generate_normal(Real *data, int n, Real mean, Real stddev, unsigned long long seed, cudaStream_t stream);

// Batched matrix addition kernels
template <typename Real>
__global__ void batched_matrix_add_kernel(
    Real** d_A_array, const int* ldda_A,
    Real** d_B_array, const int* lda, const int* ldda_B,
    Real alpha, int batchCount);

template <typename Real>
__global__ void batched_vector_add_kernel(
    Real** d_A_array, const int* ldda_A,
    Real** d_B_array, const int* lda, const int* ldda_B,
    Real alpha, int batchCount);

template <typename Real>
void batched_matrix_add(
    Real** d_A_array, const int* ldda_A,
    Real** d_B_array, const int* lda, const int* ldda_B,
    Real alpha, int batchCount, cudaStream_t stream);

template <typename Real>
void batched_vector_add(
    Real** d_A_array, const int* ldda_A,
    Real** d_B_array, const int* lda, const int* ldda_B,
    Real alpha, int batchCount, cudaStream_t stream);

#endif // GPU_COVARIANCE_H