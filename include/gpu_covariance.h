#ifndef GPU_COVARIANCE_H
#define GPU_COVARIANCE_H

#include <cuda_runtime.h>
#include <vector>
#include "input_parser_helper.h"
// Function declaration for GPU covariance matrix generation
__global__ void RBF_matcov_kernel(const double* X1, int ldx1, int incx1, int stridex1,
                                         const double* X2, int ldx2, int incx2, int stridex2,
                                         double* C, int ldc, int n, int dim, 
                                         double sigma2, double range, double nugget, bool nugget_tag);

__global__ void PowerExp_matcov_scaled_kernel(const double* X1, int ldx1, int incx1, int stridex1,
                                         const double* X2, int ldx2, int incx2, int stridex2,
                                         double* C, int ldc, int n, int dim, 
                                         double sigma2, double smoothness, double nugget, 
                                         const double* range, bool nugget_tag);

__global__ void Matern72_matcov_kernel(const double* X1, int ldx1, int incx1, int stridex1,
                                          const double* X2, int ldx2, int incx2, int stridex2,
                                          double* C, int ldc, int n, int dim, 
                                          double sigma2, double range, 
                                          double nugget, bool nugget_tag);

void Matern_matcov(const double* d_X1, int ldx1, int incx1, int stridex1,
                const double* d_X2, int ldx2, int incx2, int stridex2,
                double* d_C, int ldc, int n, int dim, 
                const std::vector<double> &theta, bool nugget_tag, 
                cudaStream_t stream);

void PowerExp_scaled_matcov(const double* d_X1, int ldx1, int incx1, int stridex1,
                     const double* d_X2, int ldx2, int incx2, int stridex2,
                     double* d_C, int ldc, int n, int dim, 
                     const std::vector<double> &theta, const double* range, 
                     bool nugget_tag,
                     cudaStream_t stream);

// Host function to launch the kernel
// (coalesced memory access)
void RBF_matcov(const double* d_X1, int ldx1, int incx1, int stridex1,
                const double* d_X2, int ldx2, int incx2, int stridex2,
                double* d_C, int ldc, int n, int dim, const std::vector<double> &theta, bool nugget_tag, cudaStream_t stream);

// example of how to use kernel types
void compute_covariance(const double* d_X1, int ldx1, int incx1, int stridex1,
                      const double* d_X2, int ldx2, int incx2, int stridex2,
                      double* d_C, int ldc, int n, int dim, 
                      const std::vector<double> &theta, const double* range,
                      bool nugget_tag,
                      cudaStream_t stream, const Opts &opts);

void compute_covariance_vbatched(double** d_X1, const int* ldx1, int incx1, int stridex1,
                      double** d_X2, const int* ldx2, int incx2, int stridex2,
                      double** d_C, const int* ldc, const int* n, 
                      int batchCount,
                      int dim, const std::vector<double> &theta, const double* range,
                      bool nugget_tag,
                      cudaStream_t stream, const Opts &opts);


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

__global__ void generate_normal_kernel(double *data, int n, double mean, double stddev, unsigned long long seed);

void generate_normal(double *data, int n, double mean, double stddev, unsigned long long seed, cudaStream_t stream);

// Batched matrix addition kernels
__global__ void batched_matrix_add_kernel(
    double** d_A_array, const int* ldda_A,
    double** d_B_array, const int* lda, const int* ldda_B,
    double alpha, int batchCount);

__global__ void batched_vector_add_kernel(
    double** d_A_array, const int* ldda_A,
    double** d_B_array, const int* lda, const int* ldda_B,
    double alpha, int batchCount);

void batched_matrix_add(
    double** d_A_array, const int* ldda_A,
    double** d_B_array, const int* lda, const int* ldda_B,
    double alpha, int batchCount, cudaStream_t stream);

void batched_vector_add(
    double** d_A_array, const int* ldda_A,
    double** d_B_array, const int* lda, const int* ldda_B,
    double alpha, int batchCount, cudaStream_t stream);

#endif // GPU_COVARIANCE_H