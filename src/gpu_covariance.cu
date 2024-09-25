#include <cmath>
#include <iostream>
#include "gpu_covariance.h"
#include "block_info.h"

__device__ double squaredEuclideanDistance(const double* x1, const double* x2) {
    double dx = 0;
    for (int i = 0; i < DIMENSION; i++) {
        dx += (x1[i] - x2[i]) * (x1[i] - x2[i]);
    }
    return sqrt(dx);
}

__global__ void covarianceMatrixKernel(const double* X1, int ldx1, int incx1,
                                       const double* X2, int ldx2, int incx2,
                                       double* C, int ldc, int n, double sigma2, double range) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < ldx1 && j < ldx2 && i >= 0 && j >= 0) {
        const double* x1 = X1 + i * incx1 * DIMENSION;
        const double* x2 = X2 + j * incx2 * DIMENSION;
        double scaled_dist = squaredEuclideanDistance(x1, x2)/range;
        C[i + j * ldc] = sigma2 *  exp(-scaled_dist);
    }
}

// (coalesced memory access)
__global__ void covarianceMatrixKernel_v1(const double* X1, const double* Y1, int ldx1, int incx1,
                                          const double* X2, const double* Y2, int ldx2, int incx2,
                                          double* C, int ldc, int n, double sigma2, double range) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < ldx1 && j < ldx2 && i >= 0 && j >= 0) {
        double x1 = X1[i * incx1];
        double y1 = Y1[i * incx1];
        double x2 = X2[j * incx2];
        double y2 = Y2[j * incx2];
        double scaled_dist = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))/range;
        C[i + j * ldc] = sigma2 *  exp(-scaled_dist);
    }
}

void covarianceMatern1_2(const double* d_X1, int ldx1, int incx1,
                         const double* d_X2, int ldx2, int incx2,
                         double* d_C, int ldc, int n, const std::vector<double> &theta,
                         cudaStream_t stream) {
    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);
    covarianceMatrixKernel<<<gridDim, blockDim, 0, stream>>>(d_X1, ldx1, incx1, d_X2, ldx2, incx2, d_C, ldc, n, theta[0], theta[1]);
}

void covarianceMatern1_2_v1(const double* d_X1, const double* d_Y1, int ldx1, int incx1,
                            const double* d_X2, const double* d_Y2, int ldx2, int incx2,
                            double* d_C, int ldc, int n, const std::vector<double> &theta,
                            cudaStream_t stream) {
    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);
    covarianceMatrixKernel_v1<<<gridDim, blockDim, 0, stream>>>(d_X1, d_Y1, ldx1, incx1, d_X2, d_Y2, ldx2, incx2, d_C, ldc, n, theta[0], theta[1]);
}