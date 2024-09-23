#ifndef GPU_OPERATIONS_H
#define GPU_OPERATIONS_H

#include <vector>
#include <utility>
#include <cuda_runtime.h>
#include <magma_v2.h>
#include "block_info.h"

// Define the GpuData structure to store GPU memory pointers and leading dimensions
struct GpuData
{
    double **h_points_array;
    double **h_nearestNeighbors_array;
    double **h_cov_array;
    double **h_cross_cov_array;
    double **h_conditioning_cov_array;
    std::vector<int> ldda_points;      // Leading dimensions for GPU points (varied)
    std::vector<int> ldda_neighbors;   // Leading dimension for GPU nearest neighbors (varied)
    std::vector<int> ldda_cov;
    std::vector<int> ldda_cross_cov;
    std::vector<int> ldda_conditioning_cov;
    std::vector<int> lda_points;      // leading dimension for CPU points (varied)
    std::vector<int> lda_neighbors;   // leading dimension for CPU points (varied)
    double *d_points_memory;           // Contiguous memory for points
    double *d_nearestNeighbors_memory; // Contiguous memory for nearest neighbors
    double *d_cov_memory;           // Contiguous memory for covariance matrix
    double *d_cross_cov_memory;           // Contiguous memory for cross covariance matrix
    double *d_conditioning_cov_memory;           // Contiguous memory for conditioning covariance matrix
};

// Function to copy data from CPU to GPU and allocate memory with leading dimensions
GpuData copyDataToGPU(int gpu_id, const std::vector<BlockInfo> &blockInfos);

// Function to perform computation on the GPU
void performComputationOnGPU(const GpuData &gpuData);

// Function to clean up GPU memory
void cleanupGpuMemory(GpuData &gpuData);

#endif // GPU_OPERATIONS_H
