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
    // locations 
    // double **h_locs_array;
    // double **h_locs_nearestNeighbors_array;
    double **h_locs_x_array;
    double **h_locs_y_array;
    double **h_locs_nearestNeighbors_x_array;
    double **h_locs_nearestNeighbors_y_array;
    // observations
    double **h_observations_array;
    double **h_observations_nearestNeighbors_array;
    // covariance matrix, used to store the pointer of contiguous memory
    double **h_cov_array;
    double **h_cross_cov_array;
    double **h_conditioning_cov_array;
    // leading dimensions for GPU points (varied)
    std::vector<int> ldda_locs;      // Leading dimensions for GPU points (varied)
    std::vector<int> ldda_neighbors;   // Leading dimension for GPU nearest neighbors (varied)
    std::vector<int> ldda_cov;
    std::vector<int> ldda_cross_cov;
    std::vector<int> ldda_conditioning_cov;
    std::vector<int> lda_locs;      // leading dimension for CPU points (varied)
    std::vector<int> lda_locs_neighbors;   // leading dimension for CPU points (varied)
    // contiguous memory for GPU points
    // double *d_locs_device;           // Contiguous memory for locations
    // double *d_locs_nearestNeighbors_device; // Contiguous memory for locations of nearest neighbors
    // only for 2D points (testing the coalesced memory access)
    double *d_locs_x_device; // Contiguous memory for locations of x
    double *d_locs_y_device; // Contiguous memory for locations of y
    double *d_locs_nearestNeighbors_x_device; // Contiguous memory for locations of x
    double *d_locs_nearestNeighbors_y_device; // Contiguous memory for locations of y
    double *d_observations_device;           // Contiguous memory for observations points
    double *d_observations_nearestNeighbors_device; // Contiguous memory for observations nearest neighbors
    double *d_cov_device;           // Contiguous memory for covariance matrix
    double *d_cross_cov_device;           // Contiguous memory for cross covariance matrix
    double *d_conditioning_cov_device;           // Contiguous memory for conditioning covariance matrix
    // pointers for matrix and vectors 
    // double **d_locs_array;
    // double **d_locs_nearestNeighbors_array;
    double **d_locs_x_array;
    double **d_locs_y_array;
    double **d_locs_nearestNeighbors_x_array;
    double **d_locs_nearestNeighbors_y_array;
    double **d_observations_points_array;
    double **d_observations_nearestNeighbors_array;
    double **d_cov_array;
    double **d_cross_cov_array;
    double **d_conditioning_cov_array;
};

// Function to copy data from CPU to GPU and allocate memory with leading dimensions
GpuData copyDataToGPU(int gpu_id, const std::vector<BlockInfo> &blockInfos);

// Function to perform computation on the GPU
void performComputationOnGPU(const GpuData &gpuData, const std::vector<double> &theta);

// Function to clean up GPU memory
void cleanupGpuMemory(GpuData &gpuData);

#endif // GPU_OPERATIONS_H
