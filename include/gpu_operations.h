#ifndef GPU_OPERATIONS_H
#define GPU_OPERATIONS_H

#include <vector>
#include <utility>
#include <cuda_runtime.h>
#include <magma_v2.h>
#include "block_info.h"
#include "input_parser.h"
// Define the GpuData structure to store GPU memory pointers and leading dimensions
struct GpuData
{
    // magma operations
    magma_int_t *dinfo_magma;
    // locations 
    double **h_locs_array;
    double **h_locs_neighbors_array;
    // observations
    double **h_observations_array;
    double **h_observations_neighbors_array;
    // covariance matrix, used to store the pointer of contiguous memory
    double **h_cov_array;
    double **h_cross_cov_array;
    double **h_conditioning_cov_array;
    double **h_observations_neighbors_copy_array;
    double **h_observations_copy_array;
    double **h_mu_correction_array;
    double **h_cov_correction_array;
    // leading dimensions for GPU points (varied)
    std::vector<int> ldda_locs;      // Leading dimensions for GPU points (varied)
    std::vector<int> ldda_neighbors;   // Leading dimension for GPU nearest neighbors (varied)
    std::vector<int> ldda_cov;
    std::vector<int> ldda_cross_cov;
    std::vector<int> ldda_conditioning_cov;
    std::vector<int> lda_locs;      // leading dimension for CPU points (varied)
    std::vector<int> lda_locs_neighbors;   // leading dimension for CPU points (varied)
    std::vector<int> h_const1;
    magma_int_t max_m, max_n1, max_n2;
    int *d_ldda_locs;
    int *d_ldda_neighbors;
    int *d_ldda_cov;
    int *d_ldda_cross_cov;
    int *d_ldda_conditioning_cov;
    int *d_lda_locs;
    int *d_lda_locs_neighbors;
    int *d_const1;
    int total_observations_points_size; // bytes
    int total_observations_neighbors_size; // bytes
    int total_locs_num_device; // number of points (including the padding)
    int total_locs_neighbors_num_device; // number of nearest neighbors (including the padding)
    // contiguous memory for GPU points
    // double *d_locs_device;           // Contiguous memory for locations
    // double *d_locs_nearestNeighbors_device; // Contiguous memory for locations of nearest neighbors
    // only for 2D points (testing the coalesced memory access)
    double *d_locs_device; // Contiguous memory for locations
    double *d_locs_neighbors_device; // Contiguous memory for locations 
    double *d_observations_device;           // Contiguous memory for observations points
    double *d_observations_neighbors_device; // Contiguous memory for observations nearest neighbors
    double *d_cov_device;           // Contiguous memory for covariance matrix
    double *d_cross_cov_device;           // Contiguous memory for cross covariance matrix
    double *d_conditioning_cov_device;           // Contiguous memory for conditioning covariance matrix
    double *d_observations_neighbors_copy_device;
    double *d_observations_copy_device;
    double *d_mu_correction_device;
    double *d_cov_correction_device;
    
    // pointers for matrix and vectors 
    double **d_locs_array;
    double **d_locs_neighbors_array;
    double **d_observations_points_array;
    double **d_observations_neighbors_array;
    double **d_cov_array;
    double **d_cross_cov_array;
    double **d_conditioning_cov_array;
    double **d_observations_neighbors_copy_array;
    double **d_observations_copy_array; 
    double **d_mu_correction_array;
    double **d_cov_correction_array;
};

// Function to copy data from CPU to GPU and allocate memory with leading dimensions
GpuData copyDataToGPU(const Opts &opts, const std::vector<BlockInfo> &blockInfos);

// Function to perform computation on the GPU
double performComputationOnGPU(const GpuData &gpuData, const std::vector<double> &theta, const Opts &opts);
// Function to clean up GPU memory
void cleanupGpuMemory(GpuData &gpuData);

double gflopsTotal(const GpuData &gpuData, const Opts &opts);

#endif // GPU_OPERATIONS_H
