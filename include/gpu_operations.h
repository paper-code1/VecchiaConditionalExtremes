#ifndef GPU_OPERATIONS_H
#define GPU_OPERATIONS_H

#include <vector>
#include <utility>
#include <cuda_runtime.h>
#include <magma_v2.h>
#include "block_info.h"
#include "input_parser.h"
// Define a templated GpuData structure to store GPU memory pointers and leading dimensions
template <typename Real>
struct GpuDataT
{
    // magma operations
    magma_int_t *dinfo_magma;
    // locations 
    Real **h_locs_array;
    Real **h_locs_neighbors_array;
    // observations
    Real **h_observations_array;
    Real **h_observations_neighbors_array;
    // covariance matrix, used to store the pointer of contiguous memory
    Real **h_cov_array;
    Real **h_cross_cov_array;
    Real **h_conditioning_cov_array;
    Real **h_observations_neighbors_copy_array;
    Real **h_observations_copy_array;
    Real **h_mu_correction_array;
    Real **h_cov_correction_array;
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
    int numPointsPerProcess;// number of point after outer partition
    int total_observations_points_size; // bytes
    int total_observations_neighbors_size; // bytes
    int total_locs_num_device; // number of points (including the padding)
    int total_locs_neighbors_num_device; // number of nearest neighbors (including the padding)
    // contiguous memory for GPU points
    // Real *d_locs_device;           // Contiguous memory for locations
    // Real *d_locs_nearestNeighbors_device; // Contiguous memory for locations of nearest neighbors
    // only for 2D points (testing the coalesced memory access)
    Real *d_locs_device; // Contiguous memory for locations
    Real *d_locs_neighbors_device; // Contiguous memory for locations 
    Real *d_observations_device;           // Contiguous memory for observations points
    Real *d_observations_neighbors_device; // Contiguous memory for observations nearest neighbors
    Real *d_cov_device;           // Contiguous memory for covariance matrix
    Real *d_cross_cov_device;           // Contiguous memory for cross covariance matrix
    Real *d_conditioning_cov_device;           // Contiguous memory for conditioning covariance matrix
    Real *d_observations_neighbors_copy_device;
    Real *d_observations_copy_device;
    Real *d_mu_correction_device;
    Real *d_cov_correction_device;
    Real *d_range_device;
    
    // pointers for matrix and vectors 
    Real **d_locs_array;
    Real **d_locs_neighbors_array;
    Real **d_observations_points_array;
    Real **d_observations_neighbors_array;
    Real **d_cov_array;
    Real **d_cross_cov_array;
    Real **d_conditioning_cov_array;
    Real **d_observations_neighbors_copy_array;
    Real **d_observations_copy_array; 
    Real **d_mu_correction_array;
    Real **d_cov_correction_array;

    // total sizes in bytes for contiguous buffers (for conversions)
    size_t total_cov_size_bytes = 0;
    size_t total_cross_cov_size_bytes = 0;
    size_t total_conditioning_cov_size_bytes = 0;

    // Optional double-precision mirrors for mixed-precision execution (used when base Real=float)
    // Device contiguous buffers (allocated only if requested via opts)
    double *d_locs_device_f64 = nullptr;
    double *d_locs_neighbors_device_f64 = nullptr;
    double *d_observations_device_f64 = nullptr;
    double *d_observations_neighbors_device_f64 = nullptr;
    double *d_cov_device_f64 = nullptr;
    double *d_cross_cov_device_f64 = nullptr;
    double *d_conditioning_cov_device_f64 = nullptr;
    double *d_observations_neighbors_copy_device_f64 = nullptr;
    double *d_observations_copy_device_f64 = nullptr;
    double *d_mu_correction_device_f64 = nullptr;
    double *d_cov_correction_device_f64 = nullptr;
    double *d_range_device_f64 = nullptr;

    // Host arrays of pointers for double-precision matrices/vectors
    double **h_locs_array_f64 = nullptr;
    double **h_locs_neighbors_array_f64 = nullptr;
    double **h_observations_array_f64 = nullptr;
    double **h_observations_neighbors_array_f64 = nullptr;
    double **h_cov_array_f64 = nullptr;
    double **h_cross_cov_array_f64 = nullptr;
    double **h_conditioning_cov_array_f64 = nullptr;
    double **h_observations_neighbors_copy_array_f64 = nullptr;
    double **h_observations_copy_array_f64 = nullptr;
    double **h_mu_correction_array_f64 = nullptr;
    double **h_cov_correction_array_f64 = nullptr;

    // Device arrays of pointers for double-precision
    double **d_locs_array_f64 = nullptr;
    double **d_locs_neighbors_array_f64 = nullptr;
    double **d_observations_points_array_f64 = nullptr;
    double **d_observations_neighbors_array_f64 = nullptr;
    double **d_cov_array_f64 = nullptr;
    double **d_cross_cov_array_f64 = nullptr;
    double **d_conditioning_cov_array_f64 = nullptr;
    double **d_observations_neighbors_copy_array_f64 = nullptr;
    double **d_observations_copy_array_f64 = nullptr;
    double **d_mu_correction_array_f64 = nullptr;
    double **d_cov_correction_array_f64 = nullptr;
};

using GpuData = GpuDataT<double>;
using GpuDataF = GpuDataT<float>;

// Function to copy data from CPU to GPU and allocate memory with leading dimensions
template <typename Real>
GpuDataT<Real> copyDataToGPU(const Opts &opts, const std::vector<BlockInfo> &blockInfos);

// Function to perform computation on the GPU
template <typename Real>
double performComputationOnGPU(const GpuDataT<Real> &gpuData, const std::vector<double> &theta, Opts &opts);
// Function to clean up GPU memory
template <typename Real>
void cleanupGpuMemory(GpuDataT<Real> &gpuData);

template <typename Real>
double gflopsTotal(const GpuDataT<Real> &gpuData, const Opts &opts);

#endif // GPU_OPERATIONS_H
