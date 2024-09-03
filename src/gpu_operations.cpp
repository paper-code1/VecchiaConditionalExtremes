#include <iostream>
#include <mpi.h>
#include "gpu_operations.h"

// Function to copy data from CPU to GPU and allocate memory with leading dimensions
GpuData copyDataToGPU(int gpu_id, const std::vector<BlockInfo> &blockInfos)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    GpuData gpuData;
    gpuData.ldda_points.resize(blockInfos.size());
    gpuData.ldda_neighbors.resize(blockInfos.size());
    gpuData.ldda_cov.resize(blockInfos.size());
    gpuData.ldda_cross_cov.resize(blockInfos.size());
    gpuData.ldda_conditioning_cov.resize(blockInfos.size());

    // Allocate arrays of pointers on the host
    gpuData.h_points_array = new double *[blockInfos.size()];
    gpuData.h_nearestNeighbors_array = new double *[blockInfos.size()];
    gpuData.h_cov_array = new double *[blockInfos.size()];
    gpuData.h_cross_cov_array = new double *[blockInfos.size()];
    gpuData.h_conditioning_cov_array = new double *[blockInfos.size()];

    // Set the GPU
    cudaSetDevice(gpu_id);

    // Calculate the total memory needed for points and nearest neighbors
    size_t total_points_size = 0;
    size_t total_neighbors_size = 0;
    size_t total_cov_size = 0;
    size_t total_cross_cov_size = 0;
    size_t total_conditioning_cov_size = 0;

    for (size_t i = 0; i < blockInfos.size(); ++i)
    {
        // number of clusters and their nearest neighbors
        int m_points = blockInfos[i].points.size();
        int m_nearest_neighbor = blockInfos[i].nearestNeighbors.size();
        // 32 is the aligned 32 threads in a warp in GPU
        gpuData.ldda_points[i] = magma_roundup(m_points, 32); 
        gpuData.ldda_neighbors[i] = magma_roundup(m_nearest_neighbor, 32); 
        gpuData.ldda_cov[i] = gpuData.ldda_points[i];
        gpuData.ldda_cross_cov[i] = gpuData.ldda_neighbors[i];
        gpuData.ldda_conditioning_cov[i] = gpuData.ldda_neighbors[i];
        // total size of contiguous memory 
        total_points_size += gpuData.ldda_points[i] * sizeof(std::pair<double, double>);
        total_neighbors_size += gpuData.ldda_neighbors[i] * sizeof(std::pair<double, double>);
        total_cov_size += gpuData.ldda_cov[i] * m_points * sizeof(double);
        total_conditioning_cov_size += gpuData.ldda_conditioning_cov[i] * m_nearest_neighbor * sizeof(double);
        total_cross_cov_size += gpuData.ldda_conditioning_cov[i] * m_points * sizeof(double);
    }

    // Allocate contiguous memory on GPU
    cudaMalloc(&gpuData.h_points_memory, total_points_size);
    cudaMalloc(&gpuData.h_nearestNeighbors_memory, total_neighbors_size);
    cudaMalloc(&gpuData.h_cov_memory, total_cov_size);
    cudaMalloc(&gpuData.h_conditioning_cov_memory, total_conditioning_cov_size);
    cudaMalloc(&gpuData.h_cross_cov_memory, total_cross_cov_size);

    // std::cout << "rank " << rank << " ," 
    // << "total_points_size " << total_points_size / sizeof(std::pair<double, double>)<< " ," 
    // << "total_neighbors_size " << total_neighbors_size / sizeof(std::pair<double, double>) << " ," 
    // << "total_cov_size " << total_cov_size / sizeof(double)<< " ," 
    // << "total_conditioning_cov_size " << total_conditioning_cov_size / sizeof(double) << " ," 
    // << "total_cross_cov_size " << total_cross_cov_size / sizeof(double) << "." 
    // << std::endl;
    
    // Assign pointers to the beginning of each block's memory and copy data
    double *points_ptr = gpuData.h_points_memory;
    double *neighbors_ptr = gpuData.h_nearestNeighbors_memory;

    for (size_t i = 0; i < blockInfos.size(); ++i)
    {
        gpuData.h_points_array[i] = points_ptr;
        gpuData.h_nearestNeighbors_array[i] = neighbors_ptr;
        cudaMemcpy(points_ptr, blockInfos[i].points.data(), blockInfos[i].points.size() * sizeof(std::pair<double, double>), cudaMemcpyHostToDevice);
        cudaMemcpy(neighbors_ptr, blockInfos[i].nearestNeighbors.data(), blockInfos[i].nearestNeighbors.size() * sizeof(std::pair<double, double>), cudaMemcpyHostToDevice);
        points_ptr += gpuData.ldda_points[i] * sizeof(std::pair<double, double>);
        neighbors_ptr += gpuData.ldda_neighbors[i] * sizeof(std::pair<double, double>);
    }

    return gpuData;
}

// Function to perform computation on the GPU
void performComputationOnGPU(const GpuData &gpuData)
{
    // Use the data on the GPU for computation
    // Example: Assuming you have some MAGMA operation to perform
    // for (size_t i = 0; i < gpuData.h_points_array.size(); ++i) {
    //     magma_operation(gpuData.h_points_array[i], gpuData.h_nearestNeighbors_array[i], gpuData.ldda_points[i], gpuData.ldda_neighbors);
    // }
}

// Function to clean up GPU memory
void cleanupGpuMemory(GpuData &gpuData)
{
    cudaFree(gpuData.h_points_memory);
    cudaFree(gpuData.h_nearestNeighbors_memory);
    cudaFree(gpuData.h_cov_memory);
    cudaFree(gpuData.h_conditioning_cov_memory);
    cudaFree(gpuData.h_cross_cov_memory);

    delete[] gpuData.h_points_array;
    delete[] gpuData.h_nearestNeighbors_array;
    delete[] gpuData.h_cov_array;
    delete[] gpuData.h_conditioning_cov_array;
    delete[] gpuData.h_cross_cov_array;
}
