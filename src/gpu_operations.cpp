#include <iostream>
#include <mpi.h>
#include "gpu_operations.h"
#include "gpu_covariance.h"

// Updated function to check CUDA errors with file and line information
#define checkCudaError(error) _checkCudaError(error, __FILE__, __LINE__)

void _checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in " << file << ":" << line 
                  << ": " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Function to copy data from CPU to GPU and allocate memory with leading dimensions
GpuData copyDataToGPU(int gpu_id, const std::vector<BlockInfo> &blockInfos)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    GpuData gpuData;
    // cpu leading dimensions
    gpuData.lda_locs.resize(blockInfos.size());
    gpuData.lda_locs_neighbors.resize(blockInfos.size());
    // gpu leading dimensions
    gpuData.ldda_locs.resize(blockInfos.size());
    gpuData.ldda_neighbors.resize(blockInfos.size());
    gpuData.ldda_cov.resize(blockInfos.size());
    gpuData.ldda_cross_cov.resize(blockInfos.size());
    gpuData.ldda_conditioning_cov.resize(blockInfos.size());

    // Allocate arrays of pointers on the host
    // gpuData.h_locs_array = new double *[blockInfos.size()];
    // gpuData.h_locs_nearestNeighbors_array = new double *[blockInfos.size()];
    gpuData.h_locs_x_array = new double *[blockInfos.size()];
    gpuData.h_locs_y_array = new double *[blockInfos.size()];
    gpuData.h_locs_nearestNeighbors_x_array = new double *[blockInfos.size()];
    gpuData.h_locs_nearestNeighbors_y_array = new double *[blockInfos.size()];
    gpuData.h_observations_array = new double *[blockInfos.size()];
    gpuData.h_observations_nearestNeighbors_array = new double *[blockInfos.size()];
    gpuData.h_cov_array = new double *[blockInfos.size()];
    gpuData.h_cross_cov_array = new double *[blockInfos.size()];
    gpuData.h_conditioning_cov_array = new double *[blockInfos.size()];
    // array of pointers for the device
    // checkCudaError(cudaMalloc(&gpuData.d_locs_array, blockInfos.size() * sizeof(double *)));
    // checkCudaError(cudaMalloc(&gpuData.d_locs_nearestNeighbors_array, blockInfos.size() * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_locs_x_array, blockInfos.size() * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_locs_y_array, blockInfos.size() * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_locs_nearestNeighbors_x_array, blockInfos.size() * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_locs_nearestNeighbors_y_array, blockInfos.size() * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_observations_points_array, blockInfos.size() * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_observations_nearestNeighbors_array, blockInfos.size() * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_cov_array, blockInfos.size() * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_cross_cov_array, blockInfos.size() * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_conditioning_cov_array, blockInfos.size() * sizeof(double *)));

    // Set the GPU
    checkCudaError(cudaSetDevice(gpu_id));

    // Calculate the total memory needed for points and nearest neighbors
    // size_t total_points_size = 0;
    // size_t total_neighbors_size = 0;
    size_t total_observations_points_size = 0;
    size_t total_observations_nearestNeighbors_size = 0;
    size_t total_cov_size = 0;
    size_t total_cross_cov_size = 0;
    size_t total_conditioning_cov_size = 0;
    size_t total_locs_x_size_host = 0;
    size_t total_locs_y_size_host = 0;
    size_t total_locs_nearestNeighbors_x_size_host = 0;
    size_t total_locs_nearestNeighbors_y_size_host = 0;
    size_t total_locs_x_size_device = 0;
    size_t total_locs_y_size_device = 0;
    size_t total_locs_nearestNeighbors_x_size_device = 0;
    size_t total_locs_nearestNeighbors_y_size_device = 0;

    for (size_t i = 0; i < blockInfos.size(); ++i)
    {
        // number of clusters and their nearest neighbors
        int m_points = blockInfos[i].points.size();
        int m_nearest_neighbor = blockInfos[i].nearestNeighbors.size();
        // 32 is the aligned 32 threads in a warp in GPU
        gpuData.ldda_locs[i] = magma_roundup(m_points, 32); 
        gpuData.ldda_neighbors[i] = magma_roundup(m_nearest_neighbor, 32); 
        gpuData.lda_locs[i] = m_points;
        gpuData.lda_locs_neighbors[i] = m_nearest_neighbor;
        gpuData.ldda_cov[i] = gpuData.ldda_locs[i];
        gpuData.ldda_cross_cov[i] = gpuData.ldda_neighbors[i];
        gpuData.ldda_conditioning_cov[i] = gpuData.ldda_neighbors[i];
        // total size of contiguous memory 
        // total_points_size += gpuData.ldda_locs[i] * sizeof(std::array<double, DIMENSION>);
        // total_neighbors_size += gpuData.ldda_neighbors[i] * sizeof(std::array<double, DIMENSION>);
        total_cov_size += gpuData.ldda_cov[i] * m_points * sizeof(double);
        total_conditioning_cov_size += gpuData.ldda_conditioning_cov[i] * m_nearest_neighbor * sizeof(double);
        total_cross_cov_size += gpuData.ldda_conditioning_cov[i] * m_points * sizeof(double);
        total_observations_points_size += gpuData.ldda_locs[i] * sizeof(double);
        total_observations_nearestNeighbors_size += gpuData.ldda_neighbors[i] * sizeof(double);
        total_locs_x_size_host += m_points * sizeof(double) ;
        total_locs_y_size_host += m_points * sizeof(double);
        total_locs_nearestNeighbors_x_size_host += m_nearest_neighbor * sizeof(double);
        total_locs_nearestNeighbors_y_size_host += m_nearest_neighbor * sizeof(double);
        total_locs_x_size_device += gpuData.ldda_locs[i] * sizeof(double);
        total_locs_y_size_device += gpuData.ldda_locs[i] * sizeof(double);
        total_locs_nearestNeighbors_x_size_device += gpuData.ldda_neighbors[i] * sizeof(double);
        total_locs_nearestNeighbors_y_size_device += gpuData.ldda_neighbors[i] * sizeof(double);
    }

    // Allocate contiguous memory on GPU
    // checkCudaError(cudaMalloc(&gpuData.d_locs_device, total_points_size));
    // checkCudaError(cudaMalloc(&gpuData.d_locs_nearestNeighbors_device, total_neighbors_size));
    checkCudaError(cudaMalloc(&gpuData.d_locs_x_device, total_locs_x_size_device));
    checkCudaError(cudaMalloc(&gpuData.d_locs_y_device, total_locs_y_size_device));
    checkCudaError(cudaMalloc(&gpuData.d_locs_nearestNeighbors_x_device, total_locs_nearestNeighbors_x_size_device));
    checkCudaError(cudaMalloc(&gpuData.d_locs_nearestNeighbors_y_device, total_locs_nearestNeighbors_y_size_device));
    checkCudaError(cudaMalloc(&gpuData.d_observations_device, total_observations_points_size));
    checkCudaError(cudaMalloc(&gpuData.d_observations_nearestNeighbors_device, total_observations_nearestNeighbors_size));
    checkCudaError(cudaMalloc(&gpuData.d_cov_device, total_cov_size));
    checkCudaError(cudaMalloc(&gpuData.d_conditioning_cov_device, total_conditioning_cov_size));
    checkCudaError(cudaMalloc(&gpuData.d_cross_cov_device, total_cross_cov_size));
    
    // Prepare to store points data for coalesced memory access
    double *locs_x_data = new double[total_locs_x_size_host/sizeof(double)];
    double *locs_y_data = new double[total_locs_y_size_host/sizeof(double)];
    double *locs_nearestNeighbors_x_data = new double[total_locs_nearestNeighbors_x_size_host/sizeof(double)];
    double *locs_nearestNeighbors_y_data = new double[total_locs_nearestNeighbors_y_size_host/sizeof(double)];

    size_t locs_x_index = 0;
    size_t locs_y_index = 0;
    size_t locs_nearestNeighbors_x_index = 0;
    size_t locs_nearestNeighbors_y_index = 0;
    for (size_t i = 0; i < blockInfos.size(); ++i)
    {
        int m_points = blockInfos[i].points.size();
        int m_nearest_neighbor = blockInfos[i].nearestNeighbors.size();
        // copy locations (coalesced memory access)
        for (int j = 0; j < m_points; ++j)
        {
            locs_x_data[locs_x_index++] = blockInfos[i].points[j][0];
            locs_y_data[locs_y_index++] = blockInfos[i].points[j][1];
        }
        for (int j = 0; j < m_nearest_neighbor; ++j)
        {
            locs_nearestNeighbors_x_data[locs_nearestNeighbors_x_index++] = blockInfos[i].nearestNeighbors[j][0];
            locs_nearestNeighbors_y_data[locs_nearestNeighbors_y_index++] = blockInfos[i].nearestNeighbors[j][1];
        }
    }

    // Assign pointers to the beginning of each block's memory and copy data
    // double *locs_ptr = gpuData.d_locs_device;
    // double *locs_neighbors_ptr = gpuData.d_locs_nearestNeighbors_device;
    double *locs_x_ptr = gpuData.d_locs_x_device;
    double *locs_y_ptr = gpuData.d_locs_y_device;
    double *locs_nearestNeighbors_x_ptr = gpuData.d_locs_nearestNeighbors_x_device;
    double *locs_nearestNeighbors_y_ptr = gpuData.d_locs_nearestNeighbors_y_device;
    double *observations_points_ptr = gpuData.d_observations_device;
    double *observations_nearestNeighbors_ptr = gpuData.d_observations_nearestNeighbors_device;
    double *cov_ptr = gpuData.d_cov_device;
    double *conditioning_cov_ptr = gpuData.d_conditioning_cov_device;
    double *cross_cov_ptr = gpuData.d_cross_cov_device;

    // calculate size 
    size_t index_locs_x = 0;
    size_t index_locs_y = 0;
    size_t index_locs_nearestNeighbors_x = 0;
    size_t index_locs_nearestNeighbors_y = 0;
    for (size_t i = 0; i < blockInfos.size(); ++i)
    {
        int m_points = blockInfos[i].points.size();
        int m_nearest_neighbor = blockInfos[i].nearestNeighbors.size();
        // gpuData.h_locs_array[i] = locs_ptr;
        // gpuData.h_locs_nearestNeighbors_array[i] = locs_neighbors_ptr;
        gpuData.h_locs_x_array[i] = locs_x_ptr;
        gpuData.h_locs_y_array[i] = locs_y_ptr;
        gpuData.h_locs_nearestNeighbors_x_array[i] = locs_nearestNeighbors_x_ptr;
        gpuData.h_locs_nearestNeighbors_y_array[i] = locs_nearestNeighbors_y_ptr;
        gpuData.h_observations_array[i] = observations_points_ptr;
        gpuData.h_observations_nearestNeighbors_array[i] = observations_nearestNeighbors_ptr;
        gpuData.h_cov_array[i] = cov_ptr;
        gpuData.h_conditioning_cov_array[i] = conditioning_cov_ptr;
        gpuData.h_cross_cov_array[i] = cross_cov_ptr;
        // (observations)
        checkCudaError(cudaMemcpyAsync(observations_points_ptr, 
                                   blockInfos[i].observations_points.data(), 
                                   blockInfos[i].observations_points.size() * sizeof(double), 
                                   cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpyAsync(observations_nearestNeighbors_ptr, 
                                   blockInfos[i].observations_nearestNeighbors.data(), 
                                   blockInfos[i].observations_nearestNeighbors.size() * sizeof(double), 
                                   cudaMemcpyHostToDevice));
        // copy the locations from the host to the device (locations + observations)
        // checkCudaError(cudaMemcpyAsync(locs_ptr, 
        //                            blockInfos[i].points.data(), 
        //                            blockInfos[i].points.size() * sizeof(std::array<double, DIMENSION>), 
        //                            cudaMemcpyHostToDevice));
        // checkCudaError(cudaMemcpyAsync(locs_neighbors_ptr, 
        //                            blockInfos[i].nearestNeighbors.data(), 
        //                            blockInfos[i].nearestNeighbors.size() * sizeof(std::array<double, DIMENSION>), 
        //                            cudaMemcpyHostToDevice));
        // copy locations (coalesced memory access)
        checkCudaError(cudaMemcpyAsync(locs_x_ptr, 
                                   locs_x_data + index_locs_x, 
                                   m_points * sizeof(double), 
                                   cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpyAsync(locs_y_ptr, 
                                   locs_y_data + index_locs_y, 
                                   m_points * sizeof(double), 
                                   cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpyAsync(locs_nearestNeighbors_x_ptr, 
                                   locs_nearestNeighbors_x_data + index_locs_nearestNeighbors_x, 
                                   m_nearest_neighbor * sizeof(double), 
                                   cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpyAsync(locs_nearestNeighbors_y_ptr, 
                                   locs_nearestNeighbors_y_data + index_locs_nearestNeighbors_y, 
                                   m_nearest_neighbor * sizeof(double), 
                                   cudaMemcpyHostToDevice));
        // next pointer
        // locs_ptr += gpuData.ldda_locs[i] * DIMENSION;
        // locs_neighbors_ptr += gpuData.ldda_neighbors[i] * DIMENSION;
        locs_x_ptr += gpuData.ldda_locs[i];
        locs_y_ptr += gpuData.ldda_locs[i];
        locs_nearestNeighbors_x_ptr += gpuData.ldda_neighbors[i];
        locs_nearestNeighbors_y_ptr += gpuData.ldda_neighbors[i];
        observations_points_ptr += gpuData.ldda_locs[i];
        observations_nearestNeighbors_ptr += gpuData.ldda_neighbors[i];
        cov_ptr += gpuData.ldda_cov[i] * m_points;
        conditioning_cov_ptr += gpuData.ldda_conditioning_cov[i] * m_nearest_neighbor;
        cross_cov_ptr += gpuData.ldda_conditioning_cov[i] * m_points;
        // index update
        index_locs_x += m_points;
        index_locs_y += m_points;
        index_locs_nearestNeighbors_x += m_nearest_neighbor;
        index_locs_nearestNeighbors_y += m_nearest_neighbor;    
    }

    // copy data array to the GPU
    // checkCudaError(cudaMemcpyAsync(gpuData.d_locs_array, 
    //            gpuData.h_locs_array, 
    //            blockInfos.size() * sizeof(double *), 
    //            cudaMemcpyHostToDevice));
    // checkCudaError(cudaMemcpyAsync(gpuData.d_locs_nearestNeighbors_array, 
    //            gpuData.h_locs_nearestNeighbors_array, 
    //            blockInfos.size() * sizeof(double *), 
    //            cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpyAsync(gpuData.d_locs_x_array, 
               gpuData.h_locs_x_array, 
               blockInfos.size() * sizeof(double *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpyAsync(gpuData.d_locs_y_array, 
               gpuData.h_locs_y_array, 
               blockInfos.size() * sizeof(double *), 
               cudaMemcpyHostToDevice));    
    checkCudaError(cudaMemcpyAsync(gpuData.d_locs_nearestNeighbors_x_array, 
               gpuData.h_locs_nearestNeighbors_x_array, 
               blockInfos.size() * sizeof(double *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpyAsync(gpuData.d_locs_nearestNeighbors_y_array, 
               gpuData.h_locs_nearestNeighbors_y_array, 
               blockInfos.size() * sizeof(double *), 
               cudaMemcpyHostToDevice));    
    checkCudaError(cudaMemcpyAsync(gpuData.d_observations_points_array, 
               gpuData.h_observations_array, 
               blockInfos.size() * sizeof(double *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpyAsync(gpuData.d_observations_nearestNeighbors_array, 
               gpuData.h_observations_nearestNeighbors_array, 
               blockInfos.size() * sizeof(double *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpyAsync(gpuData.d_cov_array, 
               gpuData.h_cov_array, 
               blockInfos.size() * sizeof(double *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpyAsync(gpuData.d_conditioning_cov_array, 
               gpuData.h_conditioning_cov_array, 
               blockInfos.size() * sizeof(double *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpyAsync(gpuData.d_cross_cov_array, 
               gpuData.h_cross_cov_array, 
               blockInfos.size() * sizeof(double *), 
               cudaMemcpyHostToDevice));

    delete[] locs_x_data;
    delete[] locs_y_data;
    delete[] locs_nearestNeighbors_x_data;
    delete[] locs_nearestNeighbors_y_data;

    return gpuData;
}

// Function to perform computation on the GPU
void performComputationOnGPU(const GpuData &gpuData, const std::vector<double> &theta)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Use the data on the GPU for computation
    // 1. generate the covariance matrix, cross covariance matrix, conditioning covariance matrix
    // take record of the time
    for (size_t i = 0; i < gpuData.ldda_locs.size(); ++i)
    {   
        covarianceMatern1_2_v1(gpuData.h_locs_x_array[i], gpuData.h_locs_y_array[i], 
                               gpuData.lda_locs[i], 1,
                               gpuData.h_locs_x_array[i], gpuData.h_locs_y_array[i], 
                               gpuData.lda_locs[i], 1,
                               gpuData.h_cov_array[i], gpuData.ldda_cov[i], gpuData.lda_locs[i],
                               theta, stream);
        covarianceMatern1_2_v1(gpuData.h_locs_nearestNeighbors_x_array[i], gpuData.h_locs_nearestNeighbors_y_array[i], 
                               gpuData.lda_locs_neighbors[i], 1,
                               gpuData.h_locs_x_array[i], gpuData.h_locs_y_array[i], 
                               gpuData.lda_locs[i], 1,
                               gpuData.h_cross_cov_array[i], gpuData.ldda_cross_cov[i], gpuData.lda_locs[i],
                               theta, stream);
        covarianceMatern1_2_v1(gpuData.h_locs_nearestNeighbors_x_array[i], gpuData.h_locs_nearestNeighbors_y_array[i], 
                               gpuData.lda_locs_neighbors[i], 1,
                               gpuData.h_locs_nearestNeighbors_x_array[i], gpuData.h_locs_nearestNeighbors_y_array[i], 
                               gpuData.lda_locs_neighbors[i], 1,
                               gpuData.h_conditioning_cov_array[i], gpuData.ldda_conditioning_cov[i], gpuData.lda_locs_neighbors[i],
                               theta, stream);
        // Synchronize to make sure the kernel has finished
        checkCudaError(cudaStreamSynchronize(stream));
    }    

    // 2. perform the computation
    // 2.1 compute the correction term for mean and variance (i.e., Schur complement)
    // 2.2 compute the conditional mean
    // 2.3 compute the conditional variance
    // 2.4 compute the log-likelihood

    // 3. copy the result back to the CPU
}

// Function to clean up GPU memory
void cleanupGpuMemory(GpuData &gpuData)
{
    // cudaFree(gpuData.d_locs_device);
    // cudaFree(gpuData.d_locs_nearestNeighbors_device);
    cudaFree(gpuData.d_cov_device);
    cudaFree(gpuData.d_conditioning_cov_device);
    cudaFree(gpuData.d_cross_cov_device);
    cudaFree(gpuData.d_locs_x_device);
    cudaFree(gpuData.d_locs_y_device);
    cudaFree(gpuData.d_locs_nearestNeighbors_x_device);
    cudaFree(gpuData.d_locs_nearestNeighbors_y_device);
    cudaFree(gpuData.d_observations_device);
    cudaFree(gpuData.d_observations_nearestNeighbors_device);

    // delete[] gpuData.h_locs_array;
    // delete[] gpuData.h_locs_nearestNeighbors_array;
    delete[] gpuData.h_cov_array;
    delete[] gpuData.h_conditioning_cov_array;
    delete[] gpuData.h_cross_cov_array;
    delete[] gpuData.h_locs_x_array;
    delete[] gpuData.h_locs_y_array;
    delete[] gpuData.h_locs_nearestNeighbors_x_array;
    delete[] gpuData.h_locs_nearestNeighbors_y_array;
    delete[] gpuData.h_observations_array;
    delete[] gpuData.h_observations_nearestNeighbors_array;
}
