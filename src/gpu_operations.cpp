#include <iostream>
#include <mpi.h>
#include <magma_v2.h>
#include "gpu_operations.h"
#include "gpu_covariance.h"
#include "flops.h"
#include "error_checking.h"
#include "magma_dprint_gpu.h"

// Function to copy data from CPU to GPU and allocate memory with leading dimensions
GpuData copyDataToGPU(const Opts &opts, const std::vector<BlockInfo> &blockInfos)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Set the GPU
    checkCudaError(cudaSetDevice(opts.gpu_id));

    GpuData gpuData;
    // cpu leading dimensions (+1 for magma)
    gpuData.lda_locs.resize(blockInfos.size() + 1);
    gpuData.lda_locs_neighbors.resize(blockInfos.size() + 1);
    // gpu leading dimensions
    gpuData.ldda_locs.resize(blockInfos.size() + 1);
    gpuData.ldda_neighbors.resize(blockInfos.size() + 1);
    gpuData.ldda_cov.resize(blockInfos.size() + 1);
    gpuData.ldda_cross_cov.resize(blockInfos.size() + 1);
    gpuData.ldda_conditioning_cov.resize(blockInfos.size() + 1);
    gpuData.h_const1.resize(blockInfos.size() + 1);

    // Allocate arrays of pointers on the host
    gpuData.h_locs_array = new double *[blockInfos.size() * opts.dim];
    gpuData.h_locs_neighbors_array = new double *[blockInfos.size() * opts.dim];
    gpuData.h_observations_array = new double *[blockInfos.size()];
    gpuData.h_observations_neighbors_array = new double *[blockInfos.size()];
    gpuData.h_cov_array = new double *[blockInfos.size()];
    gpuData.h_cross_cov_array = new double *[blockInfos.size()];
    gpuData.h_conditioning_cov_array = new double *[blockInfos.size()];
    gpuData.h_observations_neighbors_copy_array = new double *[blockInfos.size()];
    gpuData.h_observations_copy_array = new double *[blockInfos.size()];
    gpuData.h_mu_correction_array = new double *[blockInfos.size()];
    gpuData.h_cov_correction_array = new double *[blockInfos.size()];

    // array of pointers for the device
    checkCudaError(cudaMalloc(&gpuData.d_locs_array, blockInfos.size() * opts.dim * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_locs_neighbors_array, blockInfos.size() * opts.dim * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_observations_points_array, blockInfos.size() * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_observations_neighbors_array, blockInfos.size() * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_cov_array, blockInfos.size() * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_cross_cov_array, blockInfos.size() * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_conditioning_cov_array, blockInfos.size() * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_observations_neighbors_copy_array, blockInfos.size() * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_observations_copy_array, blockInfos.size() * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_mu_correction_array, blockInfos.size() * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_cov_correction_array, blockInfos.size() * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_range_device, opts.dim * sizeof(double)));

    // Calculate the total memory needed for blocks and nearest neighbors
    size_t total_observations_points_size = 0;
    size_t total_observations_nearestNeighbors_size = 0;
    size_t total_cov_size = 0;
    size_t total_cross_cov_size = 0;
    size_t total_conditioning_cov_size = 0;
    size_t total_locs_size_host = 0;
    size_t total_locs_nearestNeighbors_size_host = 0;
    size_t total_locs_size_device = 0;
    size_t total_locs_nearestNeighbors_size_device = 0;

    gpuData.numPointsPerProcess = 0;
    for (size_t i = 0; i < blockInfos.size(); ++i)
    {
        // number of clusters and their nearest neighbors
        int m_blocks = blockInfos[i].blocks.size();
        int m_nearest_neighbor = blockInfos[i].nearestNeighbors.size();
        // number of points per process
        gpuData.numPointsPerProcess += m_blocks;
        // 32 is the aligned 32 threads in a warp in GPU
        gpuData.ldda_locs[i] = magma_roundup(m_blocks, 32); 
        gpuData.ldda_neighbors[i] = magma_roundup(m_nearest_neighbor, 32); 
        gpuData.lda_locs[i] = m_blocks;
        gpuData.lda_locs_neighbors[i] = m_nearest_neighbor;
        gpuData.ldda_cov[i] = gpuData.ldda_locs[i];
        gpuData.ldda_cross_cov[i] = gpuData.ldda_neighbors[i];
        gpuData.ldda_conditioning_cov[i] = gpuData.ldda_neighbors[i];
        gpuData.h_const1[i] = 1;
        // total size of contiguous memory 
        total_cov_size += gpuData.ldda_cov[i] * m_blocks * sizeof(double);
        total_conditioning_cov_size += gpuData.ldda_conditioning_cov[i] * m_nearest_neighbor * sizeof(double);
        total_cross_cov_size += gpuData.ldda_conditioning_cov[i] * m_blocks * sizeof(double);
        total_observations_points_size += gpuData.ldda_locs[i] * sizeof(double);
        total_observations_nearestNeighbors_size += gpuData.ldda_neighbors[i] * sizeof(double);
        total_locs_size_host += m_blocks * sizeof(double) * opts.dim;
        total_locs_nearestNeighbors_size_host += m_nearest_neighbor * sizeof(double) * opts.dim;
        total_locs_size_device += gpuData.ldda_locs[i] * sizeof(double) * opts.dim;
        total_locs_nearestNeighbors_size_device += gpuData.ldda_neighbors[i] * sizeof(double) * opts.dim;
    }

    // Allocate contiguous memory on GPU
    gpuData.total_observations_points_size = total_observations_points_size;
    gpuData.total_observations_neighbors_size = total_observations_nearestNeighbors_size;
    gpuData.total_locs_num_device = total_locs_size_device/sizeof(double)/opts.dim;
    gpuData.total_locs_neighbors_num_device = total_locs_nearestNeighbors_size_device/sizeof(double)/opts.dim;
    checkCudaError(cudaMalloc(&gpuData.d_locs_device, total_locs_size_device));
    checkCudaError(cudaMalloc(&gpuData.d_locs_neighbors_device, total_locs_nearestNeighbors_size_device));
    checkCudaError(cudaMalloc(&gpuData.d_observations_device, total_observations_points_size));
    checkCudaError(cudaMalloc(&gpuData.d_observations_neighbors_device, total_observations_nearestNeighbors_size));
    checkCudaError(cudaMalloc(&gpuData.d_cov_device, total_cov_size));
    checkCudaError(cudaMalloc(&gpuData.d_conditioning_cov_device, total_conditioning_cov_size));
    checkCudaError(cudaMalloc(&gpuData.d_cross_cov_device, total_cross_cov_size));
    checkCudaError(cudaMalloc(&gpuData.d_observations_neighbors_copy_device, total_observations_nearestNeighbors_size));
    checkCudaError(cudaMalloc(&gpuData.d_observations_copy_device, total_observations_points_size));
    checkCudaError(cudaMalloc(&gpuData.d_mu_correction_device, total_observations_points_size));
    checkCudaError(cudaMalloc(&gpuData.d_cov_correction_device, total_cov_size));

    // set device memory to zero
    checkCudaError(cudaMemset(gpuData.d_locs_device, 0, total_locs_size_device));
    checkCudaError(cudaMemset(gpuData.d_locs_neighbors_device, 0, total_locs_nearestNeighbors_size_device));
    checkCudaError(cudaMemset(gpuData.d_observations_device, 0, total_observations_points_size));
    checkCudaError(cudaMemset(gpuData.d_observations_neighbors_device, 0, total_observations_nearestNeighbors_size));
    checkCudaError(cudaMemset(gpuData.d_cov_device, 0, total_cov_size));
    checkCudaError(cudaMemset(gpuData.d_conditioning_cov_device, 0, total_conditioning_cov_size));
    checkCudaError(cudaMemset(gpuData.d_cross_cov_device, 0, total_cross_cov_size));
    checkCudaError(cudaMemset(gpuData.d_observations_neighbors_copy_device, 0, total_observations_nearestNeighbors_size));
    checkCudaError(cudaMemset(gpuData.d_observations_copy_device, 0, total_observations_points_size));
    checkCudaError(cudaMemset(gpuData.d_mu_correction_device, 0, total_observations_points_size));
    checkCudaError(cudaMemset(gpuData.d_cov_correction_device, 0, total_cov_size)); 
    
    // Prepare to store blocks data for coalesced memory access
    double *locs_blocks_data = new double[total_locs_size_host/sizeof(double)];
    double *locs_nearestNeighbors_data = new double[total_locs_nearestNeighbors_size_host/sizeof(double)];

    size_t locs_index = 0;
    size_t locs_nearestNeighbors_index = 0;
    size_t _total_locs_num_host = total_locs_size_host/sizeof(double)/opts.dim;
    size_t _total_locs_nearestNeighbors_num_host = total_locs_nearestNeighbors_size_host/sizeof(double)/opts.dim;
    size_t _total_locs_num_device = total_locs_size_device/sizeof(double)/opts.dim;
    size_t _total_locs_nearestNeighbors_num_device = total_locs_nearestNeighbors_size_device/sizeof(double)/opts.dim;
    for (size_t i = 0; i < blockInfos.size(); ++i)
    {
        int m_blocks = blockInfos[i].blocks.size();
        int m_nearest_neighbor = blockInfos[i].nearestNeighbors.size();
        // copy locations (coalesced memory access)
        for (int j = 0; j < m_blocks; ++j)
        {
            for (int d = 0; d < opts.dim; ++d)
            {
                locs_blocks_data[locs_index + d * _total_locs_num_host] = blockInfos[i].blocks[j][d];
            }
            locs_index++;
        }
        for (int j = 0; j < m_nearest_neighbor; ++j)
        {
            for (int d = 0; d < opts.dim; ++d)
            {
                locs_nearestNeighbors_data[locs_nearestNeighbors_index + d * _total_locs_nearestNeighbors_num_host] = blockInfos[i].nearestNeighbors[j][d];
            }
            locs_nearestNeighbors_index++;
        }
    }

    // Assign pointers to the beginning of each block's memory and copy data
    double *locs_ptr = gpuData.d_locs_device;
    double *locs_nearestNeighbors_ptr = gpuData.d_locs_neighbors_device;
    double *observations_points_ptr = gpuData.d_observations_device;
    double *observations_nearestNeighbors_ptr = gpuData.d_observations_neighbors_device;
    double *cov_ptr = gpuData.d_cov_device;
    double *conditioning_cov_ptr = gpuData.d_conditioning_cov_device;
    double *cross_cov_ptr = gpuData.d_cross_cov_device;
    double *observations_neighbors_copy_ptr = gpuData.d_observations_neighbors_copy_device;
    double *observations_copy_ptr = gpuData.d_observations_copy_device;
    double *mu_correction_ptr = gpuData.d_mu_correction_device;
    double *cov_correction_ptr = gpuData.d_cov_correction_device;

    // calculate size, and GPU pointers array
    size_t index_locs = 0;
    size_t index_locs_nearestNeighbors = 0;
    size_t block_num = blockInfos.size();



    for (size_t i = 0; i < block_num; ++i)
    {   
        int m_blocks = blockInfos[i].blocks.size();
        int m_nearest_neighbor = blockInfos[i].nearestNeighbors.size();
        for (int d = 0; d < opts.dim; ++d){
            gpuData.h_locs_array[i + block_num * d] = locs_ptr + block_num * d;
            gpuData.h_locs_neighbors_array[i + block_num * d] = locs_nearestNeighbors_ptr + block_num * d;
        }
        gpuData.h_observations_array[i] = observations_points_ptr;
        gpuData.h_observations_neighbors_array[i] = observations_nearestNeighbors_ptr;
        gpuData.h_cov_array[i] = cov_ptr;
        gpuData.h_conditioning_cov_array[i] = conditioning_cov_ptr;
        gpuData.h_cross_cov_array[i] = cross_cov_ptr;
        gpuData.h_observations_neighbors_copy_array[i] = observations_neighbors_copy_ptr;
        gpuData.h_observations_copy_array[i] = observations_copy_ptr;
        gpuData.h_mu_correction_array[i] = mu_correction_ptr;
        gpuData.h_cov_correction_array[i] = cov_correction_ptr;
        // (observations)
        checkCudaError(cudaMemcpy(observations_points_ptr, 
                                   blockInfos[i].observations_blocks.data(), 
                                   blockInfos[i].observations_blocks.size() * sizeof(double), 
                                   cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(observations_nearestNeighbors_ptr, 
                                   blockInfos[i].observations_nearestNeighbors.data(), 
                                   blockInfos[i].observations_nearestNeighbors.size() * sizeof(double), 
                                   cudaMemcpyHostToDevice));
        // copy the locations from the host to the device (locations + observations)
        // copy locations (coalesced memory access)
        for (int d = 0; d < opts.dim; ++d){
            checkCudaError(cudaMemcpy(locs_ptr + d * _total_locs_num_device, 
                                   locs_blocks_data + index_locs + d * _total_locs_num_host, 
                                   m_blocks * sizeof(double), 
                                   cudaMemcpyHostToDevice));
            checkCudaError(cudaMemcpy(locs_nearestNeighbors_ptr + d * _total_locs_nearestNeighbors_num_device, 
                                   locs_nearestNeighbors_data + index_locs_nearestNeighbors + d * _total_locs_nearestNeighbors_num_host, 
                                   m_nearest_neighbor * sizeof(double), 
                                   cudaMemcpyHostToDevice));
        }
        // next pointer
        locs_ptr += gpuData.ldda_locs[i];
        locs_nearestNeighbors_ptr += gpuData.ldda_neighbors[i];
        observations_points_ptr += gpuData.ldda_locs[i];
        observations_nearestNeighbors_ptr += gpuData.ldda_neighbors[i];
        cov_ptr += gpuData.ldda_cov[i] * m_blocks;
        conditioning_cov_ptr += gpuData.ldda_conditioning_cov[i] * m_nearest_neighbor;
        cross_cov_ptr += gpuData.ldda_conditioning_cov[i] * m_blocks;
        // index update
        index_locs += m_blocks;
        index_locs_nearestNeighbors += m_nearest_neighbor;
        // copy update
        observations_neighbors_copy_ptr += gpuData.ldda_neighbors[i];
        observations_copy_ptr += gpuData.ldda_locs[i];
        mu_correction_ptr += gpuData.ldda_locs[i];
        cov_correction_ptr += gpuData.ldda_cov[i] * m_blocks;
    }

    // copy data array to the GPU
    checkCudaError(cudaMemcpy(gpuData.d_locs_array, 
               gpuData.h_locs_array, 
               blockInfos.size() * opts.dim * sizeof(double *), 
               cudaMemcpyHostToDevice));   
    checkCudaError(cudaMemcpy(gpuData.d_locs_neighbors_array, 
               gpuData.h_locs_neighbors_array, 
               blockInfos.size() * opts.dim * sizeof(double *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_observations_points_array, 
               gpuData.h_observations_array, 
               blockInfos.size() * sizeof(double *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_observations_neighbors_array, 
               gpuData.h_observations_neighbors_array, 
               blockInfos.size() * sizeof(double *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_cov_array, 
               gpuData.h_cov_array, 
               blockInfos.size() * sizeof(double *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_conditioning_cov_array, 
               gpuData.h_conditioning_cov_array, 
               blockInfos.size() * sizeof(double *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_cross_cov_array, 
               gpuData.h_cross_cov_array, 
               blockInfos.size() * sizeof(double *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_observations_neighbors_copy_array, 
               gpuData.h_observations_neighbors_copy_array, 
               blockInfos.size() * sizeof(double *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_observations_copy_array, 
               gpuData.h_observations_copy_array, 
               blockInfos.size() * sizeof(double *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_mu_correction_array, 
               gpuData.h_mu_correction_array,  
               blockInfos.size() * sizeof(double *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_cov_correction_array, 
               gpuData.h_cov_correction_array,  
               blockInfos.size() * sizeof(double *), 
               cudaMemcpyHostToDevice));

    // copy data from host to device
    size_t batchCount = gpuData.ldda_locs.size() - 1;
    // allocate memory for magma use
    checkMagmaError(magma_imalloc(&gpuData.dinfo_magma, batchCount + 1));
    // Set dinfo_magma to 0
    checkCudaError(cudaMemset(gpuData.dinfo_magma, 0, (batchCount + 1) * sizeof(magma_int_t)));
    checkCudaError(cudaMalloc((void**)&gpuData.d_ldda_locs, (batchCount + 1) * sizeof(int)));
    checkCudaError(cudaMalloc((void**)&gpuData.d_ldda_neighbors, (batchCount + 1) * sizeof(int)));
    checkCudaError(cudaMalloc((void**)&gpuData.d_ldda_cov, (batchCount + 1) * sizeof(int)));
    checkCudaError(cudaMalloc((void**)&gpuData.d_ldda_cross_cov, (batchCount + 1) * sizeof(int)));
    checkCudaError(cudaMalloc((void**)&gpuData.d_ldda_conditioning_cov, (batchCount + 1) * sizeof(int)));
    checkCudaError(cudaMalloc((void**)&gpuData.d_lda_locs, (batchCount + 1) * sizeof(int)));
    checkCudaError(cudaMalloc((void**)&gpuData.d_lda_locs_neighbors, (batchCount + 1) * sizeof(int)));
    checkCudaError(cudaMalloc((void**)&gpuData.d_const1, (batchCount + 1) * sizeof(int)));

    // set values
    magma_setvector(batchCount, sizeof(int), gpuData.ldda_locs.data(), 1, gpuData.d_ldda_locs, 1, opts.queue);
    magma_setvector(batchCount, sizeof(int), gpuData.ldda_neighbors.data(), 1, gpuData.d_ldda_neighbors, 1, opts.queue);
    magma_setvector(batchCount, sizeof(int), gpuData.ldda_cov.data(), 1, gpuData.d_ldda_cov, 1, opts.queue);
    magma_setvector(batchCount, sizeof(int), gpuData.ldda_cross_cov.data(), 1, gpuData.d_ldda_cross_cov, 1, opts.queue);
    magma_setvector(batchCount, sizeof(int), gpuData.ldda_conditioning_cov.data(), 1, gpuData.d_ldda_conditioning_cov, 1, opts.queue);
    magma_setvector(batchCount, sizeof(int), gpuData.lda_locs.data(), 1, gpuData.d_lda_locs, 1, opts.queue);
    magma_setvector(batchCount, sizeof(int), gpuData.lda_locs_neighbors.data(), 1, gpuData.d_lda_locs_neighbors, 1, opts.queue);
    magma_setvector(batchCount, sizeof(int), gpuData.h_const1.data(), 1, gpuData.d_const1, 1, opts.queue);

    // get the max. dimensions for the trsm
    magma_imax_size_2(gpuData.d_lda_locs_neighbors, gpuData.d_lda_locs, batchCount, opts.queue);
    magma_getvector(1, sizeof(magma_int_t), &gpuData.d_lda_locs_neighbors[batchCount], 
                    1, &gpuData.max_m, 1, opts.queue);
    magma_getvector(1, sizeof(magma_int_t), &gpuData.d_lda_locs[batchCount], 
                    1, &gpuData.max_n1, 1, opts.queue);
    magma_imax_size_2(gpuData.d_lda_locs_neighbors, gpuData.d_const1, batchCount, opts.queue);
    magma_getvector(1, sizeof(magma_int_t), &gpuData.d_const1[batchCount], 
                    1, &gpuData.max_n2, 1, opts.queue);
    delete[] locs_blocks_data;
    delete[] locs_nearestNeighbors_data;

    return gpuData;
}

// Function to perform computation on the GPU
double performComputationOnGPU(const GpuData &gpuData, const std::vector<double> &theta, const Opts &opts)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Set the GPU
    checkCudaError(cudaSetDevice(opts.gpu_id));

    // set the stream
    cudaStream_t stream=opts.stream;
    magma_queue_t queue = opts.queue;
    
    size_t batchCount = gpuData.ldda_locs.size() - 1;
    magma_int_t *dinfo_magma = gpuData.dinfo_magma;
    int *d_ldda_locs = gpuData.d_ldda_locs;
    int *d_ldda_neighbors = gpuData.d_ldda_neighbors;
    int *d_ldda_cov = gpuData.d_ldda_cov;
    int *d_ldda_cross_cov = gpuData.d_ldda_cross_cov;
    int *d_ldda_conditioning_cov = gpuData.d_ldda_conditioning_cov;
    int *d_lda_locs = gpuData.d_lda_locs;
    int *d_lda_locs_neighbors = gpuData.d_lda_locs_neighbors;
    int *d_const1 = gpuData.d_const1;
    magma_int_t max_m = gpuData.max_m;
    magma_int_t max_n1 = gpuData.max_n1;
    magma_int_t max_n2 = gpuData.max_n2;
    int range_offset = opts.range_offset;


    // copy the data from the device to the device
    checkCudaError(cudaMemcpy(gpuData.d_observations_neighbors_copy_device, 
                                   gpuData.d_observations_neighbors_device, 
                                   gpuData.total_observations_neighbors_size, 
                                   cudaMemcpyDeviceToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_observations_copy_device, 
                                   gpuData.d_observations_device, 
                                   gpuData.total_observations_points_size, 
                                   cudaMemcpyDeviceToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_range_device, 
                                theta.data() + range_offset, 
                                opts.dim * sizeof(double), 
                                cudaMemcpyHostToDevice));
    checkCudaError(cudaStreamSynchronize(stream));

    // Use the data on the GPU for computation
    // 1. generate the covariance matrix, cross covariance matrix, conditioning covariance matrix
    // take record of the time
    // for (size_t k = 0; k < opts.dim; ++k){
    //     magma_dprint_gpu(gpuData.lda_locs_neighbors[1], 1, gpuData.h_locs_neighbors_array[1] + k * gpuData.total_locs_neighbors_num_device, gpuData.ldda_conditioning_cov[1], queue);
    // }
    // for (size_t i = 0; i < batchCount; ++i)
    // {   
    //     // print h_locs_array[i]
    //     // std::cout << "before cholesky factorization" << std::endl;
    //     // magma_dprint_gpu_custom(gpuData.lda_locs[0], opts.dim, gpuData.h_locs_array[0], gpuData.ldda_cov[0], queue, 5);
    //     compute_covariance(gpuData.h_locs_array[i],
    //                 gpuData.lda_locs[i], 1, gpuData.total_locs_num_device,
    //                 gpuData.h_locs_array[i],
    //                 gpuData.lda_locs[i], 1, gpuData.total_locs_num_device,
    //                 gpuData.h_cov_array[i], gpuData.ldda_cov[i], gpuData.lda_locs[i],
    //                 opts.dim, theta, gpuData.d_range_device, true, stream, opts);
    //     compute_covariance(gpuData.h_locs_neighbors_array[i], 
    //                 gpuData.lda_locs_neighbors[i], 1, gpuData.total_locs_neighbors_num_device,
    //                 gpuData.h_locs_array[i],
    //                 gpuData.lda_locs[i], 1, gpuData.total_locs_num_device,
    //                 gpuData.h_cross_cov_array[i], gpuData.ldda_cross_cov[i], gpuData.lda_locs[i],
    //                 opts.dim, theta, gpuData.d_range_device, false, stream, opts);
    //     compute_covariance(gpuData.h_locs_neighbors_array[i],
    //                 gpuData.lda_locs_neighbors[i], 1, gpuData.total_locs_neighbors_num_device,
    //                 gpuData.h_locs_neighbors_array[i], 
    //                 gpuData.lda_locs_neighbors[i], 1, gpuData.total_locs_neighbors_num_device,
    //                 gpuData.h_conditioning_cov_array[i], gpuData.ldda_conditioning_cov[i], gpuData.lda_locs_neighbors[i],
    //                 opts.dim, theta, gpuData.d_range_device, true, stream, opts);
    //     // Synchronize to make sure the kernel has finished
    //     checkCudaError(cudaStreamSynchronize(stream));
    // }    
    // std::cout << "gpuData.lda_locs[0]: " << gpuData.lda_locs[0] << std::endl;
    // magma_dprint_gpu_custom(gpuData.lda_locs_neighbors[1], gpuData.lda_locs_neighbors[1], gpuData.h_conditioning_cov_array[1], gpuData.ldda_conditioning_cov[1], queue, 10);
    // magma_dprint_gpu_custom(gpuData.lda_locs[0], gpuData.lda_locs[0], gpuData.h_cov_array[0], gpuData.ldda_cov[0], queue, 5);
    
    compute_covariance_vbatched(gpuData.d_locs_array,
                gpuData.d_lda_locs, 1, gpuData.total_locs_num_device,
                gpuData.d_locs_array,
                gpuData.d_lda_locs, 1, gpuData.total_locs_num_device,
                gpuData.d_cov_array, gpuData.d_ldda_cov, gpuData.d_lda_locs,
                batchCount,
                opts.dim, theta, gpuData.d_range_device, true, stream, opts);
    compute_covariance_vbatched(gpuData.d_locs_neighbors_array, 
                gpuData.d_lda_locs_neighbors, 1, gpuData.total_locs_neighbors_num_device,
                gpuData.d_locs_array,
                gpuData.d_lda_locs, 1, gpuData.total_locs_num_device,
                gpuData.d_cross_cov_array, gpuData.d_ldda_cross_cov, gpuData.d_lda_locs,
                batchCount,
                opts.dim, theta, gpuData.d_range_device, false, stream, opts);
    compute_covariance_vbatched(gpuData.d_locs_neighbors_array,
                gpuData.d_lda_locs_neighbors, 1, gpuData.total_locs_neighbors_num_device,
                gpuData.d_locs_neighbors_array, 
                gpuData.d_lda_locs_neighbors, 1, gpuData.total_locs_neighbors_num_device,
                gpuData.d_conditioning_cov_array, gpuData.d_ldda_conditioning_cov, gpuData.d_lda_locs_neighbors,
                batchCount,
                opts.dim, theta, gpuData.d_range_device, true, stream, opts);
    // Synchronize to make sure the kernel has finished
    checkCudaError(cudaStreamSynchronize(stream));

    // magma_dprint_gpu_custom(gpuData.lda_locs_neighbors[1], gpuData.lda_locs_neighbors[1], gpuData.h_conditioning_cov_array[1], gpuData.ldda_conditioning_cov[1], queue, 10);
    
    // 2. perform the computation
    // 2.1 compute the correction term for mean and variance (i.e., Schur complement)
    // checkMagmaError(magma_dpotrf_vbatched(MagmaLower, d_lda_locs_neighbors,
    //                     gpuData.d_conditioning_cov_array, d_ldda_conditioning_cov,
    //                     dinfo_magma, batchCount, queue));
    checkMagmaError(magma_dpotrf_vbatched_max_nocheck(
            MagmaLower, d_lda_locs_neighbors,
            gpuData.d_conditioning_cov_array, d_ldda_conditioning_cov,
            dinfo_magma, batchCount, max_m, queue));
    

    // trsm
    // magma_dprint_gpu_custom(gpuData.lda_locs_neighbors[1], gpuData.lda_locs[1], gpuData.h_cross_cov_array[1], gpuData.ldda_cross_cov[1], queue, 10);
    // magma_dprint_gpu(gpuData.lda_locs_neighbors[1], gpuData.lda_locs[1], gpuData.h_cross_cov_array[1], gpuData.ldda_cross_cov[1], queue);
    // magma_dprint_gpu(gpuData.lda_locs_neighbors[1], gpuData.lda_locs_neighbors[1], gpuData.h_conditioning_cov_array[1], gpuData.ldda_conditioning_cov[1], queue);
    magmablas_dtrsm_vbatched_max_nocheck(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit, 
                        max_m, max_n1, 
                        d_lda_locs_neighbors, d_lda_locs,
                        1.,
                        gpuData.d_conditioning_cov_array, d_ldda_conditioning_cov,
                        gpuData.d_cross_cov_array, d_ldda_cross_cov,
                        batchCount, queue);
    magmablas_dtrsm_vbatched_max_nocheck(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit, 
                        max_m, max_n2, 
                        d_lda_locs_neighbors, d_const1,
                        1.,
                        gpuData.d_conditioning_cov_array, d_ldda_conditioning_cov,
                        gpuData.d_observations_neighbors_copy_array, d_ldda_neighbors,
                        batchCount, queue);
    // gemm
    // magma_dprint_gpu(gpuData.lda_locs_neighbors[1], gpuData.lda_locs[1], gpuData.h_cross_cov_array[1], gpuData.ldda_cross_cov[1], queue);
    magmablas_dgemm_vbatched_max_nocheck(MagmaTrans, MagmaNoTrans,
                             d_lda_locs, d_lda_locs, d_lda_locs_neighbors,
                             1, gpuData.d_cross_cov_array, d_ldda_cross_cov,
                                gpuData.d_cross_cov_array, d_ldda_cross_cov,
                             0, gpuData.d_cov_correction_array, d_ldda_cov,
                             batchCount, 
                             max_n1, max_n1, max_m, 
                             queue);
    magmablas_dgemm_vbatched_max_nocheck(MagmaTrans, MagmaNoTrans,
                             d_lda_locs, d_const1, d_lda_locs_neighbors,
                             1, gpuData.d_cross_cov_array, d_ldda_cross_cov,
                                gpuData.d_observations_neighbors_copy_array, d_ldda_neighbors,
                             0, gpuData.d_mu_correction_array, d_ldda_locs,
                             batchCount, 
                             max_n1, max_n2, max_m,
                             queue);
    // magma_dprint_gpu_custom(gpuData.lda_locs_neighbors[1], gpuData.lda_locs[1], gpuData.h_cross_cov_array[1], gpuData.ldda_cross_cov[1], queue, 10);
    checkCudaError(cudaStreamSynchronize(stream));
    // std::cout << "before cholesky factorization" << std::endl;
    // magma_dprint_gpu(gpuData.lda_locs[0], gpuData.lda_locs[0], gpuData.h_cov_array[0], gpuData.ldda_cov[0], queue);
    // magma_dprint_gpu(gpuData.lda_locs[1], gpuData.lda_locs[1], gpuData.h_cov_correction_array[0], gpuData.ldda_cov[1], queue);
    // 2.2 compute the conditional mean and variance
    for (size_t i = 0; i < batchCount; ++i){
        // compute conditional variance
        magmablas_dgeadd(gpuData.lda_locs[i], gpuData.lda_locs[i],
                        -1.,
                        gpuData.h_cov_correction_array[i], gpuData.ldda_locs[i], 
                        gpuData.h_cov_array[i], gpuData.ldda_cov[i],
                        queue);
        // compute conditional mean
        magmablas_dgeadd(gpuData.lda_locs[i], 1,
                        -1.,
                        gpuData.h_mu_correction_array[i], gpuData.ldda_locs[i], 
                        gpuData.h_observations_copy_array[i], gpuData.ldda_locs[i],
                        queue);
        // print h_mu_correction_array[i]
        // print gpuData.ldda_locs[i]
        // std::cout << "gpuData.ldda_locs[" << i << "]: " << gpuData.ldda_locs[i] << ", gpuData.lda_locs[" << i << "]: " << gpuData.lda_locs[i] << std::endl;
        // magma_dprint_gpu(gpuData.lda_locs[i], 1, gpuData.h_mu_correction_array[i], gpuData.ldda_locs[i], queue);
    }
    checkCudaError(cudaStreamSynchronize(stream));

    // 2.3 compute the log-likelihood
    // std::cout << "before cholesky factorization" << std::endl;
    // magma_dprint_gpu(gpuData.lda_locs[0], gpuData.lda_locs[0], gpuData.h_cov_correction_array[0], gpuData.ldda_cov[0], queue);
    // magma_dprint_gpu(gpuData.lda_locs[0], gpuData.lda_locs[0], gpuData.h_cov_array[0], gpuData.ldda_cov[0], queue);
    // copy and print dinfo_magma
    checkMagmaError(magma_dpotrf_vbatched(
            MagmaLower, d_lda_locs,
            gpuData.d_cov_array, d_ldda_cov,
            dinfo_magma, batchCount, queue));
    // std::cout << "after cholesky factorization" << std::endl;
    // std::cout << "gpuData.lda_locs[0]: " << gpuData.lda_locs[0] << std::endl;
    // magma_dprint_gpu(gpuData.lda_locs[0], 5, gpuData.h_cov_array[0], gpuData.ldda_cov[0], queue);
    magmablas_dtrsm_vbatched(
        MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit,
        d_lda_locs, d_const1, 1.,
        gpuData.d_cov_array, d_ldda_cov,
        gpuData.d_observations_copy_array, d_ldda_locs,
        batchCount, queue);
    checkCudaError(cudaStreamSynchronize(stream));

    // norm for all blocks
    double norm2_item = norm2_batch(d_lda_locs, gpuData.d_observations_copy_array, d_ldda_locs, batchCount, stream);
    // determinant for all blocks
    double log_det_item = log_det_batch(d_lda_locs, gpuData.d_cov_array, d_ldda_cov, batchCount, stream);
    // sum for log-likelihood
    double log_likelihood = -0.5 *(
        log_det_item + norm2_item // the constant term is removed for simplicity
    );
    //print log_det_item and norm2_item
    // std::cout << "batchcount: " << batchCount << ", log_det_item: " << log_det_item << ", norm2_item: " << norm2_item << std::endl;
    double log_likelihood_all = 0;
    // mpi sum for log-likelihood
    MPI_Allreduce(&log_likelihood, &log_likelihood_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    return log_likelihood_all;
}

// Function to clean up GPU memory
void cleanupGpuMemory(GpuData &gpuData)
{
    cudaFree(gpuData.d_cov_device);
    cudaFree(gpuData.d_conditioning_cov_device);
    cudaFree(gpuData.d_cross_cov_device);
    cudaFree(gpuData.d_observations_device);
    cudaFree(gpuData.d_observations_neighbors_device);
    cudaFree(gpuData.d_observations_neighbors_copy_device);
    cudaFree(gpuData.d_observations_copy_device);
    cudaFree(gpuData.d_mu_correction_device);
    cudaFree(gpuData.d_cov_correction_device);

    delete[] gpuData.h_cov_array;
    delete[] gpuData.h_conditioning_cov_array;
    delete[] gpuData.h_cross_cov_array;
    delete[] gpuData.h_locs_array;
    delete[] gpuData.h_locs_neighbors_array;
    delete[] gpuData.h_observations_array;
    delete[] gpuData.h_observations_neighbors_array;
    delete[] gpuData.h_observations_copy_array;
    delete[] gpuData.h_mu_correction_array;
    delete[] gpuData.h_cov_correction_array;
    delete[] gpuData.h_observations_neighbors_copy_array;
}

// calculate the total flops
double gflopsTotal(const GpuData &gpuData, const Opts &opts)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    size_t batchCount = gpuData.ldda_locs.size() - 1;
    double gflops = 0;
    for (size_t i = 0; i < batchCount; ++i)
    {
        // 1. matrix generation
        gflops += (3 * opts.dim + 11) * (gpuData.lda_locs[i] * gpuData.lda_locs[i] + 
                    gpuData.lda_locs_neighbors[i] * gpuData.lda_locs_neighbors[i] + 
                    gpuData.lda_locs[i] * gpuData.lda_locs_neighbors[i]) / 1e9;
        
        // 2. conditioning corretion
        // // cholesky factorization
        gflops += FLOPS_DPOTRF(gpuData.lda_locs_neighbors[i]) / 1e9;
        // // trsm
        gflops += FLOPS_DTRSM(MagmaLeft, gpuData.lda_locs_neighbors[i], gpuData.lda_locs[i]) / 1e9;
        gflops += FLOPS_DTRSM(MagmaLeft, gpuData.lda_locs_neighbors[i], 1) / 1e9;
        // // matrix multiplication
        gflops += FLOPS_DGEMM(gpuData.lda_locs[i], gpuData.lda_locs[i], gpuData.lda_locs_neighbors[i]) / 1e9;
        gflops += FLOPS_DGEMM(gpuData.lda_locs[i], 1, gpuData.lda_locs_neighbors[i]) / 1e9;
        // // addition
        gflops += FLOPS_DAXPY(gpuData.lda_locs[i] * gpuData.lda_locs[i]) / 1e9;
        gflops += FLOPS_DAXPY(gpuData.lda_locs[i]) / 1e9;
        
        // 3. log-likelihood calculation
        // cholesky factorization + trsv + norm + determinant
        gflops += FLOPS_DPOTRF(gpuData.lda_locs[i]) / 1e9;
        gflops += FLOPS_DTRSM(MagmaLeft, gpuData.lda_locs[i], 1) / 1e9;
        gflops += 4 * FLOPS_DAXPY(gpuData.lda_locs[i]) / 1e9;
    }
    double total_gflops = 0;
    // mpi sum for gflops
    MPI_Allreduce(&gflops, &total_gflops, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Total Gflops: " << total_gflops << std::endl;
    }

    return total_gflops;
}