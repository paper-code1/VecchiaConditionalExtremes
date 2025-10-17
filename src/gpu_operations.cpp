#include <iostream>
#include <mpi.h>
#include <magma_v2.h>
#include <sys/cdefs.h>
#include "gpu_operations.h"
#include "gpu_covariance.h"
#include "flops.h"
#include "error_checking.h"
#include "magma_dprint_gpu.h"
#include <type_traits>

template <typename Real>
struct MagmaOps;

template <>
struct MagmaOps<double> {
    static inline void potrf_neighbors(magma_uplo_t uplo, magma_int_t* d_n, double** d_A_array, magma_int_t* d_ldda, magma_int_t* dinfo, magma_int_t batchCount, magma_int_t max_n, magma_queue_t queue) {
        magma_dpotrf_vbatched_max_nocheck(uplo, d_n, d_A_array, d_ldda, dinfo, batchCount, max_n, queue);
    }
    static inline void trsm_max(magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
                                magma_int_t max_m, magma_int_t max_n,
                                magma_int_t* d_m, magma_int_t* d_n,
                                double alpha,
                                double** d_A_array, magma_int_t* d_ldda_A,
                                double** d_B_array, magma_int_t* d_ldda_B,
                                magma_int_t batchCount, magma_queue_t queue) {
        magmablas_dtrsm_vbatched_max_nocheck(side, uplo, transA, diag, max_m, max_n, d_m, d_n, alpha, d_A_array, d_ldda_A, d_B_array, d_ldda_B, batchCount, queue);
    }
    static inline void gemm_max(magma_trans_t transA, magma_trans_t transB,
                                magma_int_t* d_m, magma_int_t* d_n, magma_int_t* d_k,
                                double alpha,
                                double const* const* d_A_array, magma_int_t* d_ldda_A,
                                double const* const* d_B_array, magma_int_t* d_ldda_B,
                                double beta,
                                double** d_C_array, magma_int_t* d_ldda_C,
                                magma_int_t batchCount,
                                magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
                                magma_queue_t queue) {
        magmablas_dgemm_vbatched_max_nocheck(transA, transB, d_m, d_n, d_k, alpha, d_A_array, d_ldda_A, d_B_array, d_ldda_B, beta, d_C_array, d_ldda_C, batchCount, max_m, max_n, max_k, queue);
    }
    static inline void potrf_final(magma_uplo_t uplo, magma_int_t* d_n, double** d_A_array, magma_int_t* d_ldda, magma_int_t* dinfo, magma_int_t batchCount, magma_queue_t queue) {
        magma_dpotrf_vbatched(uplo, d_n, d_A_array, d_ldda, dinfo, batchCount, queue);
    }
    static inline void trsm_final(magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
                                  magma_int_t* d_m, magma_int_t* d_n, double alpha,
                                  double** d_A_array, magma_int_t* d_ldda_A,
                                  double** d_B_array, magma_int_t* d_ldda_B,
                                  magma_int_t batchCount, magma_queue_t queue) {
        magmablas_dtrsm_vbatched(side, uplo, transA, diag, d_m, d_n, alpha, d_A_array, d_ldda_A, d_B_array, d_ldda_B, batchCount, queue);
    }
};

template <>
struct MagmaOps<float> {
    static inline void potrf_neighbors(magma_uplo_t uplo, magma_int_t* d_n, float** d_A_array, magma_int_t* d_ldda, magma_int_t* dinfo, magma_int_t batchCount, magma_int_t max_n, magma_queue_t queue) {
        magma_spotrf_vbatched_max_nocheck(uplo, d_n, d_A_array, d_ldda, dinfo, batchCount, max_n, queue);
    }
    static inline void trsm_max(magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
                                magma_int_t max_m, magma_int_t max_n,
                                magma_int_t* d_m, magma_int_t* d_n,
                                float alpha,
                                float** d_A_array, magma_int_t* d_ldda_A,
                                float** d_B_array, magma_int_t* d_ldda_B,
                                magma_int_t batchCount, magma_queue_t queue) {
        magmablas_strsm_vbatched_max_nocheck(side, uplo, transA, diag, max_m, max_n, d_m, d_n, alpha, d_A_array, d_ldda_A, d_B_array, d_ldda_B, batchCount, queue);
    }
    static inline void gemm_max(magma_trans_t transA, magma_trans_t transB,
                                magma_int_t* d_m, magma_int_t* d_n, magma_int_t* d_k,
                                float alpha,
                                float const* const* d_A_array, magma_int_t* d_ldda_A,
                                float const* const* d_B_array, magma_int_t* d_ldda_B,
                                float beta,
                                float** d_C_array, magma_int_t* d_ldda_C,
                                magma_int_t batchCount,
                                magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
                                magma_queue_t queue) {
        magmablas_sgemm_vbatched_max_nocheck(transA, transB, d_m, d_n, d_k, alpha, d_A_array, d_ldda_A, d_B_array, d_ldda_B, beta, d_C_array, d_ldda_C, batchCount, max_m, max_n, max_k, queue);
    }
    static inline void potrf_final(magma_uplo_t uplo, magma_int_t* d_n, float** d_A_array, magma_int_t* d_ldda, magma_int_t* dinfo, magma_int_t batchCount, magma_queue_t queue) {
        magma_spotrf_vbatched(uplo, d_n, d_A_array, d_ldda, dinfo, batchCount, queue);
    }
    static inline void trsm_final(magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
                                  magma_int_t* d_m, magma_int_t* d_n, float alpha,
                                  float** d_A_array, magma_int_t* d_ldda_A,
                                  float** d_B_array, magma_int_t* d_ldda_B,
                                  magma_int_t batchCount, magma_queue_t queue) {
        magmablas_strsm_vbatched(side, uplo, transA, diag, d_m, d_n, alpha, d_A_array, d_ldda_A, d_B_array, d_ldda_B, batchCount, queue);
    }
};

// Function to copy data from CPU to GPU and allocate memory with leading dimensions
template <typename Real>
GpuDataT<Real> copyDataToGPU(const Opts &opts, const std::vector<BlockInfo> &blockInfos)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Set the GPU
    checkCudaError(cudaSetDevice(opts.gpu_id));

    GpuDataT<Real> gpuData;
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
    gpuData.h_locs_array = new Real *[blockInfos.size() * opts.dim];
    gpuData.h_locs_neighbors_array = new Real *[blockInfos.size() * opts.dim];
    gpuData.h_observations_array = new Real *[blockInfos.size()];
    gpuData.h_observations_neighbors_array = new Real *[blockInfos.size()];
    gpuData.h_cov_array = new Real *[blockInfos.size()];
    gpuData.h_cross_cov_array = new Real *[blockInfos.size()];
    gpuData.h_conditioning_cov_array = new Real *[blockInfos.size()];
    gpuData.h_observations_neighbors_copy_array = new Real *[blockInfos.size()];
    gpuData.h_observations_copy_array = new Real *[blockInfos.size()];
    gpuData.h_mu_correction_array = new Real *[blockInfos.size()];
    gpuData.h_cov_correction_array = new Real *[blockInfos.size()];

    // array of pointers for the device
    checkCudaError(cudaMalloc(&gpuData.d_locs_array, blockInfos.size() * opts.dim * sizeof(Real *)));
    checkCudaError(cudaMalloc(&gpuData.d_locs_neighbors_array, blockInfos.size() * opts.dim * sizeof(Real *)));
    checkCudaError(cudaMalloc(&gpuData.d_observations_points_array, blockInfos.size() * sizeof(Real *)));
    checkCudaError(cudaMalloc(&gpuData.d_observations_neighbors_array, blockInfos.size() * sizeof(Real *)));
    checkCudaError(cudaMalloc(&gpuData.d_cov_array, blockInfos.size() * sizeof(Real *)));
    checkCudaError(cudaMalloc(&gpuData.d_cross_cov_array, blockInfos.size() * sizeof(Real *)));
    checkCudaError(cudaMalloc(&gpuData.d_conditioning_cov_array, blockInfos.size() * sizeof(Real *)));
    checkCudaError(cudaMalloc(&gpuData.d_observations_neighbors_copy_array, blockInfos.size() * sizeof(Real *)));
    checkCudaError(cudaMalloc(&gpuData.d_observations_copy_array, blockInfos.size() * sizeof(Real *)));
    checkCudaError(cudaMalloc(&gpuData.d_mu_correction_array, blockInfos.size() * sizeof(Real *)));
    checkCudaError(cudaMalloc(&gpuData.d_cov_correction_array, blockInfos.size() * sizeof(Real *)));
    checkCudaError(cudaMalloc(&gpuData.d_range_device, opts.dim * sizeof(Real)));

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
        total_cov_size += gpuData.ldda_cov[i] * m_blocks * sizeof(Real);
        total_conditioning_cov_size += gpuData.ldda_conditioning_cov[i] * m_nearest_neighbor * sizeof(Real);
        total_cross_cov_size += gpuData.ldda_conditioning_cov[i] * m_blocks * sizeof(Real);
        total_observations_points_size += gpuData.ldda_locs[i] * sizeof(Real);
        total_observations_nearestNeighbors_size += gpuData.ldda_neighbors[i] * sizeof(Real);
        total_locs_size_host += m_blocks * sizeof(Real) * opts.dim;
        total_locs_nearestNeighbors_size_host += m_nearest_neighbor * sizeof(Real) * opts.dim;
        total_locs_size_device += gpuData.ldda_locs[i] * sizeof(Real) * opts.dim;
        total_locs_nearestNeighbors_size_device += gpuData.ldda_neighbors[i] * sizeof(Real) * opts.dim;
    }

    // Allocate contiguous memory on GPU
    gpuData.total_observations_points_size = total_observations_points_size;
    gpuData.total_observations_neighbors_size = total_observations_nearestNeighbors_size;
    gpuData.total_locs_num_device = total_locs_size_device/sizeof(Real)/opts.dim;
    gpuData.total_locs_neighbors_num_device = total_locs_nearestNeighbors_size_device/sizeof(Real)/opts.dim;
    gpuData.total_cov_size_bytes = total_cov_size;
    gpuData.total_cross_cov_size_bytes = total_cross_cov_size;
    gpuData.total_conditioning_cov_size_bytes = total_conditioning_cov_size;
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
    Real *locs_blocks_data = new Real[total_locs_size_host/sizeof(Real)];
    Real *locs_nearestNeighbors_data = new Real[total_locs_nearestNeighbors_size_host/sizeof(Real)];
    // Optional host-side double copies to avoid precision loss when feeding double kernels
    std::vector<double> locs_blocks_data64;
    std::vector<double> locs_nearestNeighbors_data64;

    size_t locs_index = 0;
    size_t locs_nearestNeighbors_index = 0;
    size_t _total_locs_num_host = total_locs_size_host/sizeof(Real)/opts.dim;
    size_t _total_locs_nearestNeighbors_num_host = total_locs_nearestNeighbors_size_host/sizeof(Real)/opts.dim;
    size_t _total_locs_num_device = total_locs_size_device/sizeof(Real)/opts.dim;
    size_t _total_locs_nearestNeighbors_num_device = total_locs_nearestNeighbors_size_device/sizeof(Real)/opts.dim;
    for (size_t i = 0; i < blockInfos.size(); ++i)
    {
        int m_blocks = blockInfos[i].blocks.size();
        int m_nearest_neighbor = blockInfos[i].nearestNeighbors.size();
        // copy locations (coalesced memory access)
        for (int j = 0; j < m_blocks; ++j)
        {
            for (int d = 0; d < opts.dim; ++d)
            {
                locs_blocks_data[locs_index + d * _total_locs_num_host] = static_cast<Real>(blockInfos[i].blocks[j][d]);
                if constexpr (std::is_same<Real, float>::value) {
                    if (locs_blocks_data64.empty()) {
                        locs_blocks_data64.resize((size_t)_total_locs_num_host * opts.dim);
                    }
                    locs_blocks_data64[(size_t)locs_index + (size_t)d * _total_locs_num_host] = blockInfos[i].blocks[j][d];
                }
            }
            locs_index++;
        }
        for (int j = 0; j < m_nearest_neighbor; ++j)
        {
            for (int d = 0; d < opts.dim; ++d)
            {
                locs_nearestNeighbors_data[locs_nearestNeighbors_index + d * _total_locs_nearestNeighbors_num_host] = static_cast<Real>(blockInfos[i].nearestNeighbors[j][d]);
                if constexpr (std::is_same<Real, float>::value) {
                    if (locs_nearestNeighbors_data64.empty()) {
                        locs_nearestNeighbors_data64.resize((size_t)_total_locs_nearestNeighbors_num_host * opts.dim);
                    }
                    locs_nearestNeighbors_data64[(size_t)locs_nearestNeighbors_index + (size_t)d * _total_locs_nearestNeighbors_num_host] = blockInfos[i].nearestNeighbors[j][d];
                }
            }
            locs_nearestNeighbors_index++;
        }
    }

    // Assign pointers to the beginning of each block's memory and copy data
    Real *locs_ptr = gpuData.d_locs_device;
    Real *locs_nearestNeighbors_ptr = gpuData.d_locs_neighbors_device;
    Real *observations_points_ptr = gpuData.d_observations_device;
    Real *observations_nearestNeighbors_ptr = gpuData.d_observations_neighbors_device;
    Real *cov_ptr = gpuData.d_cov_device;
    Real *conditioning_cov_ptr = gpuData.d_conditioning_cov_device;
    Real *cross_cov_ptr = gpuData.d_cross_cov_device;
    Real *observations_neighbors_copy_ptr = gpuData.d_observations_neighbors_copy_device;
    Real *observations_copy_ptr = gpuData.d_observations_copy_device;
    Real *mu_correction_ptr = gpuData.d_mu_correction_device;
    Real *cov_correction_ptr = gpuData.d_cov_correction_device;

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
        // cast observations to Real
        {
            std::vector<Real> tmp_obs(blockInfos[i].observations_blocks.size());
            for (size_t t=0;t<tmp_obs.size();++t) tmp_obs[t] = static_cast<Real>(blockInfos[i].observations_blocks[t]);
            checkCudaError(cudaMemcpy(observations_points_ptr, tmp_obs.data(), tmp_obs.size() * sizeof(Real), cudaMemcpyHostToDevice));
        }
        {
            std::vector<Real> tmp_obs_nn(blockInfos[i].observations_nearestNeighbors.size());
            for (size_t t=0;t<tmp_obs_nn.size();++t) tmp_obs_nn[t] = static_cast<Real>(blockInfos[i].observations_nearestNeighbors[t]);
            checkCudaError(cudaMemcpy(observations_nearestNeighbors_ptr, tmp_obs_nn.data(), tmp_obs_nn.size() * sizeof(Real), cudaMemcpyHostToDevice));
        }
        // copy the locations from the host to the device (locations + observations)
        // copy locations (coalesced memory access)
        for (int d = 0; d < opts.dim; ++d){
            checkCudaError(cudaMemcpy(locs_ptr + d * _total_locs_num_device, 
                                   locs_blocks_data + index_locs + d * _total_locs_num_host, 
                                   m_blocks * sizeof(Real), 
                                   cudaMemcpyHostToDevice));
            checkCudaError(cudaMemcpy(locs_nearestNeighbors_ptr + d * _total_locs_nearestNeighbors_num_device, 
                                   locs_nearestNeighbors_data + index_locs_nearestNeighbors + d * _total_locs_nearestNeighbors_num_host, 
                                   m_nearest_neighbor * sizeof(Real), 
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
               blockInfos.size() * opts.dim * sizeof(Real *), 
               cudaMemcpyHostToDevice));   
    checkCudaError(cudaMemcpy(gpuData.d_locs_neighbors_array, 
               gpuData.h_locs_neighbors_array, 
               blockInfos.size() * opts.dim * sizeof(Real *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_observations_points_array, 
               gpuData.h_observations_array, 
               blockInfos.size() * sizeof(Real *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_observations_neighbors_array, 
               gpuData.h_observations_neighbors_array, 
               blockInfos.size() * sizeof(Real *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_cov_array, 
               gpuData.h_cov_array, 
               blockInfos.size() * sizeof(Real *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_conditioning_cov_array, 
               gpuData.h_conditioning_cov_array, 
               blockInfos.size() * sizeof(Real *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_cross_cov_array, 
               gpuData.h_cross_cov_array, 
               blockInfos.size() * sizeof(Real *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_observations_neighbors_copy_array, 
               gpuData.h_observations_neighbors_copy_array, 
               blockInfos.size() * sizeof(Real *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_observations_copy_array, 
               gpuData.h_observations_copy_array, 
               blockInfos.size() * sizeof(Real *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_mu_correction_array, 
               gpuData.h_mu_correction_array,  
               blockInfos.size() * sizeof(Real *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_cov_correction_array, 
               gpuData.h_cov_correction_array,  
               blockInfos.size() * sizeof(Real *), 
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

    // Mixed-precision setup: allocate double mirrors if baseline Real=float and flags are enabled
    if constexpr (std::is_same<Real, float>::value) {
        bool need_double = (opts.precision == PrecisionType::Float) && (opts.mp_cov_double || opts.mp_schur_double || opts.mp_final_double || opts.mp_all_double_ops);
        if (need_double) {
            size_t blocks = blockInfos.size();
            // allocate contiguous double buffers
            checkCudaError(cudaMalloc(&gpuData.d_locs_device_f64, (size_t)((total_locs_size_device/sizeof(float))*sizeof(double))));
            checkCudaError(cudaMalloc(&gpuData.d_locs_neighbors_device_f64, (size_t)((total_locs_nearestNeighbors_size_device/sizeof(float))*sizeof(double))));
            checkCudaError(cudaMalloc(&gpuData.d_cov_device_f64, (size_t)((total_cov_size/sizeof(float))*sizeof(double))));
            checkCudaError(cudaMalloc(&gpuData.d_cross_cov_device_f64, (size_t)((total_cross_cov_size/sizeof(float))*sizeof(double))));
            checkCudaError(cudaMalloc(&gpuData.d_conditioning_cov_device_f64, (size_t)((total_conditioning_cov_size/sizeof(float))*sizeof(double))));
            checkCudaError(cudaMalloc(&gpuData.d_observations_neighbors_copy_device_f64, (size_t)((total_observations_nearestNeighbors_size/sizeof(float))*sizeof(double))));
            checkCudaError(cudaMalloc(&gpuData.d_observations_copy_device_f64, (size_t)((total_observations_points_size/sizeof(float))*sizeof(double))));
            // persistent double observations (optional, for precision)
            checkCudaError(cudaMalloc(&gpuData.d_observations_neighbors_device_f64, (size_t)((total_observations_nearestNeighbors_size/sizeof(float))*sizeof(double))));
            checkCudaError(cudaMalloc(&gpuData.d_observations_device_f64, (size_t)((total_observations_points_size/sizeof(float))*sizeof(double))));
            checkCudaError(cudaMalloc(&gpuData.d_mu_correction_device_f64, (size_t)((total_observations_points_size/sizeof(float))*sizeof(double))));
            checkCudaError(cudaMalloc(&gpuData.d_cov_correction_device_f64, (size_t)((total_cov_size/sizeof(float))*sizeof(double))));
            checkCudaError(cudaMalloc(&gpuData.d_range_device_f64, opts.dim * sizeof(double)));

            // host arrays for double pointers
            gpuData.h_locs_array_f64 = new double*[blocks * opts.dim];
            gpuData.h_locs_neighbors_array_f64 = new double*[blocks * opts.dim];
            gpuData.h_cov_array_f64 = new double*[blocks];
            gpuData.h_cross_cov_array_f64 = new double*[blocks];
            gpuData.h_conditioning_cov_array_f64 = new double*[blocks];
            gpuData.h_observations_neighbors_copy_array_f64 = new double*[blocks];
            gpuData.h_observations_copy_array_f64 = new double*[blocks];
            gpuData.h_observations_array_f64 = new double*[blocks];
            gpuData.h_observations_neighbors_array_f64 = new double*[blocks];
            gpuData.h_mu_correction_array_f64 = new double*[blocks];
            gpuData.h_cov_correction_array_f64 = new double*[blocks];

            // assign per-batch pointers mirroring float layout
            double *locs_ptr64 = gpuData.d_locs_device_f64;
            double *locs_neighbors_ptr64 = gpuData.d_locs_neighbors_device_f64;
            double *cov_ptr64 = gpuData.d_cov_device_f64;
            double *conditioning_cov_ptr64 = gpuData.d_conditioning_cov_device_f64;
            double *cross_cov_ptr64 = gpuData.d_cross_cov_device_f64;
            double *observations_neighbors_copy_ptr64 = gpuData.d_observations_neighbors_copy_device_f64;
            double *observations_copy_ptr64 = gpuData.d_observations_copy_device_f64;
            double *observations_neighbors_ptr64 = gpuData.d_observations_neighbors_device_f64;
            double *observations_points_ptr64 = gpuData.d_observations_device_f64;
            double *mu_correction_ptr64 = gpuData.d_mu_correction_device_f64;
            double *cov_correction_ptr64 = gpuData.d_cov_correction_device_f64;

            for (size_t i = 0; i < blocks; ++i) {
                int m_blocks = gpuData.lda_locs[i];
                int m_nearest_neighbor = gpuData.lda_locs_neighbors[i];
                for (int d = 0; d < opts.dim; ++d) {
                    gpuData.h_locs_array_f64[i + blocks * d] = locs_ptr64 + blocks * d;
                    gpuData.h_locs_neighbors_array_f64[i + blocks * d] = locs_neighbors_ptr64 + blocks * d;
                }
                gpuData.h_cov_array_f64[i] = cov_ptr64;
                gpuData.h_conditioning_cov_array_f64[i] = conditioning_cov_ptr64;
                gpuData.h_cross_cov_array_f64[i] = cross_cov_ptr64;
                gpuData.h_observations_array_f64[i] = observations_points_ptr64;
                gpuData.h_observations_neighbors_array_f64[i] = observations_neighbors_ptr64;
                gpuData.h_observations_neighbors_copy_array_f64[i] = observations_neighbors_copy_ptr64;
                gpuData.h_observations_copy_array_f64[i] = observations_copy_ptr64;
                gpuData.h_mu_correction_array_f64[i] = mu_correction_ptr64;
                gpuData.h_cov_correction_array_f64[i] = cov_correction_ptr64;

                locs_ptr64 += gpuData.ldda_locs[i];
                locs_neighbors_ptr64 += gpuData.ldda_neighbors[i];
                cov_ptr64 += gpuData.ldda_cov[i] * m_blocks;
                conditioning_cov_ptr64 += gpuData.ldda_conditioning_cov[i] * m_nearest_neighbor;
                cross_cov_ptr64 += gpuData.ldda_conditioning_cov[i] * m_blocks;
                observations_neighbors_copy_ptr64 += gpuData.ldda_neighbors[i];
                observations_copy_ptr64 += gpuData.ldda_locs[i];
                observations_neighbors_ptr64 += gpuData.ldda_neighbors[i];
                observations_points_ptr64 += gpuData.ldda_locs[i];
                mu_correction_ptr64 += gpuData.ldda_locs[i];
                cov_correction_ptr64 += gpuData.ldda_cov[i] * m_blocks;
            }

            // device arrays of double pointers
            checkCudaError(cudaMalloc(&gpuData.d_locs_array_f64, blocks * opts.dim * sizeof(double*)));
            checkCudaError(cudaMalloc(&gpuData.d_locs_neighbors_array_f64, blocks * opts.dim * sizeof(double*)));
            checkCudaError(cudaMalloc(&gpuData.d_cov_array_f64, blocks * sizeof(double*)));
            checkCudaError(cudaMalloc(&gpuData.d_cross_cov_array_f64, blocks * sizeof(double*)));
            checkCudaError(cudaMalloc(&gpuData.d_conditioning_cov_array_f64, blocks * sizeof(double*)));
            checkCudaError(cudaMalloc(&gpuData.d_observations_neighbors_copy_array_f64, blocks * sizeof(double*)));
            checkCudaError(cudaMalloc(&gpuData.d_observations_copy_array_f64, blocks * sizeof(double*)));
            checkCudaError(cudaMalloc(&gpuData.d_observations_neighbors_array_f64, blocks * sizeof(double*)));
            checkCudaError(cudaMalloc(&gpuData.d_observations_points_array_f64, blocks * sizeof(double*)));
            checkCudaError(cudaMalloc(&gpuData.d_mu_correction_array_f64, blocks * sizeof(double*)));
            checkCudaError(cudaMalloc(&gpuData.d_cov_correction_array_f64, blocks * sizeof(double*)));

            // copy host pointer arrays to device
            checkCudaError(cudaMemcpy(gpuData.d_locs_array_f64, gpuData.h_locs_array_f64, blocks * opts.dim * sizeof(double*), cudaMemcpyHostToDevice));
            checkCudaError(cudaMemcpy(gpuData.d_locs_neighbors_array_f64, gpuData.h_locs_neighbors_array_f64, blocks * opts.dim * sizeof(double*), cudaMemcpyHostToDevice));
            checkCudaError(cudaMemcpy(gpuData.d_cov_array_f64, gpuData.h_cov_array_f64, blocks * sizeof(double*), cudaMemcpyHostToDevice));
            checkCudaError(cudaMemcpy(gpuData.d_cross_cov_array_f64, gpuData.h_cross_cov_array_f64, blocks * sizeof(double*), cudaMemcpyHostToDevice));
            checkCudaError(cudaMemcpy(gpuData.d_conditioning_cov_array_f64, gpuData.h_conditioning_cov_array_f64, blocks * sizeof(double*), cudaMemcpyHostToDevice));
            checkCudaError(cudaMemcpy(gpuData.d_observations_neighbors_copy_array_f64, gpuData.h_observations_neighbors_copy_array_f64, blocks * sizeof(double*), cudaMemcpyHostToDevice));
            checkCudaError(cudaMemcpy(gpuData.d_observations_copy_array_f64, gpuData.h_observations_copy_array_f64, blocks * sizeof(double*), cudaMemcpyHostToDevice));
            checkCudaError(cudaMemcpy(gpuData.d_observations_neighbors_array_f64, gpuData.h_observations_neighbors_array_f64, blocks * sizeof(double*), cudaMemcpyHostToDevice));
            checkCudaError(cudaMemcpy(gpuData.d_observations_points_array_f64, gpuData.h_observations_array_f64, blocks * sizeof(double*), cudaMemcpyHostToDevice));
            checkCudaError(cudaMemcpy(gpuData.d_mu_correction_array_f64, gpuData.h_mu_correction_array_f64, blocks * sizeof(double*), cudaMemcpyHostToDevice));
            checkCudaError(cudaMemcpy(gpuData.d_cov_correction_array_f64, gpuData.h_cov_correction_array_f64, blocks * sizeof(double*), cudaMemcpyHostToDevice));

            // populate static locations and observations in double directly from host to avoid precision loss
            // copy locations (coalesced) into double device arrays
            size_t index_locs64 = 0;
            size_t index_locs_nn64 = 0;
            double *locs_ptr64w = gpuData.d_locs_device_f64;
            double *locs_nn_ptr64w = gpuData.d_locs_neighbors_device_f64;
            for (size_t i = 0; i < blocks; ++i) {
                int m_blocks = gpuData.lda_locs[i];
                int m_nearest_neighbor = gpuData.lda_locs_neighbors[i];
                for (int d = 0; d < opts.dim; ++d) {
                    checkCudaError(cudaMemcpy(locs_ptr64w + d * _total_locs_num_device,
                                              locs_blocks_data64.data() + index_locs64 + d * _total_locs_num_host,
                                              (size_t)m_blocks * sizeof(double), cudaMemcpyHostToDevice));
                    checkCudaError(cudaMemcpy(locs_nn_ptr64w + d * _total_locs_nearestNeighbors_num_device,
                                              locs_nearestNeighbors_data64.data() + index_locs_nn64 + d * _total_locs_nearestNeighbors_num_host,
                                              (size_t)m_nearest_neighbor * sizeof(double), cudaMemcpyHostToDevice));
                }
                // observations to double
                {
                    std::vector<double> tmp_obs_d(m_blocks);
                    for (int t=0; t<m_blocks; ++t) tmp_obs_d[t] = (double)blockInfos[i].observations_blocks[t];
                    checkCudaError(cudaMemcpy(observations_points_ptr64, tmp_obs_d.data(), (size_t)m_blocks * sizeof(double), cudaMemcpyHostToDevice));
                }
                {
                    std::vector<double> tmp_obs_nn_d(m_nearest_neighbor);
                    for (int t=0; t<m_nearest_neighbor; ++t) tmp_obs_nn_d[t] = (double)blockInfos[i].observations_nearestNeighbors[t];
                    checkCudaError(cudaMemcpy(observations_neighbors_ptr64, tmp_obs_nn_d.data(), (size_t)m_nearest_neighbor * sizeof(double), cudaMemcpyHostToDevice));
                }
                locs_ptr64w += gpuData.ldda_locs[i];
                locs_nn_ptr64w += gpuData.ldda_neighbors[i];
                index_locs64 += m_blocks;
                index_locs_nn64 += m_nearest_neighbor;
                observations_points_ptr64 += gpuData.ldda_locs[i];
                observations_neighbors_ptr64 += gpuData.ldda_neighbors[i];
            }
        }
    }

    return gpuData;
}

// Function to perform computation on the GPU
template <typename Real>
double performComputationOnGPU(const GpuDataT<Real> &gpuData, const std::vector<double> &theta, Opts &opts)
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
    {
        std::vector<Real> range_host(opts.dim);
        for (int i=0;i<opts.dim;++i) range_host[i] = static_cast<Real>(theta[range_offset + i]);
        checkCudaError(cudaMemcpy(gpuData.d_range_device, range_host.data(), opts.dim * sizeof(Real), cudaMemcpyHostToDevice));
    }
    checkCudaError(cudaStreamSynchronize(stream));

    if constexpr (std::is_same<Real, double>::value) {
        compute_covariance_vbatched<Real>(gpuData.d_locs_array,
                gpuData.d_lda_locs, 1, gpuData.total_locs_num_device,
                gpuData.d_locs_array,
                gpuData.d_lda_locs, 1, gpuData.total_locs_num_device,
                gpuData.d_cov_array, gpuData.d_ldda_cov, gpuData.d_lda_locs,
                batchCount,
                opts.dim, theta, gpuData.d_range_device, true, stream, opts);
        compute_covariance_vbatched<Real>(gpuData.d_locs_neighbors_array, 
                gpuData.d_lda_locs_neighbors, 1, gpuData.total_locs_neighbors_num_device,
                gpuData.d_locs_array,
                gpuData.d_lda_locs, 1, gpuData.total_locs_num_device,
                gpuData.d_cross_cov_array, gpuData.d_ldda_cross_cov, gpuData.d_lda_locs,
                batchCount,
                opts.dim, theta, gpuData.d_range_device, false, stream, opts);
        compute_covariance_vbatched<Real>(gpuData.d_locs_neighbors_array,
                gpuData.d_lda_locs_neighbors, 1, gpuData.total_locs_neighbors_num_device,
                gpuData.d_locs_neighbors_array, 
                gpuData.d_lda_locs_neighbors, 1, gpuData.total_locs_neighbors_num_device,
                gpuData.d_conditioning_cov_array, gpuData.d_ldda_conditioning_cov, gpuData.d_lda_locs_neighbors,
                batchCount,
                opts.dim, theta, gpuData.d_range_device, true, stream, opts);
    
    // cholesky factorization
    MagmaOps<Real>::potrf_neighbors(MagmaLower, d_lda_locs_neighbors, gpuData.d_conditioning_cov_array, d_ldda_conditioning_cov, dinfo_magma, batchCount, max_m, queue);
    // trsm
    MagmaOps<Real>::trsm_max(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit,
                        max_m, max_n1,
                        d_lda_locs_neighbors, d_lda_locs,
                        (Real)1.0,
                        gpuData.d_conditioning_cov_array, d_ldda_conditioning_cov,
                        gpuData.d_cross_cov_array, d_ldda_cross_cov,
                        batchCount, queue);
    MagmaOps<Real>::trsm_max(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit,
                        max_m, max_n2,
                        d_lda_locs_neighbors, d_const1,
                        (Real)1.0,
                        gpuData.d_conditioning_cov_array, d_ldda_conditioning_cov,
                        gpuData.d_observations_neighbors_copy_array, d_ldda_neighbors,
                        batchCount, queue);
    // gemm
    MagmaOps<Real>::gemm_max(MagmaTrans, MagmaNoTrans,
                             d_lda_locs, d_lda_locs, d_lda_locs_neighbors,
                             (Real)1.0, gpuData.d_cross_cov_array, d_ldda_cross_cov,
                             gpuData.d_cross_cov_array, d_ldda_cross_cov,
                             (Real)0.0, gpuData.d_cov_correction_array, d_ldda_cov,
                             batchCount,
                             max_n1, max_n1, max_m,
                             queue);
    MagmaOps<Real>::gemm_max(MagmaTrans, MagmaNoTrans,
                             d_lda_locs, d_const1, d_lda_locs_neighbors,
                             (Real)1.0, gpuData.d_cross_cov_array, d_ldda_cross_cov,
                             gpuData.d_observations_neighbors_copy_array, d_ldda_neighbors,
                             (Real)0.0, gpuData.d_mu_correction_array, d_ldda_locs,
                             batchCount,
                             max_n1, max_n2, max_m,
                             queue);

    // 2.2 compute the conditional mean and variance using batched kernels
    batched_matrix_add<Real>(
        gpuData.d_cov_array, gpuData.d_ldda_cov,
        gpuData.d_cov_correction_array, gpuData.d_lda_locs, gpuData.d_ldda_locs,
        -1.0, batchCount, stream);
    batched_vector_add<Real>(
        gpuData.d_observations_copy_array, gpuData.d_ldda_locs,
        gpuData.d_mu_correction_array, gpuData.d_lda_locs, gpuData.d_ldda_locs,
        -1.0, batchCount, stream); 
    checkCudaError(cudaStreamSynchronize(stream));

    // 2.3 compute the log-likelihood
    MagmaOps<Real>::potrf_final(MagmaLower, d_lda_locs, gpuData.d_cov_array, d_ldda_cov, dinfo_magma, batchCount, queue);
    MagmaOps<Real>::trsm_final(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit,
        d_lda_locs, d_const1, (Real)1.0,
        gpuData.d_cov_array, d_ldda_cov,
        gpuData.d_observations_copy_array, d_ldda_locs,
        batchCount, queue);

        // norm for all blocks
        double norm2_item = (double)norm2_batch<Real>(d_lda_locs, gpuData.d_observations_copy_array, d_ldda_locs, batchCount, stream);
        // determinant for all blocks
        double log_det_item = (double)log_det_batch<Real>(d_lda_locs, gpuData.d_cov_array, d_ldda_cov, batchCount, stream);
        double log_likelihood = -0.5 * (log_det_item + norm2_item);
        double log_likelihood_all = 0;
        MPI_Allreduce(&log_likelihood, &log_likelihood_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return log_likelihood_all;
    } else {
        // Real == float with optional mixed-precision routing
        bool cov64 = opts.mp_cov_double || opts.mp_all_double_ops;
        bool schur64 = opts.mp_schur_double || opts.mp_all_double_ops;
        bool final64 = opts.mp_final_double || opts.mp_all_double_ops;

        // prepare range in double if any double stage enabled
        if (cov64 || schur64 || final64) {
            std::vector<double> range_host64(opts.dim);
            for (int i=0;i<opts.dim;++i) range_host64[i] = theta[range_offset + i];
            checkCudaError(cudaMemcpy(gpuData.d_range_device_f64, range_host64.data(), opts.dim * sizeof(double), cudaMemcpyHostToDevice));
        }

        // covariance generation
        if (cov64) {
            compute_covariance_vbatched<double>(gpuData.d_locs_array_f64, gpuData.d_lda_locs, 1, gpuData.total_locs_num_device,
                gpuData.d_locs_array_f64, gpuData.d_lda_locs, 1, gpuData.total_locs_num_device,
                gpuData.d_cov_array_f64, gpuData.d_ldda_cov, gpuData.d_lda_locs, batchCount, opts.dim, theta, gpuData.d_range_device_f64, true, stream, opts);
            compute_covariance_vbatched<double>(gpuData.d_locs_neighbors_array_f64, gpuData.d_lda_locs_neighbors, 1, gpuData.total_locs_neighbors_num_device,
                gpuData.d_locs_array_f64, gpuData.d_lda_locs, 1, gpuData.total_locs_num_device,
                gpuData.d_cross_cov_array_f64, gpuData.d_ldda_cross_cov, gpuData.d_lda_locs, batchCount, opts.dim, theta, gpuData.d_range_device_f64, false, stream, opts);
            compute_covariance_vbatched<double>(gpuData.d_locs_neighbors_array_f64, gpuData.d_lda_locs_neighbors, 1, gpuData.total_locs_neighbors_num_device,
                gpuData.d_locs_neighbors_array_f64, gpuData.d_lda_locs_neighbors, 1, gpuData.total_locs_neighbors_num_device,
                gpuData.d_conditioning_cov_array_f64, gpuData.d_ldda_conditioning_cov, gpuData.d_lda_locs_neighbors, batchCount, opts.dim, theta, gpuData.d_range_device_f64, true, stream, opts);
        } else {
            compute_covariance_vbatched<float>(gpuData.d_locs_array, gpuData.d_lda_locs, 1, gpuData.total_locs_num_device,
                gpuData.d_locs_array, gpuData.d_lda_locs, 1, gpuData.total_locs_num_device,
                gpuData.d_cov_array, gpuData.d_ldda_cov, gpuData.d_lda_locs, batchCount, opts.dim, theta, gpuData.d_range_device, true, stream, opts);
            compute_covariance_vbatched<float>(gpuData.d_locs_neighbors_array, gpuData.d_lda_locs_neighbors, 1, gpuData.total_locs_neighbors_num_device,
                gpuData.d_locs_array, gpuData.d_lda_locs, 1, gpuData.total_locs_num_device,
                gpuData.d_cross_cov_array, gpuData.d_ldda_cross_cov, gpuData.d_lda_locs, batchCount, opts.dim, theta, gpuData.d_range_device, false, stream, opts);
            compute_covariance_vbatched<float>(gpuData.d_locs_neighbors_array, gpuData.d_lda_locs_neighbors, 1, gpuData.total_locs_neighbors_num_device,
                gpuData.d_locs_neighbors_array, gpuData.d_lda_locs_neighbors, 1, gpuData.total_locs_neighbors_num_device,
                gpuData.d_conditioning_cov_array, gpuData.d_ldda_conditioning_cov, gpuData.d_lda_locs_neighbors, batchCount, opts.dim, theta, gpuData.d_range_device, true, stream, opts);
        }

        // observation copies
        if (schur64 || final64) {
            convert_array<double, float>(gpuData.d_observations_neighbors_device, gpuData.d_observations_neighbors_copy_device_f64, (size_t)(gpuData.total_observations_neighbors_size/sizeof(float)), stream);
            convert_array<double, float>(gpuData.d_observations_device, gpuData.d_observations_copy_device_f64, (size_t)(gpuData.total_observations_points_size/sizeof(float)), stream);
        } else {
            checkCudaError(cudaMemcpy(gpuData.d_observations_neighbors_copy_device, gpuData.d_observations_neighbors_device, gpuData.total_observations_neighbors_size, cudaMemcpyDeviceToDevice));
            checkCudaError(cudaMemcpy(gpuData.d_observations_copy_device, gpuData.d_observations_device, gpuData.total_observations_points_size, cudaMemcpyDeviceToDevice));
        }

        // Schur complement
        if (schur64) {
            if (!cov64) {
                convert_array<double, float>(gpuData.d_conditioning_cov_device, gpuData.d_conditioning_cov_device_f64, (size_t)(gpuData.total_conditioning_cov_size_bytes/sizeof(float)), stream);
                convert_array<double, float>(gpuData.d_cross_cov_device, gpuData.d_cross_cov_device_f64, (size_t)(gpuData.total_cross_cov_size_bytes/sizeof(float)), stream);
            }
            MagmaOps<double>::potrf_neighbors(MagmaLower, d_lda_locs_neighbors, gpuData.d_conditioning_cov_array_f64, d_ldda_conditioning_cov, dinfo_magma, batchCount, max_m, queue);
            MagmaOps<double>::trsm_max(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit, max_m, max_n1, d_lda_locs_neighbors, d_lda_locs, 1.0, gpuData.d_conditioning_cov_array_f64, d_ldda_conditioning_cov, gpuData.d_cross_cov_array_f64, d_ldda_cross_cov, batchCount, queue);
            MagmaOps<double>::trsm_max(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit, max_m, max_n2, d_lda_locs_neighbors, d_const1, 1.0, gpuData.d_conditioning_cov_array_f64, d_ldda_conditioning_cov, gpuData.d_observations_neighbors_copy_array_f64, d_ldda_neighbors, batchCount, queue);
            MagmaOps<double>::gemm_max(MagmaTrans, MagmaNoTrans, d_lda_locs, d_lda_locs, d_lda_locs_neighbors, 1.0, gpuData.d_cross_cov_array_f64, d_ldda_cross_cov, gpuData.d_cross_cov_array_f64, d_ldda_cross_cov, 0.0, gpuData.d_cov_correction_array_f64, d_ldda_cov, batchCount, max_n1, max_n1, max_m, queue);
            MagmaOps<double>::gemm_max(MagmaTrans, MagmaNoTrans, d_lda_locs, d_const1, d_lda_locs_neighbors, 1.0, gpuData.d_cross_cov_array_f64, d_ldda_cross_cov, gpuData.d_observations_neighbors_copy_array_f64, d_ldda_neighbors, 0.0, gpuData.d_mu_correction_array_f64, d_ldda_locs, batchCount, max_n1, max_n2, max_m, queue);
        } else {
            MagmaOps<float>::potrf_neighbors(MagmaLower, d_lda_locs_neighbors, gpuData.d_conditioning_cov_array, d_ldda_conditioning_cov, dinfo_magma, batchCount, max_m, queue);
            MagmaOps<float>::trsm_max(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit, max_m, max_n1, d_lda_locs_neighbors, d_lda_locs, 1.0f, gpuData.d_conditioning_cov_array, d_ldda_conditioning_cov, gpuData.d_cross_cov_array, d_ldda_cross_cov, batchCount, queue);
            MagmaOps<float>::trsm_max(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit, max_m, max_n2, d_lda_locs_neighbors, d_const1, 1.0f, gpuData.d_conditioning_cov_array, d_ldda_conditioning_cov, gpuData.d_observations_neighbors_copy_array, d_ldda_neighbors, batchCount, queue);
            MagmaOps<float>::gemm_max(MagmaTrans, MagmaNoTrans, d_lda_locs, d_lda_locs, d_lda_locs_neighbors, 1.0f, gpuData.d_cross_cov_array, d_ldda_cross_cov, gpuData.d_cross_cov_array, d_ldda_cross_cov, 0.0f, gpuData.d_cov_correction_array, d_ldda_cov, batchCount, max_n1, max_n1, max_m, queue);
            MagmaOps<float>::gemm_max(MagmaTrans, MagmaNoTrans, d_lda_locs, d_const1, d_lda_locs_neighbors, 1.0f, gpuData.d_cross_cov_array, d_ldda_cross_cov, gpuData.d_observations_neighbors_copy_array, d_ldda_neighbors, 0.0f, gpuData.d_mu_correction_array, d_ldda_locs, batchCount, max_n1, max_n2, max_m, queue);
        }

        // mean/variance combine and final
        if (final64) {
            if (!cov64) { convert_array<double, float>(gpuData.d_cov_device, gpuData.d_cov_device_f64, (size_t)(gpuData.total_cov_size_bytes/sizeof(float)), stream); }
            if (!schur64) {
                convert_array<double, float>(gpuData.d_cov_correction_device, gpuData.d_cov_correction_device_f64, (size_t)(gpuData.total_cov_size_bytes/sizeof(float)), stream);
                convert_array<double, float>(gpuData.d_mu_correction_device, gpuData.d_mu_correction_device_f64, (size_t)(gpuData.total_observations_points_size/sizeof(float)), stream);
            }
            batched_matrix_add<double>(gpuData.d_cov_array_f64, gpuData.d_ldda_cov, gpuData.d_cov_correction_array_f64, gpuData.d_lda_locs, gpuData.d_ldda_locs, -1.0, batchCount, stream);
            batched_vector_add<double>(gpuData.d_observations_copy_array_f64, gpuData.d_ldda_locs, gpuData.d_mu_correction_array_f64, gpuData.d_lda_locs, gpuData.d_ldda_locs, -1.0, batchCount, stream);
            checkCudaError(cudaStreamSynchronize(stream));
            MagmaOps<double>::potrf_final(MagmaLower, d_lda_locs, gpuData.d_cov_array_f64, d_ldda_cov, dinfo_magma, batchCount, queue);
            MagmaOps<double>::trsm_final(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit, d_lda_locs, d_const1, 1.0, gpuData.d_cov_array_f64, d_ldda_cov, gpuData.d_observations_copy_array_f64, d_ldda_locs, batchCount, queue);
            double norm2_item = (double)norm2_batch<double>(d_lda_locs, const_cast<const double**>(gpuData.d_observations_copy_array_f64), d_ldda_locs, batchCount, stream);
            double log_det_item = (double)log_det_batch<double>(d_lda_locs, const_cast<const double**>(gpuData.d_cov_array_f64), d_ldda_cov, batchCount, stream);
            double log_likelihood = -0.5 * (log_det_item + norm2_item);
            double log_likelihood_all = 0; MPI_Allreduce(&log_likelihood, &log_likelihood_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); return log_likelihood_all;
        } else {
            if (schur64) {
                convert_array<float, double>(gpuData.d_cov_correction_device_f64, gpuData.d_cov_correction_device, (size_t)(gpuData.total_cov_size_bytes/sizeof(float)), stream);
                convert_array<float, double>(gpuData.d_mu_correction_device_f64, gpuData.d_mu_correction_device, (size_t)(gpuData.total_observations_points_size/sizeof(float)), stream);
                if (cov64) { convert_array<float, double>(gpuData.d_cov_device_f64, gpuData.d_cov_device, (size_t)(gpuData.total_cov_size_bytes/sizeof(float)), stream); }
                convert_array<float, double>(gpuData.d_observations_copy_device_f64, gpuData.d_observations_copy_device, (size_t)(gpuData.total_observations_points_size/sizeof(float)), stream);
            }
            batched_matrix_add<float>(gpuData.d_cov_array, gpuData.d_ldda_cov, gpuData.d_cov_correction_array, gpuData.d_lda_locs, gpuData.d_ldda_locs, -1.0f, batchCount, stream);
            batched_vector_add<float>(gpuData.d_observations_copy_array, gpuData.d_ldda_locs, gpuData.d_mu_correction_array, gpuData.d_lda_locs, gpuData.d_ldda_locs, -1.0f, batchCount, stream);
            checkCudaError(cudaStreamSynchronize(stream));
            MagmaOps<float>::potrf_final(MagmaLower, d_lda_locs, gpuData.d_cov_array, d_ldda_cov, dinfo_magma, batchCount, queue);
            MagmaOps<float>::trsm_final(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit, d_lda_locs, d_const1, 1.0f, gpuData.d_cov_array, d_ldda_cov, gpuData.d_observations_copy_array, d_ldda_locs, batchCount, queue);
            double norm2_item = (double)norm2_batch<float>(d_lda_locs, const_cast<const float**>(gpuData.d_observations_copy_array), d_ldda_locs, batchCount, stream);
            double log_det_item = (double)log_det_batch<float>(d_lda_locs, const_cast<const float**>(gpuData.d_cov_array), d_ldda_cov, batchCount, stream);
            double log_likelihood = -0.5 * (log_det_item + norm2_item);
            double log_likelihood_all = 0; MPI_Allreduce(&log_likelihood, &log_likelihood_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); return log_likelihood_all;
        }
    }
}

// Function to clean up GPU memory
template <typename Real>
void cleanupGpuMemory(GpuDataT<Real> &gpuData)
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

    // Optional double-precision mirrors cleanup
    cudaFree(gpuData.d_locs_device_f64);
    cudaFree(gpuData.d_locs_neighbors_device_f64);
    cudaFree(gpuData.d_cov_device_f64);
    cudaFree(gpuData.d_cross_cov_device_f64);
    cudaFree(gpuData.d_conditioning_cov_device_f64);
    cudaFree(gpuData.d_observations_neighbors_copy_device_f64);
    cudaFree(gpuData.d_observations_copy_device_f64);
    cudaFree(gpuData.d_mu_correction_device_f64);
    cudaFree(gpuData.d_cov_correction_device_f64);
    cudaFree(gpuData.d_range_device_f64);
    delete[] gpuData.h_locs_array_f64;
    delete[] gpuData.h_locs_neighbors_array_f64;
    delete[] gpuData.h_cov_array_f64;
    delete[] gpuData.h_cross_cov_array_f64;
    delete[] gpuData.h_conditioning_cov_array_f64;
    delete[] gpuData.h_observations_neighbors_copy_array_f64;
    delete[] gpuData.h_observations_copy_array_f64;
    delete[] gpuData.h_mu_correction_array_f64;
    delete[] gpuData.h_cov_correction_array_f64;
    cudaFree(gpuData.d_locs_array_f64);
    cudaFree(gpuData.d_locs_neighbors_array_f64);
    cudaFree(gpuData.d_cov_array_f64);
    cudaFree(gpuData.d_cross_cov_array_f64);
    cudaFree(gpuData.d_conditioning_cov_array_f64);
    cudaFree(gpuData.d_observations_neighbors_copy_array_f64);
    cudaFree(gpuData.d_observations_copy_array_f64);
    cudaFree(gpuData.d_mu_correction_array_f64);
    cudaFree(gpuData.d_cov_correction_array_f64);
}

// calculate the total flops
template <typename Real>
double gflopsTotal(const GpuDataT<Real> &gpuData, const Opts &opts)
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

// Explicit instantiations for float and double
template GpuDataT<double> copyDataToGPU<double>(const Opts&, const std::vector<BlockInfo>&);
template GpuDataT<float> copyDataToGPU<float>(const Opts&, const std::vector<BlockInfo>&);
template double performComputationOnGPU<double>(const GpuDataT<double>&, const std::vector<double>&, Opts&);
template double performComputationOnGPU<float>(const GpuDataT<float>&, const std::vector<double>&, Opts&);
template void cleanupGpuMemory<double>(GpuDataT<double>&);
template void cleanupGpuMemory<float>(GpuDataT<float>&);
template double gflopsTotal<double>(const GpuDataT<double>&, const Opts&);
template double gflopsTotal<float>(const GpuDataT<float>&, const Opts&);