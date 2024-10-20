#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <magma_v2.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <mpi.h>
#include <vector>
#include <iostream>
#include <cmath>

#include "gpu_operations.h"
#include "gpu_covariance.h"
#include "prediction.h"
#include "error_checking.h"

// Custom functor for strided access
struct strided_access
{
    double* ptr;
    int stride;
    strided_access(double* _ptr, int _stride) : ptr(_ptr), stride(_stride) {}
    __host__ __device__
    double operator()(int i) const { return ptr[i * stride]; }
};


// Function to perform prediction on the GPU
void performPredictionOnGPU(const GpuData &gpuData, const std::vector<double> &theta, const Opts &opts)
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

    // copy the data from the device to the device
    checkCudaError(cudaMemcpy(gpuData.d_observations_neighbors_copy_device, 
                                   gpuData.d_observations_neighbors_device, 
                                   gpuData.total_observations_neighbors_size, 
                                   cudaMemcpyDeviceToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_observations_copy_device, 
                                   gpuData.d_observations_device, 
                                   gpuData.total_observations_points_size, 
                                   cudaMemcpyDeviceToDevice));

    // Use the data on the GPU for computation
    // 1. generate the covariance matrix, cross covariance matrix, conditioning covariance matrix
    // take record of the time
    for (size_t i = 0; i < batchCount; ++i)
    {   
        RBF_matcov(gpuData.h_locs_array[i],
                    gpuData.lda_locs[i], 1, gpuData.total_locs_num_device,
                    gpuData.h_locs_array[i],
                    gpuData.lda_locs[i], 1, gpuData.total_locs_num_device,
                    gpuData.h_cov_array[i], gpuData.ldda_cov[i], gpuData.lda_locs[i],
                    opts.dim, theta, stream);
        RBF_matcov(gpuData.h_locs_neighbors_array[i], 
                    gpuData.lda_locs_neighbors[i], 1, gpuData.total_locs_neighbors_num_device,
                    gpuData.h_locs_array[i],
                    gpuData.lda_locs[i], 1, gpuData.total_locs_num_device,
                    gpuData.h_cross_cov_array[i], gpuData.ldda_cross_cov[i], gpuData.lda_locs[i],
                    opts.dim, theta, stream);
        RBF_matcov(gpuData.h_locs_neighbors_array[i],
                    gpuData.lda_locs_neighbors[i], 1, gpuData.total_locs_neighbors_num_device,
                    gpuData.h_locs_neighbors_array[i], 
                    gpuData.lda_locs_neighbors[i], 1, gpuData.total_locs_neighbors_num_device,
                    gpuData.h_conditioning_cov_array[i], gpuData.ldda_conditioning_cov[i], gpuData.lda_locs_neighbors[i],
                    opts.dim, theta, stream);
        // Synchronize to make sure the kernel has finished
        checkCudaError(cudaStreamSynchronize(stream));
    }    
    
    // 2. perform the computation
    // 2.1 compute the correction term for mean and variance (i.e., Schur complement)
    magma_dpotrf_vbatched(MagmaLower, d_lda_locs_neighbors,
                        gpuData.d_conditioning_cov_array, d_ldda_conditioning_cov,
                        dinfo_magma, batchCount, queue);
    // trsm
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
    checkCudaError(cudaStreamSynchronize(stream));
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
    }
    checkCudaError(cudaStreamSynchronize(stream));

    // 2.3 compute the log-likelihood
    // magma_dprint_gpu(gpuData.lda_locs[0], gpuData.lda_locs[0], gpuData.h_cov_array[0], gpuData.ldda_cov[0], queue);
    checkMagmaError(magma_dpotrf_vbatched(
            MagmaLower, d_lda_locs,
            gpuData.d_cov_array, d_ldda_cov,
            dinfo_magma, batchCount, queue));
    checkCudaError(cudaStreamSynchronize(stream));

    // do the conditional simulation
    // // allocate the memory for the conditional simulation
    int num_observations = gpuData.total_observations_points_size/sizeof(double);
    double *d_observations_conditional_device;
    double **h_observations_conditional_array = new double*[opts.numBlocksPerProcess * opts.num_simulations];
    double **d_observations_conditional_array;
    checkCudaError(cudaMalloc(&d_observations_conditional_device, 
        num_observations * sizeof(double) * opts.num_simulations));
    checkCudaError(cudaMalloc(&d_observations_conditional_array, 
        opts.num_simulations * opts.numBlocksPerProcess * sizeof(double *)));
    int offset_outer_data = 0;
    int offset_outer_array   = 0;
    for (size_t i = 0; i < opts.num_simulations; ++i){
        int offset_inner = 0;
        for (size_t j = 0; j < opts.numBlocksPerProcess; ++j){
            h_observations_conditional_array[j + offset_outer_array] = d_observations_conditional_device + offset_outer_data + offset_inner;
            offset_inner += gpuData.ldda_locs[j];
        }
        offset_outer_data += num_observations;
        offset_outer_array += opts.numBlocksPerProcess;
    }
    // copy h_observations_conditional_array to the device
    checkCudaError(cudaMemcpy(d_observations_conditional_array, h_observations_conditional_array, 
        opts.num_simulations * opts.numBlocksPerProcess * sizeof(double *), cudaMemcpyHostToDevice));
    // generate the random noise - h_mu_correction_array
    for (size_t i = 0; i < opts.num_simulations; ++i){
        // generate the random noise,
        // // gpuData.h_mu_correction_array[0] means we generate noise for all blocks
        generate_normal(gpuData.h_mu_correction_array[0], num_observations, 0, 1, rank * opts.num_simulations + i, stream);
        // cholesky factor gemm
        magmablas_dgemm_vbatched_max_nocheck(MagmaNoTrans, MagmaNoTrans,
                             d_lda_locs, d_const1, d_lda_locs,
                             1, gpuData.d_cov_array, d_ldda_cov,
                                gpuData.d_mu_correction_array, d_ldda_locs,
                             0, d_observations_conditional_array + i * opts.numBlocksPerProcess, d_ldda_locs,
                             batchCount, 
                             max_n1, max_n2, max_n1,
                             queue);
        // error term
        magma_daxpy(num_observations, -1.,
                    gpuData.d_observations_copy_device, 1,
                    d_observations_conditional_device + num_observations * i, 1,
                    queue);
    }
    checkCudaError(cudaStreamSynchronize(stream));

    // Calculate mean and standard deviation using Thrust
    thrust::device_vector<double> d_mean(num_observations);
    thrust::device_vector<double> d_stddev(num_observations);

    // Calculate mean
    for (int i = 0; i < num_observations; ++i) {
        strided_access sa(d_observations_conditional_device + i, num_observations);
        double sum = thrust::reduce(
            thrust::cuda::par.on(stream),
            thrust::make_transform_iterator(thrust::counting_iterator<int>(0), sa),
            thrust::make_transform_iterator(thrust::counting_iterator<int>(opts.num_simulations), sa),
            0.0, 
            thrust::plus<double>()
        );
        d_mean[i] = sum / opts.num_simulations;
    }

    // Calculate standard deviation
    for (int i = 0; i < num_observations; ++i) {
        strided_access sa(d_observations_conditional_device + i, num_observations);
        double mean = d_mean[i];
        double sum_sq = thrust::transform_reduce(
            thrust::cuda::par.on(stream),
            thrust::make_transform_iterator(thrust::counting_iterator<int>(0), sa),
            thrust::make_transform_iterator(thrust::counting_iterator<int>(opts.num_simulations), sa),
            [mean] __device__ (double x) { return (x - mean) * (x - mean); },
            0.0,
            thrust::plus<double>()
        );
        d_stddev[i] = sqrt(sum_sq / (opts.num_simulations - 1));
    }
    //print the mean and standard deviation
    if (rank == 0){
        for (int i = 0; i < num_observations; ++i){
            std::cout << "mean[" << i << "]: " << d_mean[i] << ", stddev[" << i << "]: " << d_stddev[i] << std::endl;
        }
    }

    // free d_observations_conditional_device
    cudaFree(d_observations_conditional_device);
    cudaFree(d_observations_conditional_array);
}

