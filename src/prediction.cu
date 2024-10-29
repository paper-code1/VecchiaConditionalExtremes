#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <magma_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <mpi.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <random>
#include <numeric>

#include "gpu_operations.h"
#include "gpu_covariance.h"
#include "prediction.h"
#include "error_checking.h"

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
                    opts.dim, theta, true, stream);
        RBF_matcov(gpuData.h_locs_neighbors_array[i], 
                    gpuData.lda_locs_neighbors[i], 1, gpuData.total_locs_neighbors_num_device,
                    gpuData.h_locs_array[i],
                    gpuData.lda_locs[i], 1, gpuData.total_locs_num_device,
                    gpuData.h_cross_cov_array[i], gpuData.ldda_cross_cov[i], gpuData.lda_locs[i],
                    opts.dim, theta, false, stream);
        RBF_matcov(gpuData.h_locs_neighbors_array[i],
                    gpuData.lda_locs_neighbors[i], 1, gpuData.total_locs_neighbors_num_device,
                    gpuData.h_locs_neighbors_array[i], 
                    gpuData.lda_locs_neighbors[i], 1, gpuData.total_locs_neighbors_num_device,
                    gpuData.h_conditioning_cov_array[i], gpuData.ldda_conditioning_cov[i], gpuData.lda_locs_neighbors[i],
                    opts.dim, theta, true, stream);
        // Synchronize to make sure the kernel has finished
        checkCudaError(cudaStreamSynchronize(stream));
        // if (i == 160){
            // magma_dprint_gpu(gpuData.lda_locs_neighbors[i], gpuData.lda_locs_neighbors[i], gpuData.h_conditioning_cov_array[i], gpuData.ldda_conditioning_cov[i], queue);
            // magma_dprint_gpu(gpuData.lda_locs_neighbors[i],gpuData.lda_locs[i], gpuData.h_cross_cov_array[i], gpuData.ldda_cross_cov[i], queue);
            // print coordinate 
            // magma_dprint_gpu(gpuData.lda_locs[i], 1, gpuData.h_locs_array[i], gpuData.ldda_cov[i], queue);
            // magma_dprint_gpu(gpuData.lda_locs[i], 1, gpuData.h_locs_array[i] + gpuData.total_locs_num_device, gpuData.ldda_cov[i], queue);
        // }
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
    // magma_dprint_gpu(gpuData.lda_locs_neighbors[160], 1, gpuData.h_observations_neighbors_copy_array[160], gpuData.ldda_conditioning_cov[160], queue);
    magmablas_dtrsm_vbatched_max_nocheck(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit, 
                        max_m, max_n2, 
                        d_lda_locs_neighbors, d_const1,
                        1.,
                        gpuData.d_conditioning_cov_array, d_ldda_conditioning_cov,
                        gpuData.d_observations_neighbors_copy_array, d_ldda_neighbors,
                        batchCount, queue);
    // magma_dprint_gpu(gpuData.lda_locs_neighbors[160], 1, gpuData.h_observations_neighbors_copy_array[160], gpuData.ldda_conditioning_cov[160], queue);
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
        // copy h_mu_correction_array to h_observations_copy_array
        checkCudaError(cudaMemcpy(gpuData.h_observations_copy_array[i], 
                                  gpuData.h_mu_correction_array[i], 
                                  gpuData.lda_locs[i] * sizeof(double), 
                                  cudaMemcpyDeviceToHost));
    }
    checkCudaError(cudaStreamSynchronize(stream));

    // New code starts here
    // 3. Copy mean and variance from GPU to CPU
    std::vector<double> h_means(gpuData.numPointsPerProcess);
    std::vector<double> h_variances(gpuData.numPointsPerProcess);
    std::vector<double> true_observations(gpuData.numPointsPerProcess);
    // Copy true observations, accounting for padding
    int offset = 0;
    for (size_t i = 0; i < batchCount; ++i) {
        checkCudaError(cudaMemcpy(true_observations.data() + offset, 
                                  gpuData.h_observations_array[i], 
                                  gpuData.lda_locs[i] * sizeof(double), 
                                  cudaMemcpyDeviceToHost));
        checkCudaError(cudaMemcpy(h_means.data() + offset, 
                                  gpuData.h_observations_copy_array[i], 
                                  gpuData.lda_locs[i] * sizeof(double), 
                                  cudaMemcpyDeviceToHost));
        //copy the diagonal of the covariance matrix
        for (size_t j = 0; j < gpuData.lda_locs[i]; ++j){
            checkCudaError(cudaMemcpy(h_variances.data() + offset + j, 
                                      &gpuData.h_cov_array[i][j * gpuData.ldda_cov[i] + j], 
                                      sizeof(double), 
                                      cudaMemcpyDeviceToHost));
        }
        offset += gpuData.lda_locs[i];
    }

    // 4. Perform sampling
    std::mt19937 gen(rank);
    std::vector<std::vector<double>> samples(gpuData.numPointsPerProcess, 
                                             std::vector<double>(opts.num_simulations));
    for (int i = 0; i < gpuData.numPointsPerProcess; ++i) {
        std::normal_distribution<double> d(h_means[i], std::sqrt(h_variances[i]));
        for (int j = 0; j < opts.num_simulations; ++j) {
            samples[i][j] = d(gen);
        }
    }

    // 5. Calculate sample mean and sample variance
    std::vector<double> sample_means(gpuData.numPointsPerProcess);
    std::vector<double> sample_variances(gpuData.numPointsPerProcess);
    
    for (int i = 0; i < gpuData.numPointsPerProcess; ++i) {
        double sum = std::accumulate(samples[i].begin(), samples[i].end(), 0.0);
        sample_means[i] = sum / opts.num_simulations;
        
        double sq_sum = std::inner_product(samples[i].begin(), samples[i].end(), samples[i].begin(), 0.0);
        sample_variances[i] = sq_sum / opts.num_simulations - sample_means[i] * sample_means[i];
    }

    // 6. Calculate MSPE and confidence interval coverage
    
    double local_mspe_sum = 0.0;
    int local_within_ci = 0;
    
    for (int i = 0; i < gpuData.numPointsPerProcess; ++i) {
        local_mspe_sum += std::pow(sample_means[i] - true_observations[i], 2);
        
        // double ci_lower = sample_means[i] - 1.96 * std::sqrt(sample_variances[i]);
        // double ci_upper = sample_means[i] + 1.96 * std::sqrt(sample_variances[i]);
        double ci_lower = h_means[i] - 1.96 * std::sqrt(h_variances[i]);
        double ci_upper = h_means[i] + 1.96 * std::sqrt(h_variances[i]);
        
        if (true_observations[i] >= ci_lower && true_observations[i] <= ci_upper) {
            local_within_ci++;
        }
        // std::cout << "true_observations["<< i <<"]: " << true_observations[i] << ", predicted mean: " << h_means[i] << ", predicted variance: " << h_variances[i] << ", sample_means["<< i <<"]: " << sample_means[i] << ", sample_variances["<< i <<"]: " << sample_variances[i] << ", ci_lower: " << ci_lower << ", ci_upper: " << ci_upper << std::endl;
    }
    
    // MPI Allreduce to sum up mspe, within_ci, and point counts across all processes
    double global_mspe_sum = 0.0;
    int global_within_ci = 0;
    MPI_Allreduce(&local_mspe_sum, &global_mspe_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_within_ci, &global_within_ci, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Calculate final MSPE and CI coverage
    double mspe = global_mspe_sum / opts.numPointsTotal_test;
    double ci_coverage = static_cast<double>(global_within_ci) / opts.numPointsTotal_test;

    // Print results
    if (rank == 0) {
        std::cout << "MSPE: " << mspe << std::endl;
        std::cout << "95% CI coverage: " << ci_coverage * 100 << "%" << std::endl;
        std::cout << "-------------------Prediction Done-----------------" << std::endl;
    }
}
