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
#include <fstream>
#include <type_traits>

#include "gpu_operations.h"
#include "gpu_covariance.h"
#include "prediction.h"
#include "error_checking.h"
#include "magma_dispatch.h"

// Templated function to perform prediction on the GPU
template <typename Real>
std::tuple<double, double, double> performPredictionOnGPU(const GpuDataT<Real> &gpuData, const std::vector<double> &theta, const Opts &opts)
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
        checkCudaError(cudaMemcpy(gpuData.d_range_device, 
                                   range_host.data(), 
                                   opts.dim * sizeof(Real), 
                                   cudaMemcpyHostToDevice));
    }

    // Use the data on the GPU for computation
    // 1. generate the covariance matrix, cross covariance matrix, conditioning covariance matrix
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
    // Synchronize to make sure the kernel has finished
    checkCudaError(cudaStreamSynchronize(stream));
    
    // 2. perform the computation
    // 2.1 compute the correction term for mean and variance (i.e., Schur complement)
    MagmaOps<Real>::potrf_neighbors(MagmaLower, d_lda_locs_neighbors,
                        gpuData.d_conditioning_cov_array, d_ldda_conditioning_cov,
                        dinfo_magma, batchCount, max_m, queue);
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
    // Fused updates: cov = cov - cross^T*cross; y = y - cross^T*y_nn
    MagmaOps<Real>::gemm_max(MagmaTrans, MagmaNoTrans,
                             d_lda_locs, d_lda_locs, d_lda_locs_neighbors,
                             (Real)(-1.0), gpuData.d_cross_cov_array, d_ldda_cross_cov,
                             gpuData.d_cross_cov_array, d_ldda_cross_cov,
                             (Real)1.0, gpuData.d_cov_array, d_ldda_cov,
                             batchCount, 
                             max_n1, max_n1, max_m, 
                             queue);
    MagmaOps<Real>::gemm_max(MagmaTrans, MagmaNoTrans,
                             d_lda_locs, d_const1, d_lda_locs_neighbors,
                             (Real)(-1.0), gpuData.d_cross_cov_array, d_ldda_cross_cov,
                             gpuData.d_observations_neighbors_copy_array, d_ldda_neighbors,
                             (Real)1.0, gpuData.d_observations_copy_array, d_ldda_locs,
                             batchCount, 
                             max_n1, max_n2, max_m,
                             queue);
    checkCudaError(cudaStreamSynchronize(stream));
    // 2.2 compute the conditional mean and variance
    // Fused already; no separate corrections needed
    checkCudaError(cudaStreamSynchronize(stream));

    // New code starts here
    // 3. Copy mean and variance from GPU to CPU
    std::vector<Real> h_means(gpuData.numPointsPerProcess);
    std::vector<Real> h_variances(gpuData.numPointsPerProcess);
    std::vector<Real> true_observations(gpuData.numPointsPerProcess);
    // Copy true observations, accounting for padding
    int offset = 0;
    for (size_t i = 0; i < batchCount; ++i) {
        checkCudaError(cudaMemcpy(true_observations.data() + offset, 
                                  gpuData.h_observations_array[i], 
                                  gpuData.lda_locs[i] * sizeof(Real), 
                                  cudaMemcpyDeviceToHost));
        checkCudaError(cudaMemcpy(h_means.data() + offset, 
                                  gpuData.h_observations_copy_array[i], 
                                  gpuData.lda_locs[i] * sizeof(Real), 
                                  cudaMemcpyDeviceToHost));
        //copy the diagonal of the covariance matrix
        for (size_t j = 0; j < gpuData.lda_locs[i]; ++j){
            Real tmp;
            checkCudaError(cudaMemcpy(&tmp, 
                                      &gpuData.h_cov_array[i][j * gpuData.ldda_cov[i] + j], 
                                      sizeof(Real), 
                                      cudaMemcpyDeviceToHost));
            h_variances[offset + j] = tmp;
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
    
    // 6. Calculate MSPE, RMSPE and confidence interval coverage
    
    double local_mspe_sum = 0.0;
    double local_rmspe_sum = 0.0;
    int local_within_ci = 0;
    
    for (int i = 0; i < gpuData.numPointsPerProcess; ++i) {
        local_mspe_sum += std::pow(sample_means[i] - true_observations[i], 2);
        // Calculate percentage error for RMSPE
        if (std::abs(true_observations[i]) > 1e-5) {  // Avoid division by zero
            double rmspe = std::pow(100 * (sample_means[i] - true_observations[i]) / true_observations[i], 2);
            local_rmspe_sum += rmspe;
        }

        // Save prediction results to CSV file
        double ci_lower = sample_means[i] - 1.96 * std::sqrt(sample_variances[i]);
        double ci_upper = sample_means[i] + 1.96 * std::sqrt(sample_variances[i]);
        
        if (true_observations[i] >= ci_lower && true_observations[i] <= ci_upper) {
            local_within_ci++;
        }
    }
    
    // MPI Allreduce to sum up mspe, rmspe, within_ci, and point counts across all processes
    double global_mspe_sum = 0.0;
    double global_rmspe_sum = 0.0;
    int global_within_ci = 0;
    MPI_Allreduce(&local_mspe_sum, &global_mspe_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_rmspe_sum, &global_rmspe_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_within_ci, &global_within_ci, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Calculate final MSPE, RMSPE and CI coverage
    double mspe = std::round(global_mspe_sum / opts.numPointsTotal_test * 1e16) / 1e16;
    double rmspe = std::sqrt(global_rmspe_sum / opts.numPointsTotal_test); 
    double ci_coverage = static_cast<double>(global_within_ci) / opts.numPointsTotal_test;

    // Print results
    if (rank == 0) {
        std::cout << "MSPE: " << mspe << std::endl;
        std::cout << "RMSPE: " << rmspe << "%" << std::endl;
        std::cout << "95% CI coverage: " << ci_coverage * 100 << "%" << std::endl;
        std::cout << "-------------------Prediction Done-----------------" << std::endl;
    }
    return std::make_tuple(mspe, rmspe, ci_coverage);
}

// Explicit instantiations
template std::tuple<double,double,double> performPredictionOnGPU<double>(const GpuDataT<double>&, const std::vector<double>&, const Opts&);
template std::tuple<double,double,double> performPredictionOnGPU<float>(const GpuDataT<float>&, const std::vector<double>&, const Opts&);
