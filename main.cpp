#include <mpi.h>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <chrono>
#include <thread>
#include <iomanip>
#include <fstream>
#include <magma_v2.h>
#include <omp.h>
#include "input_parser.h"
#include "block_info.h"
#include "random_points.h"
#include "distance_calc.h"
#include "gpu_covariance.h"
#include "gpu_operations.h"
#include "vecchia_helper.h"
#include "prediction.h"
#include "borehole.hpp"
#include <nlopt.hpp>
#include <nlopt.h>
struct OptimizationData {
    GpuData* gpuData;
    Opts* opts;
    int rank;
    MPI_Comm comm;
};

// Wrapper function for NLopt
double objective_function(const std::vector<double> &x, std::vector<double> &grad, void *data) {
    OptimizationData* opt_data = static_cast<OptimizationData*>(data);
    GpuData* gpuData = opt_data->gpuData;
    Opts* opts = opt_data->opts;
    
    // Broadcast theta values from rank 0 to all processes
    MPI_Bcast(const_cast<double*>(x.data()), x.size(), MPI_DOUBLE, 0, opt_data->comm);
    
    // Negate the log-likelihood because NLopt minimizes by default
    double result = -performComputationOnGPU(*gpuData, x, *opts);
    
    // Print optimization info
    if (opt_data->rank == 0 && opts->print) {
        std::cout << "Optimization step: " << opts->current_iter++ << ", ";
        std::cout << "f(theta): " << std::fixed << std::setprecision(6) << -result << ", ";
        std::cout << "Theta: ";
        for (const auto& val : x) {
            std::cout << std::fixed << std::setprecision(6) << val << " ";
        }
        std::cout << std::endl;
    }
    
    return result;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    magma_init();
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Opts opts;
    if (!parse_args(argc, argv, opts)) {
        if (rank == 0) {
            std::cerr << "Failed to parse command line arguments." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    omp_set_num_threads(opts.omp_num_threads);
    opts.distance_threshold_coarse = calculate_distance_threshold(opts.distance_scale, opts.numPointsTotal, opts.m, opts.nn_multiplier);
    opts.distance_threshold_finer = calculate_distance_threshold(opts.distance_scale, opts.numPointsTotal, opts.m, opts.nn_multiplier/10);

    // Use the parsed options
    if (rank == 0) {
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "Mode: " << opts.mode << std::endl;
        std::cout << "train_metadata_path: " << opts.train_metadata_path << std::endl;
        std::cout << "test_metadata_path: " << opts.test_metadata_path << std::endl;
        switch (opts.kernel_type) {
            case KernelType::PowerExponential:
                std::cout << "kernel_type: PowerExponential" << std::endl;
                break;
            case KernelType::Matern72:
                std::cout << "kernel_type: Matern72" << std::endl;
                break;
            case KernelType::Matern12:
                std::cout << "kernel_type: Matern12" << std::endl;
                break;
            case KernelType::Matern32:
                std::cout << "kernel_type: Matern32" << std::endl;
                break;
            case KernelType::Matern52:
                std::cout << "kernel_type: Matern52" << std::endl;
                break;
            default:
                std::cout << "kernel_type: Unsupported" << std::endl;
                exit(-1);
                break;
        }
        std::cout << "Number of total points: " << opts.numPointsTotal << std::endl;
        std::cout << "Number of total blocks: " << opts.numBlocksTotal << std::endl;
        std::cout << "Number of total points_test: " << opts.numPointsTotal_test << std::endl;
        std::cout << "Number of total blocks_test: " << opts.numBlocksTotal_test << std::endl;
        std::cout << "The number of nearest neighbors: " << opts.m << std::endl;
        std::cout << "The number of nearest neighbors_test: " << opts.m_test << std::endl;
        std::cout << "The distance threshold_coarse: " << opts.distance_threshold_coarse << std::endl;
        std::cout << "The distance threshold_finer: " << opts.distance_threshold_finer << std::endl;
        std::cout << "Dimension: " << opts.dim << std::endl;
        std::cout << "Range offset: " << opts.range_offset << std::endl;
        std::cout << "Distance scale: ";
        for (auto scale : opts.distance_scale) {
            std::cout << scale << ", ";
        }
        std::cout << std::endl;
        // print the varied size of theta
        std::cout << "Theta: ";
        for (auto theta : opts.theta_init) {
            std::cout << theta << ", ";
        }
        std::cout << std::endl;
        // print the lower bounds
        std::cout << "Lower bounds: ";
        for (auto bound : opts.lower_bounds) {
            std::cout << bound << ", ";
        }
        std::cout << std::endl;
        // print the upper bounds
        std::cout << "Upper bounds: ";
        for (auto bound : opts.upper_bounds) {
            std::cout << bound << ", ";
        }
        std::cout << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }

    // 1. Generate random points
    std::vector<PointMetadata> localPoints;
    std::vector<PointMetadata> localPoints_test;
    if (opts.train_metadata_path != ""){
        localPoints = readPointsConcurrently(opts.train_metadata_path, opts.numPointsTotal, opts);
        if (opts.mode == "prediction"){
            localPoints_test = readPointsConcurrently(opts.test_metadata_path, opts.numPointsTotal_test, opts);
        }
    }
    else{
        localPoints = generateRandomPoints(opts.numPointsPerProcess, opts);
        if (opts.mode == "prediction"){
            localPoints_test = generateRandomPoints(opts.numPointsPerProcess_test, opts);
        }
        // std::cout << "Sampling borehole function" << std::endl;
        // opts.dim = 8;
        // std::pair<std::vector<PointMetadata>, std::pair<double, double>> result = Borehole::sample_borehole(opts.numPointsPerProcess, rank, true);
        // localPoints = std::move(result.first);
        // double train_mean = result.second.first;
        // double train_variance = result.second.second;
        // std::cout << "localPoints.size(): " << localPoints.size() << std::endl;
        // std::cout << "rank: " << rank << ", size: " << size << std::endl;
        // if (opts.mode == "prediction"){
        //     std::pair<std::vector<PointMetadata>, std::pair<double, double>> result_test = Borehole::sample_borehole(opts.numPointsPerProcess_test, rank + size*42, false, train_mean, train_variance);
        //     localPoints_test = std::move(result_test.first);
        // }
        // std::cout << "Sampling done" << std::endl;
    }

    // do the distance scale for input points
    distanceScale(localPoints, opts.distance_scale, opts.dim);
    if (opts.mode == "prediction"){
        distanceScale(localPoints_test, opts.distance_scale, opts.dim);
    }
    
    auto start_total = std::chrono::high_resolution_clock::now();

    // 2.1 Partition points and communicate them
    if (rank == 0){
        std::cout << "Performing RAC partitioning" << std::endl;
    }
    auto start_preprocessing = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<PointMetadata>> finerPartitions;
    std::vector<std::vector<PointMetadata>> finerPartitions_test;
    partitionPointsDirectly(localPoints, finerPartitions, opts.numBlocksPerProcess, opts);
    if (opts.mode == "prediction"){
        partitionPointsDirectly(localPoints_test, finerPartitions_test, opts.numBlocksPerProcess_test, opts);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_preprocessing = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_preprocessing = end_preprocessing - start_preprocessing;
    
    // Find the maximum preprocessing duration across all processes
    double max_RAC_partitioning;
    double duration_preprocessing_seconds = duration_preprocessing.count();
    MPI_Allreduce(&duration_preprocessing_seconds, &max_RAC_partitioning, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // 2.3 Calculate centers of gravity for each block
    if (rank == 0){
        std::cout << "Calculating centers of gravity" << std::endl;
    }
    auto start_centers_of_gravity = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<double>> centers = calculateCentersOfGravity(finerPartitions, opts);
    std::vector<std::vector<double>> centers_test;
    if (opts.mode == "prediction"){
        centers_test = calculateCentersOfGravity(finerPartitions_test, opts);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_centers_of_gravity = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_centers_of_gravity = end_centers_of_gravity - start_centers_of_gravity;
    double max_duration_centers_of_gravity;
    double duration_centers_of_gravity_seconds = duration_centers_of_gravity.count();
    MPI_Allreduce(&duration_centers_of_gravity_seconds, &max_duration_centers_of_gravity, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // 2.4 Send centers of gravity to processor 0
    auto start_send_centers_of_gravity = std::chrono::high_resolution_clock::now();
    // first is the centers, second is the rank of the processor
    std::vector<std::pair<std::vector<double>, int>> allCenters;
    std::vector<std::pair<std::vector<double>, int>> allCenters_test;
    // send the centers to the root processor and add the rank (label)
    sendCentersOfGravityToRoot(centers, allCenters, opts);
    if (opts.mode == "prediction"){
        sendCentersOfGravityToRoot(centers_test, allCenters_test, opts);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_send_centers_of_gravity = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_send_centers_of_gravity = end_send_centers_of_gravity - start_send_centers_of_gravity;
    double max_duration_send_centers_of_gravity;
    double duration_send_centers_of_gravity_seconds = duration_send_centers_of_gravity.count();
    MPI_Allreduce(&duration_send_centers_of_gravity_seconds, &max_duration_send_centers_of_gravity, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // 3. Reorder centers at processor 0
    if (rank == 0){
        std::cout << "Reordering centers" << std::endl;
    }
    auto start_reorder_centers = std::chrono::high_resolution_clock::now();
    reorderCenters(allCenters, opts);
    // Broadcast reordered centers to all processors
    int numCenters = allCenters.size();
    int numCenters_test = allCenters_test.size();
    MPI_Bcast(&numCenters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    broadcastCenters(allCenters, numCenters, opts);
    if (opts.mode == "prediction"){
        MPI_Bcast(&numCenters_test, 1, MPI_INT, 0, MPI_COMM_WORLD);
        broadcastCenters(allCenters_test, numCenters_test, opts);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_reorder_centers = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_reorder_centers = end_reorder_centers - start_reorder_centers;
    double max_duration_reorder_centers;
    double duration_reorder_centers_seconds = duration_reorder_centers.count();
    MPI_Allreduce(&duration_reorder_centers_seconds, &max_duration_reorder_centers, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // 4. NN searching
    // 4.1 Create block information
    auto start_create_block_info = std::chrono::high_resolution_clock::now();
    std::vector<BlockInfo> localBlocks = createBlockInfo(finerPartitions, centers, allCenters, opts);
    std::vector<BlockInfo> localBlocks_test;
    if (opts.mode == "prediction"){
        // testset does not need to consider order of blocks
        localBlocks_test = createBlockInfo_test(finerPartitions_test, centers_test, opts);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_create_block_info = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_create_block_info = end_create_block_info - start_create_block_info;
    double max_duration_create_block_info;
    double duration_create_block_info_seconds = duration_create_block_info.count();
    MPI_Allreduce(&duration_create_block_info_seconds, &max_duration_create_block_info, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // 4.2 Block candidate preparation
    // Here, you could set the first 100 need to obtain all previous blocks and set threshold
    auto start_block_sending = std::chrono::high_resolution_clock::now();
    std::vector<BlockInfo> receivedBlocks = processAndSendBlocks(localBlocks, allCenters, allCenters_test, opts.distance_threshold_coarse, opts);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_block_sending = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_block_sending = end_block_sending - start_block_sending;
    double max_duration_block_sending;
    double duration_block_sending_seconds = duration_block_sending.count();
    MPI_Allreduce(&duration_block_sending_seconds, &max_duration_block_sending, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // print total point size of receivedBlocks and receivedBlocks size
    int total_point_size = 0;
    for (auto block : receivedBlocks){
        total_point_size += block.blocks.size();
    }
    // 4.3 NN searching
    if (rank == 0){
        std::cout << "Performing NN searching" << std::endl;
    }
    auto start_nn_searching = std::chrono::high_resolution_clock::now();
    nearest_neighbor_search(localBlocks, receivedBlocks, opts, false);
    if (opts.mode == "prediction"){
        nearest_neighbor_search(localBlocks_test, receivedBlocks, opts, true);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_nn_searching = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_nn_searching = end_nn_searching - start_nn_searching;
    double max_duration_nn_searching;
    double duration_nn_searching_seconds = duration_nn_searching.count();
    MPI_Allreduce(&duration_nn_searching_seconds, &max_duration_nn_searching, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // descale
    distanceDeScale(localBlocks, opts.distance_scale, opts.dim);
    if (opts.mode == "prediction"){
        distanceDeScale(localBlocks_test, opts.distance_scale, opts.dim);
    }
    // 5. independent computation of log-likelihood
    auto start_gpu_copy = std::chrono::high_resolution_clock::now();
    // Step 1: Copy data to GPU
    GpuData gpuData = copyDataToGPU(opts, localBlocks);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_gpu_copy = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_gpu_copy = end_gpu_copy - start_gpu_copy;
    double max_duration_gpu_copy;
    double duration_gpu_copy_seconds = duration_gpu_copy.count();
    MPI_Allreduce(&duration_gpu_copy_seconds, &max_duration_gpu_copy, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // // calculate the tota flops
    double total_gflops = gflopsTotal(gpuData, opts);

    // Step 2: Perform computation with BOBYQA optimization
    auto start_computation = std::chrono::high_resolution_clock::now();

    // Set up the optimization problem
    nlopt::opt optimizer(nlopt::LN_SBPLX, opts.theta_init.size());
    
    // Set bounds for theta_init (adjust these as needed)
    // set smoothness and nugget in power exponential kernel
    switch (opts.kernel_type) {
        case KernelType::PowerExponential:
            opts.lower_bounds[0] = 0.01; // sigma2
            opts.upper_bounds[0] = 3.0;
            opts.lower_bounds[1] = 0.01; // smoothness
            opts.upper_bounds[1] = 2.0;
            opts.lower_bounds[2] = 0.0; // nugget
            opts.upper_bounds[2] = 0.1;
            break;
        case KernelType::Matern72:
            opts.lower_bounds[0] = 0.01; // sigma2
            opts.upper_bounds[0] = 2.0;
            opts.lower_bounds[1] = 0.0; // nugget
            opts.upper_bounds[1] = 0.1;
            break;
        case KernelType::Matern12:
            opts.lower_bounds[0] = 0.01; // sigma2
            opts.upper_bounds[0] = 3.0;
            opts.lower_bounds[1] = 0.0; // nugget
            opts.upper_bounds[1] = 0.1;
            break;
        case KernelType::Matern32:
            opts.lower_bounds[0] = 0.01; // sigma2
            opts.upper_bounds[0] = 3.0;
            opts.lower_bounds[1] = 0.0; // nugget
            opts.upper_bounds[1] = 0.1;
            break;
        case KernelType::Matern52:
            opts.lower_bounds[0] = 0.01; // sigma2
            opts.upper_bounds[0] = 3.0;
            opts.lower_bounds[1] = 0.0; // nugget
            opts.upper_bounds[1] = 0.1;
            break;
        default:
            break;
    }
    optimizer.set_lower_bounds(opts.lower_bounds);
    optimizer.set_upper_bounds(opts.upper_bounds);

    // Set stopping criteria
    optimizer.set_xtol_rel(opts.xtol_rel);
    optimizer.set_ftol_rel(opts.ftol_rel);
    optimizer.set_maxeval(opts.maxeval);

    // Prepare the optimization data
    OptimizationData opt_data = {&gpuData, &opts, rank, MPI_COMM_WORLD};

    // Set the objective function
    optimizer.set_min_objective(objective_function, &opt_data);

    // print the config of optimizer
    if (rank == 0 && opts.print){
        std::cout << "Optimizer dimension: " << optimizer.get_dimension() << std::endl;
        std::cout << "Optimizer lower bounds: ";
        for (auto bound : opts.lower_bounds) {
            std::cout << bound << ", ";
        }
        std::cout << std::endl;
        std::cout << "Optimizer upper bounds: ";
        for (auto bound : opts.upper_bounds) {
            std::cout << bound << ", ";
        }
        std::cout << std::endl;
        std::cout << "Optimizer xtol_rel: " << opts.xtol_rel << std::endl;
        std::cout << "Optimizer ftol_rel: " << opts.ftol_rel << std::endl;
        std::cout << "Optimizer maxeval: " << opts.maxeval << std::endl;
    }

    // Perform the optimization
    std::vector<double> optimized_theta = opts.theta_init;  // Start with initial theta values
    double optimized_log_likelihood;
    
    try {
        nlopt::result result = optimizer.optimize(optimized_theta, optimized_log_likelihood);
        if (rank == 0 && opts.print) {
            std::cout << "Optimization result tag: " << result << std::endl;
            std::cout << "Optimized log-likelihood: " << -optimized_log_likelihood << std::endl;
            std::cout << "Optimized theta values: ";
            for (auto theta : optimized_theta) {
            std::cout << theta << " ";
            }
            std::cout << std::endl;
        }
    } catch (std::exception &e) {
        if (rank == 0 && opts.print) { 
            std::cerr << "Optimization failed: " << e.what() << std::endl;
        }
    }

    // Broadcast the final optimized theta to all processes
    MPI_Bcast(optimized_theta.data(), optimized_theta.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end_computation = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_computation = end_computation - start_computation;
    double max_duration_computation;
    double duration_computation_seconds = duration_computation.count();
    MPI_Allreduce(&duration_computation_seconds, &max_duration_computation, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0 && opts.print){
        std::cout << "-------------------Estimation Done-----------------" << std::endl;
    }
    // Step 3: Cleanup GPU memory
    auto start_cleanup_gpu = std::chrono::high_resolution_clock::now();
    cleanupGpuMemory(gpuData);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_cleanup_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_cleanup_gpu = end_cleanup_gpu - start_cleanup_gpu;
    double max_duration_cleanup_gpu;
    double duration_cleanup_gpu_seconds = duration_cleanup_gpu.count();
    MPI_Allreduce(&duration_cleanup_gpu_seconds, &max_duration_cleanup_gpu, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_total = end_total - start_total;
    double duration_total_seconds = duration_total.count();
    double max_duration_total;
    MPI_Allreduce(&duration_total_seconds, &max_duration_total, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // do the prediction for the test points
    GpuData gpuData_test;
    double mspe = -1.0;
    double rmspe = -1.0;
    double ci_coverage = -1.0;
    // std::vector<double> pre_range_factor(opts.distance_scale.begin(), opts.distance_scale.end());
    // std::vector<double> post_range_factor(optimized_theta.begin() + opts.range_offset, optimized_theta.end());
    // std::vector<double> new_range_scaled(pre_range_factor.size());
    // std::vector<double> new_theta(optimized_theta.begin(), optimized_theta.end());
    // for (size_t i = 0; i < pre_range_factor.size(); ++i) {
    //     new_range_scaled[i] = pre_range_factor[i] * post_range_factor[i];
    //     new_theta[i+opts.range_offset] = new_range_scaled[i];
    // }
    if (opts.mode == "prediction") {
        gpuData_test = copyDataToGPU(opts, localBlocks_test);
        std::tie(mspe, rmspe, ci_coverage) = performPredictionOnGPU(gpuData_test, optimized_theta, opts);
        cleanupGpuMemory(gpuData_test);
    }

    
    // save the time and gflops to a file
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        saveTimeAndGflops(
            max_RAC_partitioning, max_duration_centers_of_gravity, 
            max_duration_send_centers_of_gravity, 
            max_duration_reorder_centers, 
            max_duration_create_block_info, max_duration_block_sending, 
            max_duration_nn_searching, 
            max_duration_gpu_copy, max_duration_computation,
            max_duration_cleanup_gpu,
            max_duration_total,
            total_gflops, 
            opts.numPointsPerProcess, opts.numPointsTotal, 
            opts.numBlocksPerProcess, opts.numBlocksTotal, 
            opts.m, 
            opts.seed,
            mspe, rmspe, ci_coverage,optimized_theta.data(), -optimized_log_likelihood, opts);
    }
    
    MPI_Finalize();
    return EXIT_SUCCESS;
}
