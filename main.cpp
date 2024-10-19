#include <mpi.h>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <chrono>
#include <thread>
#include <iomanip>
#include <magma_v2.h>
#include "input_parser.h"
#include "block_info.h"
#include "random_points.h"
#include "distance_calc.h"
#include "gpu_covariance.h"
#include "gpu_operations.h"
#include "vecchia_helper.h"
#include <nlopt.hpp>

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
    if (opt_data->rank == 0 && !opt_data->opts->perf) {
        std::cout << "Optimization step: " << opts->current_iter++ << ", ";
        std::cout << "f(theta): " << std::fixed << std::setprecision(6) << result << ", ";
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

    // Use the parsed options
    if (rank == 0 && opts.print) {
        std::cout << "Number of total points: " << opts.numPointsTotal << std::endl;
        std::cout << "Number of total blocks: " << opts.numBlocksTotal << std::endl;
        std::cout << "The number of nearest neighbors: " << opts.m << std::endl;
        std::cout << "The distance threshold: " << opts.distance_threshold << std::endl;
        // print the varied size of theta
        std::cout << "Theta: ";
        for (auto theta : opts.theta) {
            std::cout << theta << ", ";
        }
        std::cout << std::endl;
    }

    // 1. Generate random points
    std::vector<PointMetadata> localPoints;
    std::vector<PointMetadata> localPoints_test;
    if (opts.train_metadata_path != ""){
        localPoints = readPointsConcurrently(opts.train_metadata_path, opts);
        localPoints_test = readPointsConcurrently(opts.test_metadata_path, opts);
    }
    else{
        localPoints = generateRandomPoints(opts.numPointsPerProcess, opts);
        localPoints_test = generateRandomPoints(opts.numPointsPerProcess_test, opts);
    }

    // time preprocessing
    auto start_preprocessing = std::chrono::high_resolution_clock::now();
    
    // 2.1 Partition points and communicate them
    std::vector<PointMetadata> localPoints_out_partitioned;
    partitionPoints(localPoints, localPoints_out_partitioned, opts);
    MPI_Barrier(MPI_COMM_WORLD);

    auto end_preprocessing = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_preprocessing = end_preprocessing - start_preprocessing;
    // find the maximum duration_preprocessing
    double avg_duration_preprocessing;
    double duration_preprocessing_seconds = duration_preprocessing.count();
    MPI_Allreduce(&duration_preprocessing_seconds, &avg_duration_preprocessing, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    avg_duration_preprocessing /= size;


    // 2.2 Perform finer partitioning within each processor
    std::vector<std::vector<PointMetadata>> finerPartitions;
    finerPartition(localPoints_out_partitioned, opts.numBlocksPerProcess, finerPartitions, opts);

    // 2.3 Calculate centers of gravity for each block
    std::vector<std::vector<double>> centers = calculateCentersOfGravity(finerPartitions, opts);
    MPI_Barrier(MPI_COMM_WORLD);
    // 2.4 Send centers of gravity to processor 0
    std::vector<std::vector<double>> allCenters;
    sendCentersOfGravityToRoot(centers, allCenters, opts);

    // 3. Reorder centers at processor 0
    reorderCenters(allCenters, opts);
    MPI_Barrier(MPI_COMM_WORLD);
    // 3.1 Broadcast reordered centers to all processors
    int numCenters = allCenters.size();
    MPI_Bcast(&numCenters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    broadcastCenters(allCenters, numCenters, opts);
    MPI_Barrier(MPI_COMM_WORLD);
    // 4. NN searching
    // 4.1 Create block information
    std::vector<BlockInfo> localBlocks = createBlockInfo(finerPartitions, centers, allCenters, opts);
    MPI_Barrier(MPI_COMM_WORLD);

    // 4.2 Block candidate preparation
    // Here, you could set the first 100 need to obtain all previous blocks and set threshold
    auto start_block_sending = std::chrono::high_resolution_clock::now();
    std::vector<BlockInfo> receivedBlocks = processAndSendBlocks(localBlocks, allCenters, opts.m, opts.distance_threshold, opts);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_block_sending = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_block_sending = end_block_sending - start_block_sending;
    double avg_duration_block_sending;
    double duration_block_sending_seconds = duration_block_sending.count();
    MPI_Allreduce(&duration_block_sending_seconds, &avg_duration_block_sending, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    avg_duration_block_sending /= size;

    // 4.3 NN searching
    nearest_neighbor_search(localBlocks, receivedBlocks, opts);
    // free the memory of receivedBlocks
    // receivedBlocks.clear();
    MPI_Barrier(MPI_COMM_WORLD);
    // 5. independent computation of log-likelihood
    
    // Step 1: Copy data to GPU
    GpuData gpuData = copyDataToGPU(opts, localBlocks);

    // // calculate the tota flops
    double total_gflops = gflopsTotal(gpuData, opts);

    // Step 2: Perform computation with BOBYQA optimization
    auto start_computation = std::chrono::high_resolution_clock::now();

    // Set up the optimization problem
    nlopt::opt optimizer(nlopt::LN_BOBYQA, opts.theta.size());
    
    // Set bounds for theta (adjust these as needed)
    optimizer.set_lower_bounds(opts.lower_bounds);
    optimizer.set_upper_bounds(opts.upper_bounds);

    // Set stopping criteria
    optimizer.set_xtol_rel(opts.xtol_rel);
    optimizer.set_maxeval(opts.maxeval);  // Maximum number of function evaluations

    // Prepare the optimization data
    OptimizationData opt_data = {&gpuData, &opts, rank, MPI_COMM_WORLD};

    // Set the objective function
    optimizer.set_min_objective(objective_function, &opt_data);

    // Perform the optimization
    std::vector<double> optimized_theta = opts.theta_init;  // Start with initial theta values
    double optimized_log_likelihood;
    
    try {
        nlopt::result result = optimizer.optimize(optimized_theta, optimized_log_likelihood);
        if (rank == 0 && !opts.perf) {
            std::cout << "Optimization result: " << result << std::endl;
            std::cout << "Optimized log-likelihood: " << optimized_log_likelihood << std::endl;
            std::cout << "Optimized theta values: ";
            for (auto theta : optimized_theta) {
            std::cout << theta << " ";
            }
            std::cout << std::endl;
        }
    } catch (std::exception &e) {
        if (rank == 0 && !opts.perf) { 
            std::cerr << "Optimization failed: " << e.what() << std::endl;
        }
    }

    // Broadcast the final optimized theta to all processes
    MPI_Bcast(optimized_theta.data(), optimized_theta.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end_computation = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_computation = end_computation - start_computation;
    double avg_duration_computation;
    double duration_computation_seconds = duration_computation.count();
    MPI_Allreduce(&duration_computation_seconds, &avg_duration_computation, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    avg_duration_computation /= size;

    
    // save the time and gflops to a file
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        saveTimeAndGflops(avg_duration_preprocessing, avg_duration_computation, avg_duration_block_sending, total_gflops, opts.numPointsPerProcess, opts.numPointsPerProcess, opts.numBlocksTotal, opts.m, opts.seed);
    }

    // Step 3: Cleanup GPU memory
    cleanupGpuMemory(gpuData);
    
    MPI_Finalize();
    return EXIT_SUCCESS;
}
