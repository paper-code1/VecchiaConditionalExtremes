#include <mpi.h>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <chrono>
#include <thread>
#include <magma_v2.h>
#include "input_parser.h"
#include "block_info.h"
#include "random_points.h"
#include "distance_calc.h"
#include "gpu_covariance.h"
#include "gpu_operations.h"
#include "vecchia_helper.h"

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
        // print the varied size of theta
        std::cout << "Theta: ";
        for (auto theta : opts.theta) {
            std::cout << theta << ", ";
        }
        std::cout << std::endl;
    }

    // 1. Generate random points
    std::vector<PointMetadata> localPoints;
    if (opts.train_metadata_path != ""){
        localPoints = readPointsConcurrently(opts.train_metadata_path, opts);
    }
    else{
        localPoints = generateRandomPoints(opts.numPointsPerProcess, opts);
    }

    // // // print the localPoints
    // if (rank == 0) {
    //     // std::cout << "Local points: " << std::endl;
    //     for (int i = 0; i < opts.numPointsPerProcess; ++i) {
    //         std::cout << "\"(" ;
    //         for (int j = 0; j < opts.dim; ++j) {
    //             std::cout << localPoints[i].coordinates[j] << ", ";
    //         }   
    //         std::cout << localPoints[i].observation << ")\" ," << std::endl;
    //     }
    // }
    // MPI_Barrier(MPI_COMM_WORLD);

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

    // Step 2: Perform computation
    // time the computation in seconds
    auto start_computation = std::chrono::high_resolution_clock::now();
    double log_likelihood = performComputationOnGPU(gpuData, opts.theta, opts);
    // print the log_likelihood
    if (rank == 0){
        std::cout << "log_likelihood: " << log_likelihood << std::endl;
    }
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
