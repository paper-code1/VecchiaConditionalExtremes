#include <mpi.h>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <chrono>
#include <thread>
#include "random_points.h"
#include "block_info.h"
#include "distance_calc.h"
#include "gpu_operations.h"
#include "input_parser.h"
#include "gpu_covariance.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
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
        std::cout << "Number of points per process: " << opts.numPointsPerProcess << std::endl;
        std::cout << "Number of blocks in X direction: " << opts.numBlocksX << std::endl;
        std::cout << "Number of blocks in Y direction: " << opts.numBlocksY << std::endl;
        std::cout << "M value: " << opts.m << std::endl;
        std::cout << "Theta: " << opts.theta[0] << ", " << opts.theta[1] << ", " << opts.theta[2] << std::endl;
    }

    // 1. Generate random points
    std::vector<PointMetadata> localPoints = generateRandomPoints(opts.numPointsPerProcess);

    // 2.1 Partition points and communicate them
    std::vector<PointMetadata> allPoints;
    partitionPoints(localPoints, allPoints);

    // 2.2 Perform finer partitioning within each processor
    std::vector<std::vector<PointMetadata>> finerPartitions;
    finerPartition(allPoints, opts.numBlocksX, opts.numBlocksY, finerPartitions);

    // 2.3 Calculate centers of gravity for each block
    std::vector<std::pair<double, double>> centers = calculateCentersOfGravity(finerPartitions);

    // 2.4 Send centers of gravity to processor 0
    std::vector<std::pair<double, double>> allCenters;
    sendCentersOfGravityToRoot(centers, allCenters, opts.print);

    // 3. Reorder centers at processor 0
    reorderCenters(allCenters);

    // 3.1 Broadcast reordered centers to all processors
    int numCenters = allCenters.size();
    MPI_Bcast(&numCenters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    broadcastCenters(allCenters, allCenters, numCenters);

    // 4. NN searching
    // 4.1 Create block information
    std::vector<BlockInfo> blockInfos = createBlockInfo(finerPartitions, centers, allCenters);

    // 4.2 Block sent info
    // // (if block is a candidiate for other blocks, and then
    // // send it to corresponding processors)
    // // Here, you could set the first 100 need to obtain all previous blocks
    // Process and send blocks based on distance threshold and special rule for the first 100 blocks
    processAndSendBlocks(blockInfos, allCenters, opts.m, opts.distance_threshold);
    MPI_Barrier(MPI_COMM_WORLD);
    // 5. independent computation of log-likelihood
    // Determine which GPU to use based on the rank
    int gpu_id = (rank < 20) ? 0 : 1; // this is used for personal server, hen/swan/...

    // Step 1: Copy data to GPU
    GpuData gpuData = copyDataToGPU(gpu_id, blockInfos);
    
    // Step 2: Perform computation
    performComputationOnGPU(gpuData, opts.theta);

    // Step 3: Cleanup GPU memory
    cleanupGpuMemory(gpuData);
    
    MPI_Finalize();
    return EXIT_SUCCESS;
}
