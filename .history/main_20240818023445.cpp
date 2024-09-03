#include <mpi.h>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include "random_points.h"
#include "block_info.h"
#include "distance_calc.h"

bool parse_args(int argc, char **argv, Options &opts);

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Options opts = {0, 0, 0, 120, false};

    if (!parse_args(argc, argv, opts))
    {
        std::cerr << "Usage: " << argv[0] << " --num_loc <number_of_locations> --sub_partition <num_blocks_x> <num_blocks_y> [--print]" << std::endl;
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    std::vector<size_t> free_memory(num_gpus);
    std::vector<size_t> total_memory(num_gpus);

    // Query each GPU for its available memory
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        cudaSetDevice(gpu_id);
        cudaMemGetInfo(&free_memory[gpu_id], &total_memory[gpu_id]);
    }

    // Sort GPUs by available memory (optional, depending on the distribution strategy)
    std::vector<int> gpu_ids(num_gpus);
    for (int i = 0; i < num_gpus; i++) gpu_ids[i] = i;
    std::sort(gpu_ids.begin(), gpu_ids.end(), [&free_memory](int a, int b) {
        return free_memory[a] > free_memory[b];
    });

    // Distribute processes to GPUs based on the available memory
    int processes_per_gpu = size / num_gpus;
    int remainder = size % num_gpus;
    int assigned_gpu = rank / processes_per_gpu;
    if (rank >= processes_per_gpu * num_gpus) {
        assigned_gpu = gpu_ids[rank % num_gpus];
    }

    cudaSetDevice(assigned_gpu);

    // Output assigned GPU for debugging purposes
    std::cout << "MPI Rank " << rank << " assigned to GPU " << assigned_gpu << std::endl;

    // 1. Generate random points
    std::vector<std::pair<double, double>> localPoints = generateRandomPoints(opts.numPointsPerProcess);

    // 2.1 Partition points and communicate them
    std::vector<std::pair<double, double>> allPoints;
    partitionPoints(localPoints, allPoints);

    // 2.2 Perform finer partitioning within each processor
    std::vector<std::vector<std::pair<double, double>>> finerPartitions;
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
    processAndSendBlocks(blockInfos, allCenters, opts.m);


    // // debug: Print block information on each processor
    // if (rank == 0)
    // {
    //     if (opts.print)
    //     {
    //         std::cout << "Processor " << rank << " block information:\n";
    //         for (const auto &blockInfo : blockInfos)
    //         {
    //             std::cout << "Local Order: " << blockInfo.localOrder
    //                       << ", Global Order: " << blockInfo.globalOrder
    //                       << ", Center: (" << blockInfo.center.first << ", " << blockInfo.center.second << ")\n";
    //             std::cout << "Points:\n";
    //             for (const auto &point : blockInfo.points)
    //             {
    //                 std::cout << "(" << point.first << ", " << point.second << ")\n";
    //             }
    //         }
    //     }
    // }

    // 5. independent computation of log-likelihood

    MPI_Finalize();
    return EXIT_SUCCESS;
}
