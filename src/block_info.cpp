#include "block_info.h"
#include <algorithm>
#include <mpi.h>
#include <cmath>
#include <omp.h>
#include <iostream>
#include "flops.h"

// Function to create block information for each processor
std::vector<BlockInfo> createBlockInfo_test(const std::vector<std::vector<PointMetadata>> &finerPartitions,
                                            const std::vector<std::vector<double>> &localCenters,
                                            const Opts &opts)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<BlockInfo> blockInfos;
    int numBlocksLocal = localCenters.size();
    blockInfos.resize(numBlocksLocal);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < numBlocksLocal; ++i)
    {
        const auto &localCenter = localCenters[i];

        // Create BlockInfo structure
        BlockInfo blockInfo;
        blockInfo.localOrder = i;
        blockInfo.globalOrder = std::numeric_limits<int>::max(); // maxint means prediction considering all blocks
        blockInfo.center = localCenter;
        for (const auto &pointMetadata : finerPartitions[i])
        {
            blockInfo.blocks.push_back(pointMetadata.coordinates);
            blockInfo.observations_blocks.push_back(pointMetadata.observation);
        }

        blockInfos[i] = blockInfo;
    }

    return blockInfos;
}

// Function to create block information for each processor
std::vector<BlockInfo> createBlockInfo(const std::vector<std::vector<PointMetadata>> &finerPartitions,
                                       const std::vector<std::vector<double>> &localCenters,
                                       const std::vector<std::pair<std::vector<double>, int>> &allCenters,
                                       const std::vector<int> &permutation,
                                       const std::vector<int> &localPermutation,
                                       const Opts &opts)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<BlockInfo> blockInfos;
    int numBlocksLocal = localCenters.size();
    blockInfos.resize(numBlocksLocal);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < numBlocksLocal; ++i)
    {
        const auto &localCenter = localCenters[i];

        // // Find the global order of the current local center among the all centers
        // auto it = std::find_if(allCenters.begin(), allCenters.end(), 
        //                       [&localCenter](const std::pair<std::vector<double>, int>& centerPair) {
        //                           return centerPair.first == localCenter;
        //                       });
        // int globalOrder = (it != allCenters.end()) ? std::distance(allCenters.begin(), it) : -1; // Use -1 if not found

        int globalOrder = permutation[localPermutation[i]];
        
        // // only used for debugging
        // // Since we're in a parallel region, use a critical section for error handling
        // if (globalOrder == -1)
        // {
        //     #pragma omp critical
        //     {
        //         std::cout << "Error: local center not found in all centers" << std::endl;
        //         exit(1);  // Note: Better error handling would be preferable in parallel code
        //     }
        // }

        // Create BlockInfo structure locally to avoid race conditions
        BlockInfo blockInfo;
        blockInfo.localOrder = i;
        blockInfo.globalOrder = globalOrder;
        blockInfo.center = localCenter;
        for (const auto &pointMetadata : finerPartitions[i])
        {
            blockInfo.blocks.push_back(pointMetadata.coordinates);
            blockInfo.observations_blocks.push_back(pointMetadata.observation);
        }

        // Each thread writes to its own designated position in blockInfos
        // This is safe since we pre-allocated with resize()
        blockInfos[i] = blockInfo;
    }

    return blockInfos;
}

// Function to calculate distance between two points (high dimensional)
double calculateDistance(const std::vector<double> &point1, const std::vector<double> &point2)
{
    double dist = 0;
    for (int i = 0; i < point1.size(); ++i)
    {
        double dx = point1[i] - point2[i];
        dist += dx * dx;
    }
    return std::sqrt(dist);
}