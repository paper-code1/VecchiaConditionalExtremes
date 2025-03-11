#include "block_info.h"
#include <algorithm>
#include <mpi.h>
#include <cmath>
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
                                       const Opts &opts)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<BlockInfo> blockInfos;
    int numBlocksLocal = localCenters.size();
    blockInfos.resize(numBlocksLocal);

    for (int i = 0; i < numBlocksLocal; ++i)
    {
        const auto &localCenter = localCenters[i];

        // Find the global order of the current local center among the all centers
        auto it = std::find_if(allCenters.begin(), allCenters.end(), 
                              [&localCenter](const std::pair<std::vector<double>, int>& centerPair) {
                                  return centerPair.first == localCenter;
                              });
        int globalOrder = (it != allCenters.end()) ? std::distance(allCenters.begin(), it) : -1; // Use -1 if not found
        if (globalOrder == -1)
        {
            std::cout << "Error: local center not found in all centers" << std::endl;
            exit(1);
        }

        // Create BlockInfo structure
        BlockInfo blockInfo;
        blockInfo.localOrder = i;
        blockInfo.globalOrder = globalOrder;
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