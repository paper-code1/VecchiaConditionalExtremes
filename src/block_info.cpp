#include "block_info.h"
#include <algorithm>
#include <mpi.h>
#include <cmath>
#include <iostream>

// Function to create block information for each processor
std::vector<BlockInfo> createBlockInfo(const std::vector<std::vector<std::pair<double, double>>>& finerPartitions, 
                                       const std::vector<std::pair<double, double>>& localCenters, 
                                       const std::vector<std::pair<double, double>>& allCenters) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<BlockInfo> blockInfos;

    for (size_t i = 0; i < localCenters.size(); ++i) {
        const auto& localCenter = localCenters[i];

        // Find the global order of the current local center
        auto it = std::find(allCenters.begin(), allCenters.end(), localCenter);
        int globalOrder = std::distance(allCenters.begin(), it);

        // Create BlockInfo structure
        BlockInfo blockInfo;
        blockInfo.localOrder = i;
        blockInfo.globalOrder = globalOrder;
        blockInfo.center = localCenter;
        blockInfo.points = finerPartitions[i];

        blockInfos.push_back(blockInfo);
    }

    return blockInfos;
}

// Function to calculate distance between two points
double calculateDistance(const std::pair<double, double>& point1, const std::pair<double, double>& point2) {
    double dx = point1.first - point2.first;
    double dy = point1.second - point2.second;
    return std::sqrt(dx * dx + dy * dy);
}