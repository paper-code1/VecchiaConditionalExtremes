#include "block_info.h"
#include <algorithm>
#include <mpi.h>
#include <cmath>
#include <iostream>

// Function to create block information for each processor
std::vector<BlockInfo> createBlockInfo(const std::vector<std::vector<PointMetadata>>& finerPartitions, 
                                       const std::vector<std::pair<double, double>>& localCenters, 
                                       const std::vector<std::pair<double, double>>& allCenters) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<BlockInfo> blockInfos;
    blockInfos.resize(localCenters.size());

    for (size_t i = 0; i < localCenters.size(); ++i) {
        const auto& localCenter = localCenters[i];

        // Find the global order of the current local center among the all centers
        auto it = std::find(allCenters.begin(), allCenters.end(), localCenter);
        int globalOrder = (it != allCenters.end()) ? std::distance(allCenters.begin(), it) : -1; // Use -1 if not found
        if (globalOrder == -1) {
            std::cout << "Error: local center not found in all centers" << std::endl;
            exit(1);
        }

        // Create BlockInfo structure
        BlockInfo blockInfo;
        blockInfo.localOrder = i;
        blockInfo.globalOrder = globalOrder;
        blockInfo.center = localCenter;
        for (const auto& pointMetadata : finerPartitions[i]) {
            blockInfo.points.push_back(pointMetadata.coordinates);
            blockInfo.observations_points.push_back(pointMetadata.observation);
        }

        blockInfos[i] = blockInfo;
    }

    return blockInfos;
}

// Function to calculate distance between two points (only for 2D)
double calculateDistance(const std::pair<double, double>& point1, const std::pair<double, double>& point2) {
    double dx = point1.first - point2.first;
    double dy = point1.second - point2.second;
    return std::sqrt(dx * dx + dy * dy);
}