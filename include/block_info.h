#ifndef BLOCK_INFO_H
#define BLOCK_INFO_H

#include <vector>
#include <utility>
#include <array>

#define DIMENSION 2

// Data structure to store block information
struct BlockInfo
{
    int localOrder;
    int globalOrder;
    std::pair<double, double> center;
    std::vector<std::array<double, DIMENSION>> points;
    std::vector<std::array<double, DIMENSION>> nearestNeighbors;
    std::vector<double> observations_points;
    std::vector<double> observations_nearestNeighbors;
};

// point info
struct PointMetadata {
    std::array<double, DIMENSION> coordinates;
    double observation;
};

// Function to calculate the Euclidean distance between two points
double calculateDistance(const std::pair<double, double>& point1, const std::pair<double, double>& point2);

// Function to create block information for each processor
std::vector<BlockInfo> createBlockInfo(
    const std::vector<std::vector<PointMetadata>> &finerPartitions,
    const std::vector<std::pair<double, double>> &localCenters,
    const std::vector<std::pair<double, double>> &allCenters);

#endif // BLOCK_INFO_H
