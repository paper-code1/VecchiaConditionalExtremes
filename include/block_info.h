#ifndef BLOCK_INFO_H
#define BLOCK_INFO_H

#include <vector>
#include <utility>
#include "input_parser.h"

// Data structure to store block information
struct BlockInfo
{
    int localOrder;
    int globalOrder;
    std::vector<double> center;
    std::vector<std::vector<double>> blocks;
    std::vector<std::vector<double>> nearestNeighbors;
    std::vector<double> observations_blocks;
    std::vector<double> observations_nearestNeighbors;

    // Default constructor: the outer size is not defined initially
    BlockInfo() = default;

    // Method to add a new vector of a specified dimension size
    void addVector(int dim_size) {
       blocks.push_back(std::vector<double>(dim_size));  // Add a new dim-sized vector
    }
};

// point info
struct PointMetadata {
    std::vector<double> coordinates;
    double observation;
};

// Function to calculate the Euclidean distance between two points
double calculateDistance(const std::vector<double>& point1, const std::vector<double>& point2);

// Function to create block information for each processor
std::vector<BlockInfo> createBlockInfo(
    const std::vector<std::vector<PointMetadata>> &finerPartitions,
    const std::vector<std::vector<double>> &localCenters,
    const std::vector<std::vector<double>> &allCenters,
    const Opts& opts);

#endif // BLOCK_INFO_H
