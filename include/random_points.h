#ifndef RANDOM_POINTS_H
#define RANDOM_POINTS_H

#include <vector>
#include <utility>

// Structure to store command line options
struct Options {
    int numPointsPerProcess;
    int numBlocksX;
    int numBlocksY;
    int m;  // the number of nearest neighbor 
    bool print;
};

// Function to generate random 2D points
std::vector<std::pair<double, double>> generateRandomPoints(int numPointsPerProcess);

// Function to partition points and communicate them to the appropriate processors
void partitionPoints(const std::vector<std::pair<double, double>>& localPoints, std::vector<std::pair<double, double>>& allPoints);

// Function to perform finer partitioning within each processor
void finerPartition(const std::vector<std::pair<double, double>>& points, int numBlocksX, int numBlocksY, std::vector<std::vector<std::pair<double, double>>>& finerPartitions);

// Function to calculate centers of gravity for each block
std::vector<std::pair<double, double>> calculateCentersOfGravity(const std::vector<std::vector<std::pair<double, double>>>& finerPartitions);

// Function to send centers of gravity to processor 0
void sendCentersOfGravityToRoot(const std::vector<std::pair<double, double>>& centers, std::vector<std::pair<double, double>>& allCenters, bool print);

// Function to randomly reorder centers at processor 0
void reorderCenters(std::vector<std::pair<double, double>>& centers);

// Function to broadcast reordered centers to all processors
void broadcastCenters(const std::vector<std::pair<double, double>>& centers, std::vector<std::pair<double, double>>& allCenters, int numCenters);

#endif // RANDOM_POINTS_H
