#include <mpi.h>
#include <iostream>
#include <numeric>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include "random_points.h"

// Function to generate random double between 0 and 1
double generateRandomDouble()
{
    return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}

// Function to generate random points
std::vector<std::pair<double, double>> generateRandomPoints(int numPointsPerProcess)
{
    std::vector<std::pair<double, double>> points;
    points.reserve(numPointsPerProcess);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::srand(rank + 1);

    for (int i = 0; i < numPointsPerProcess; ++i)
    {
        points.emplace_back(generateRandomDouble(), generateRandomDouble());
    }

    return points;
}

// Function to partition points and communicate them to the appropriate processors
void partitionPoints(const std::vector<std::pair<double, double>> &localPoints, std::vector<std::pair<double, double>> &allPoints)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Prepare to send data to the appropriate process based on x value
    std::vector<int> sendCounts(size, 0);
    std::vector<int> sendDisplacements(size, 0);

    std::vector<std::vector<std::pair<double, double>>> sendBuffers(size);

    for (const auto &point : localPoints)
    {
        int targetProcess = std::min(static_cast<int>(point.first * size), size - 1);
        sendBuffers[targetProcess].push_back(point);
    }

    for (int i = 0; i < size; ++i)
    {
        sendCounts[i] = sendBuffers[i].size() * 2; // Each point has 2 doubles
    }

    std::vector<double> sendData;
    for (int i = 0; i < size; ++i)
    {
        for (const auto &point : sendBuffers[i])
        {
            sendData.push_back(point.first);
            sendData.push_back(point.second);
        }
    }

    // Calculate displacements
    for (int i = 1; i < size; ++i)
    {
        sendDisplacements[i] = sendDisplacements[i - 1] + sendCounts[i - 1];
    }

    // Prepare receive counts and displacements
    std::vector<int> recvCounts(size, 0);
    std::vector<int> recvDisplacements(size, 0);

    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    for (int i = 1; i < size; ++i)
    {
        recvDisplacements[i] = recvDisplacements[i - 1] + recvCounts[i - 1];
    }

    int totalRecvCount = std::accumulate(recvCounts.begin(), recvCounts.end(), 0);
    std::vector<double> recvData(totalRecvCount);

    MPI_Alltoallv(sendData.data(), sendCounts.data(), sendDisplacements.data(), MPI_DOUBLE,
                  recvData.data(), recvCounts.data(), recvDisplacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    // Convert received data back to pairs
    allPoints.clear();
    for (int i = 0; i < totalRecvCount; i += 2)
    {
        allPoints.emplace_back(recvData[i], recvData[i + 1]);
    }
}

// Function to perform finer partitioning within each processor
void finerPartition(const std::vector<std::pair<double, double>> &points, int numBlocksX, int numBlocksY, std::vector<std::vector<std::pair<double, double>>> &finerPartitions)
{
    finerPartitions.clear();
    finerPartitions.resize(numBlocksX * numBlocksY);

    // Get the bounds for the current processor
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double xMin = static_cast<double>(rank) / size;
    double xMax = static_cast<double>(rank + 1) / size;
    double blockWidth = (xMax - xMin) / numBlocksX;
    double blockHeight = 1.0 / numBlocksY;

    for (const auto &point : points)
    {
        int blockX = std::min(static_cast<int>((point.first - xMin) / blockWidth), numBlocksX - 1);
        int blockY = std::min(static_cast<int>(point.second / blockHeight), numBlocksY - 1);
        finerPartitions[blockY * numBlocksX + blockX].push_back(point);
    }
}

// Function to calculate centers of gravity for each block
std::vector<std::pair<double, double>> calculateCentersOfGravity(const std::vector<std::vector<std::pair<double, double>>> &finerPartitions)
{
    std::vector<std::pair<double, double>> centers;
    centers.reserve(finerPartitions.size());

    for (const auto &block : finerPartitions)
    {
        if (block.empty())
        {
            continue;
        }
        double sumX = 0.0, sumY = 0.0;
        for (const auto &point : block)
        {
            sumX += point.first;
            sumY += point.second;
        }
        double centerX = sumX / block.size();
        double centerY = sumY / block.size();
        centers.emplace_back(centerX, centerY);
    }

    return centers;
}

// Function to send centers of gravity to processor 0
void sendCentersOfGravityToRoot(const std::vector<std::pair<double, double>> &centers, std::vector<std::pair<double, double>> &allCenters, bool print)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numCenters = centers.size();
    std::vector<int> recvCounts(size, 0);
    MPI_Gather(&numCenters, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displacements(size, 0);
    int totalCenters = 0;
    if (rank == 0)
    {
        for (int i = 0; i < size; ++i)
        {
            displacements[i] = totalCenters * 2;
            totalCenters += recvCounts[i];
            recvCounts[i] *= 2; // Each center has 2 doubles
        }
    }

    std::vector<double> sendBuffer(numCenters * 2);
    for (int i = 0; i < numCenters; ++i)
    {
        sendBuffer[2 * i] = centers[i].first;
        sendBuffer[2 * i + 1] = centers[i].second;
    }

    std::vector<double> recvBuffer;
    if (rank == 0)
    {
        recvBuffer.resize(totalCenters * 2);
    }

    MPI_Gatherv(sendBuffer.data(), numCenters * 2, MPI_DOUBLE, recvBuffer.data(), recvCounts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        allCenters.clear();
        allCenters.reserve(totalCenters);
        for (int i = 0; i < totalCenters * 2; i += 2)
        {
            allCenters.emplace_back(recvBuffer[i], recvBuffer[i + 1]);
        }

        // if (print)
        // {
        //     // Print all centers for verification
        //     std::cout << "Centers of gravity received by processor 0:\n";
        //     int i = 0;
        //     for (const auto &center : allCenters)
        //     {
        //         std::cout << i++ << "th: (" << center.first << ", " << center.second << ")\n";
        //     }
        // }
    }
}

// Function to randomly reorder centers at processor 0
void reorderCenters(std::vector<std::pair<double, double>> &centers)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        // Shuffle non-empty centers
        std::random_shuffle(centers.begin(), centers.end());
    }
}

// Function to broadcast reordered centers to all processors
void broadcastCenters(const std::vector<std::pair<double, double>> &centers, std::vector<std::pair<double, double>> &all_centers, int numCenters)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Prepare data for broadcast
    std::vector<double> centersData(numCenters * 2);
    if (rank == 0)
    {
        for (int i = 0; i < numCenters; ++i)
        {
            centersData[2 * i] = centers[i].first;
            centersData[2 * i + 1] = centers[i].second;
        }
    }

    // Broadcast the reordered centers to all processes
    MPI_Bcast(centersData.data(), numCenters * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Resize the all_centers vector
    all_centers.resize(numCenters);

    // Convert data back to pairs
    for (int i = 0; i < numCenters; ++i)
    {
        all_centers[i] = {centersData[2 * i], centersData[2 * i + 1]};
    }
}