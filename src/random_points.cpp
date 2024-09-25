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
std::vector<PointMetadata> generateRandomPoints(int numPointsPerProcess)
{
    std::vector<PointMetadata> pointsMetadata;
    pointsMetadata.resize(numPointsPerProcess);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::srand(rank + 1);

    for (int i = 0; i < numPointsPerProcess; ++i)
    {
        for (int j = 0; j < DIMENSION; ++j)
        {
            pointsMetadata[i].coordinates[j] = generateRandomDouble();
        }
        pointsMetadata[i].observation = generateRandomDouble();
    }

    return pointsMetadata;
}

// Function to partition points and communicate them to the appropriate processors
void partitionPoints(const std::vector<PointMetadata> &localMetadata, std::vector<PointMetadata> &allMetadata)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Prepare to send data to the appropriate process based on x value
    std::vector<int> sendCounts(size, 0);
    std::vector<int> sendDisplacements(size, 0);

    std::vector<std::vector<PointMetadata>> sendBuffers(size);

    for (const auto &pointmeta : localMetadata)
    {
        int targetProcess = std::min(static_cast<int>(pointmeta.coordinates[0] * size), size - 1);
        sendBuffers[targetProcess].push_back(pointmeta);
    }

    for (int i = 0; i < size; ++i)
    {
        sendCounts[i] = sendBuffers[i].size() * (DIMENSION + 1); // Each point has DIMENSION + 1 doubles
    }

    std::vector<double> sendData;
    for (int i = 0; i < size; ++i)
    {
        for (const auto &point : sendBuffers[i])
        {
            for (int j = 0; j < DIMENSION; ++j)
            {
                sendData.push_back(point.coordinates[j]);
            }
            sendData.push_back(point.observation);
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

    // Convert received data back to PointMetadata
    allMetadata.clear();
    for (int i = 0; i < totalRecvCount; i += (DIMENSION + 1))
    {
        PointMetadata pointMetadata;
        for (int j = 0; j < DIMENSION; ++j)
        {
            pointMetadata.coordinates[j] = recvData[i + j];
        }
        pointMetadata.observation = recvData[i + DIMENSION];
        allMetadata.push_back(pointMetadata);
    }
}

// Function to perform finer partitioning within each processor (specific for 2D)
void finerPartition(const std::vector<PointMetadata> &metadata, int numBlocksX, int numBlocksY, std::vector<std::vector<PointMetadata>> &finerPartitions)
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

    for (const auto &pointmeta : metadata)
    {
        int blockX = std::min(static_cast<int>((pointmeta.coordinates[0] - xMin) / blockWidth), numBlocksX - 1);
        int blockY = std::min(static_cast<int>(pointmeta.coordinates[1] / blockHeight), numBlocksY - 1);
        finerPartitions[blockY * numBlocksX + blockX].push_back(pointmeta);
    }
}

// Function to calculate centers of gravity for each block (specific for 2D)
std::vector<std::pair<double, double>> calculateCentersOfGravity(const std::vector<std::vector<PointMetadata>> &finerPartitions)
{
    std::vector<std::pair<double, double>> centers;
    centers.resize(finerPartitions.size());

    for (size_t i = 0; i < finerPartitions.size(); ++i)
    {
        auto &blockmetadata = finerPartitions[i];
        if (blockmetadata.empty())
        {
            continue;
        }
        double sumX = 0.0, sumY = 0.0;
        for (auto &pointmeta : blockmetadata)
        {
            sumX += pointmeta.coordinates[0];
            sumY += pointmeta.coordinates[1];
        }
        double centerX = sumX / blockmetadata.size();
        double centerY = sumY / blockmetadata.size();
        centers[i] = std::make_pair(centerX, centerY);
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
        allCenters.resize(totalCenters);
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