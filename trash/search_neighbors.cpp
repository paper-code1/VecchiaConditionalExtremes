#include <mpi.h>
#include <iostream>
#include <vector>
#include <utility>
#include <cmath>
#include <numeric>
#include <map>
#include <set>
#include <algorithm>
#include "block_info.h"

#define DISTANCE_THRESHOLD 99

// Function to calculate the distance between two points
double calculateDistance(const std::pair<double, double> &point1, const std::pair<double, double> &point2)
{
    double dx = point1.first - point2.first;
    double dy = point1.second - point2.second;
    return std::sqrt(dx * dx + dy * dy);
}

// Function to build the one-to-one mapping of local centers and the corresponding finer partitions
void buildMapping(const std::vector<std::pair<double, double>> &localCenters,
                  const std::vector<std::pair<double, double>> &allCenters,
                  const std::vector<std::vector<std::pair<double, double>>> &finerPartitions,
                  std::map<int, int> &localToGlobalMap,
                  std::map<int, int> &globalToLocalMap)
{
    for (long unsigned int i = 0; i < localCenters.size(); ++i)
    {
        auto it = std::find(allCenters.begin(), allCenters.end(), localCenters[i]);
        int globalIndex = std::distance(allCenters.begin(), it);
        localToGlobalMap[i] = globalIndex;
        globalToLocalMap[globalIndex] = i;
    }
}

// Function to prepare the candidate pool for all-to-all communication
void prepareCandidatePool(const std::vector<std::pair<double, double>> &localCenters,
                          const std::vector<std::pair<double, double>> &allCenters,
                          const std::map<int, int> &localToGlobalMap,
                          std::set<int> &pool,
                          int m)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (const auto &[localIndex, globalIndex] : localToGlobalMap)
    {
        std::cout << "Rank: " << rank << ", local/global: " << localIndex << " " << globalIndex << "\n";
        for (int j = 0; j < globalIndex; ++j)
        {
            double distance = calculateDistance(allCenters[globalIndex], localCenters[j]);
            if (distance < DISTANCE_THRESHOLD)
            {
                pool.insert(j);
            }
        }

        if (int(pool.size()) < m)
        {
            for (int j = 0; j < globalIndex; ++j)
            {
                pool.insert(j);
            }
        }
    }
}

// Function for all-to-all communication to share the candidate pool
void shareCandidatePool(const std::set<int> &pool,
                        std::vector<std::vector<std::pair<double, double>>> &receivedBlocks)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> sendCounts(size, 0);
    std::vector<int> recvCounts(size, 0);

    std::vector<std::vector<std::pair<double, double>>> sendBuffers(size);
    for (int idx : pool)
    {
        int targetProcess = idx % size; // Assuming blocks are evenly distributed
        sendBuffers[targetProcess].push_back(std::make_pair(idx, rank));
    }

    for (int i = 0; i < size; ++i)
    {
        sendCounts[i] = sendBuffers[i].size() * 2;
    }

    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> sendDisplacements(size, 0);
    std::vector<int> recvDisplacements(size, 0);

    for (int i = 1; i < size; ++i)
    {
        sendDisplacements[i] = sendDisplacements[i - 1] + sendCounts[i - 1];
        recvDisplacements[i] = recvDisplacements[i - 1] + recvCounts[i - 1];
    }

    int totalSendCount = std::accumulate(sendCounts.begin(), sendCounts.end(), 0);
    int totalRecvCount = std::accumulate(recvCounts.begin(), recvCounts.end(), 0);

    std::vector<int> sendBuffer(totalSendCount);
    std::vector<int> recvBuffer(totalRecvCount);

    int offset = 0;
    for (int i = 0; i < size; ++i)
    {
        for (const auto &block : sendBuffers[i])
        {
            sendBuffer[offset++] = block.first;
            sendBuffer[offset++] = block.second;
        }
    }

    MPI_Alltoallv(sendBuffer.data(), sendCounts.data(), sendDisplacements.data(), MPI_INT,
                  recvBuffer.data(), recvCounts.data(), recvDisplacements.data(), MPI_INT, MPI_COMM_WORLD);

    for (int i = 0; i < totalRecvCount; i += 2)
    {
        int blockIdx = recvBuffer[i];
        int sourceRank = recvBuffer[i + 1];
        receivedBlocks[sourceRank].push_back(std::make_pair(blockIdx, sourceRank));
    }
}

// Function to find m nearest points for local centers
void findNearestNeighbors(const std::vector<std::pair<double, double>> &localCenters,
                          const std::vector<std::vector<std::pair<double, double>>> &allBlocks,
                          const std::map<int, int> &localToGlobalMap,
                          int m,
                          std::vector<std::vector<std::pair<double, double>>> &nearestNeighbors)
{
    nearestNeighbors.clear();
    nearestNeighbors.resize(localCenters.size());

    for (long unsigned int i = 0; i < localCenters.size(); ++i)
    {
        // int globalIndex = localToGlobalMap.at(i);

        std::vector<std::pair<double, std::pair<double, double>>> distances;

        for (const auto &block : allBlocks)
        {
            for (const auto &point : block)
            {
                double distance = calculateDistance(localCenters[i], point);
                distances.push_back(std::make_pair(distance, point));
            }
        }

        std::sort(distances.begin(), distances.end());

        for (int j = 0; j < std::min(m, (int)distances.size()); ++j)
        {
            nearestNeighbors[i].push_back(distances[j].second);
        }
    }
}
