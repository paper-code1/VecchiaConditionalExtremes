#include "distance_calc.h"
#include "block_info.h"
#include <mpi.h>
#include <cmath>
#include <iostream>
#include <set>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <fstream>

// Function to pack block information into buffers
void packBlockInfo(const std::vector<BlockInfo> &blocks, std::vector<double> &buffer)
{
    for (const auto &block : blocks)
    {
        buffer.push_back(static_cast<double>(block.localOrder));
        buffer.push_back(static_cast<double>(block.globalOrder));
        buffer.push_back(block.center.first);
        buffer.push_back(block.center.second);
        buffer.push_back(static_cast<double>(block.points.size()));
        for (const auto &point : block.points)
        {
            for (const auto &coord : point)
            {
                buffer.push_back(coord);
            }
        }
        for (const auto &obs: block.observations_points)
        {
            buffer.push_back(obs);
        }
    }
}

// Function to unpack block information from buffers
std::vector<BlockInfo> unpackBlockInfo(const std::vector<double> &buffer)
{
    std::vector<BlockInfo> blocks;
    double x = 0.;
    double y = 0.;
    size_t i = 0;
    while (i < buffer.size())
    {
        BlockInfo block;
        block.localOrder = static_cast<int>(buffer[i++]);
        block.globalOrder = static_cast<int>(buffer[i++]);
        x = buffer[i++];
        y = buffer[i++];
        block.center = {x, y};
        int numPoints = static_cast<int>(buffer[i++]);
        block.points.resize(numPoints);
        block.observations_points.resize(numPoints);
        for (int j = 0; j < numPoints; ++j)
        {
            x = buffer[i++];
            y = buffer[i++];
            block.points[j] = {x, y};
        }
        for (int j = 0; j < numPoints; ++j)
        {
            block.observations_points[j] = buffer[i++];
        }
        blocks.push_back(block);
    }
    return blocks;
}

void processAndSendBlocks(std::vector<BlockInfo> &blockInfos, const std::vector<std::pair<double, double>> &allCenters, int m, double distance_threshold)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numBlocks = allCenters.size();

    // Prepare buffers to send blocks to other processors
    std::vector<std::set<int>> blockIndexSets(size);
    std::vector<std::vector<BlockInfo>> sendBuffers(size);

    for (const auto &blockInfo : blockInfos)
    {
        int globalOrder = blockInfo.globalOrder;

        // Send first 10 blocks to all processors
        if (globalOrder < 10)
        {
            for (int dest = 0; dest < size; ++dest)
            {
                if (blockIndexSets[dest].find(globalOrder) == blockIndexSets[dest].end())
                {
                    sendBuffers[dest].push_back(blockInfo);
                    blockIndexSets[dest].insert(globalOrder);
                }
            }
        }

        // Calculate distance to future global centers and check against threshold
        for (int j = globalOrder + 1; j < numBlocks; ++j)
        {
            double distance = calculateDistance(blockInfo.center, allCenters[j]);
            if (distance < distance_threshold)
            {
                int dest = std::min(static_cast<int>(allCenters[j].first * size), size - 1);
                if (blockIndexSets[dest].find(globalOrder) == blockIndexSets[dest].end())
                {
                    sendBuffers[dest].push_back(blockInfo);
                    blockIndexSets[dest].insert(globalOrder);
                }
            }
        }
    }

    // Pack data into buffers
    std::vector<std::vector<double>> packedSendBuffers(size);
    for (int dest = 0; dest < size; ++dest)
    {
        packBlockInfo(sendBuffers[dest], packedSendBuffers[dest]);
    }

    // Prepare to send data using MPI_Alltoallv
    std::vector<int> sendCounts(size), sendDispls(size);
    std::vector<int> recvCounts(size), recvDispls(size);

    for (int dest = 0; dest < size; ++dest)
    {
        sendCounts[dest] = packedSendBuffers[dest].size();
    }

    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int totalSendCount = 0, totalRecvCount = 0;
    for (int i = 0; i < size; ++i)
    {
        sendDispls[i] = totalSendCount;
        totalSendCount += sendCounts[i];
        recvDispls[i] = totalRecvCount;
        totalRecvCount += recvCounts[i];
    }

    std::vector<double> sendBuffer(totalSendCount), recvBuffer(totalRecvCount);
    for (int dest = 0; dest < size; ++dest)
    {
        std::copy(packedSendBuffers[dest].begin(), packedSendBuffers[dest].end(), sendBuffer.begin() + sendDispls[dest]);
    }

    MPI_Alltoallv(sendBuffer.data(), sendCounts.data(), sendDispls.data(), MPI_DOUBLE, recvBuffer.data(), recvCounts.data(), recvDispls.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    // Unpack received data
    std::vector<BlockInfo> receivedBlocks = unpackBlockInfo(recvBuffer);

    // Reorder received blocks based on globalOrder
    std::sort(receivedBlocks.begin(), receivedBlocks.end(), [](const BlockInfo& a, const BlockInfo& b) {
        return a.globalOrder < b.globalOrder;
    });

    // Perform nearest neighbor search
    #pragma omp parallel for
    for (size_t i = 0; i < blockInfos.size(); ++i) {
        auto& block = blockInfos[i];
        // tuple: distance, point, observation
        std::vector<std::tuple<double, std::array<double, DIMENSION>, double>> distancesMeta;

        for (auto& prevBlock: receivedBlocks) {
            if (prevBlock.globalOrder >= block.globalOrder){
                break;
            }
            for (int j = 0; j < prevBlock.points.size(); ++j) {
                std::pair<double, double> point = {prevBlock.points[j][0], prevBlock.points[j][1]};
                double distance = calculateDistance(block.center, point);
                distancesMeta.emplace_back(distance, prevBlock.points[j], prevBlock.observations_points[j]);
            }
        }

        // Sort distances and keep the m nearest neighbors
        std::sort(distancesMeta.begin(), distancesMeta.end(), [](const auto& a, const auto& b) {
            return std::get<0>(a) < std::get<0>(b);
        });
        for (auto k = 0; k < std::min(m, static_cast<int>(distancesMeta.size())); ++k) {
            block.nearestNeighbors.push_back(std::get<1>(distancesMeta[k]));
            block.observations_nearestNeighbors.push_back(std::get<2>(distancesMeta[k]));
        }
    }

    // Debug Save block information for visualization
    if (rank == 3) { // Choose the processor you want to visualize, here 2 is used as an example
        std::ofstream outfile("block_info.txt");
        if (outfile.is_open()) {
            for (const auto& block : blockInfos) {
                outfile << "Block Global Order: " << block.globalOrder << "\n";
                outfile << "Points and observations:\n";
                for (int i = 0; i < block.points.size(); ++i) {
                    outfile << "(" << block.points[i][0] << ", " << block.points[i][1] << ", " << block.observations_points[i] << ")\n";
                }
                outfile << "Block Center: (" << block.center.first << ", " << block.center.second << ")\n";
                outfile << "Nearest Neighbors and observations:\n";
                for (int i = 0; i < block.nearestNeighbors.size(); ++i) {
                    outfile << "(" << block.nearestNeighbors[i][0] << ", " << block.nearestNeighbors[i][1] << ", " << block.observations_nearestNeighbors[i] << ")\n";
                }
                outfile << "\n";
            }
            outfile.close();
        } else {
            std::cerr << "Unable to open file for writing.\n";
        }
    }
}
