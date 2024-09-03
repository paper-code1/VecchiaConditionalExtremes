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
            buffer.push_back(point.first);
            buffer.push_back(point.second);
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
        for (int j = 0; j < numPoints; ++j)
        {
            x = buffer[i++];
            y = buffer[i++];
            block.points[j] = {x, y};
        }
        blocks.push_back(block);
    }
    return blocks;
}

void processAndSendBlocks(std::vector<BlockInfo> &blockInfos, const std::vector<std::pair<double, double>> &allCenters, int m)
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

        // Send first 100 blocks to all processors
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
            if (distance < DISTANCE_THRESHOLD)
            {
                // int dest = j % size; // Determine the target processor based on the global order           
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

    // Save data for the 3rd processor
    // std::cout << "Rank " << rank << ": the number of received blocks is : " << receivedBlocks.size() << std::endl;
    // if (rank == 10)
    // {
    //     std::ofstream outfile("processor_data.txt");
    //     if (outfile.is_open())
    //     {
    //         for (const auto &blockInfo : receivedBlocks)
    //         {
    //             outfile << blockInfo.localOrder << " " << blockInfo.globalOrder << " "
    //                     << blockInfo.center.first << " " << blockInfo.center.second << " "
    //                     << blockInfo.points.size() << "\n";
    //             for (const auto &point : blockInfo.points)
    //             {
    //                 outfile << point.first << " " << point.second << "\n";
    //             }
    //         }
    //         outfile.close();
    //     }
    //     else
    //     {
    //         std::cerr << "Unable to open file for writing.\n";
    //     }
    //     std::ofstream outfilea("processor_data_blockinfo.txt");
    //     if (outfilea.is_open())
    //     {
    //         for (const auto &blockInfo : blockInfos)
    //         {
    //             outfilea << blockInfo.localOrder << " " << blockInfo.globalOrder << " "
    //                     << blockInfo.center.first << " " << blockInfo.center.second << " "
    //                     << blockInfo.points.size() << "\n";
    //             for (const auto &point : blockInfo.points)
    //             {
    //                 outfilea << point.first << " " << point.second << "\n";
    //             }
    //         }
    //         outfilea.close();
    //     }
    //     else
    //     {
    //         std::cerr << "Unable to open file for writing.\n";
    //     }
    // }

    // Reorder received blocks based on globalOrder
    std::sort(receivedBlocks.begin(), receivedBlocks.end(), [](const BlockInfo& a, const BlockInfo& b) {
        return a.globalOrder < b.globalOrder;
    });

    // Perform nearest neighbor search
    #pragma omp parallel for
    for (size_t i = 0; i < blockInfos.size(); ++i) {
        auto& block = blockInfos[i];
        std::vector<std::pair<double, std::pair<double, double>>> distances;

        for (auto& prevBlock: receivedBlocks) {
            if (prevBlock.globalOrder >= block.globalOrder){
                break;
            }
            for (const auto& point : prevBlock.points) {
                double distance = calculateDistance(block.center, point);
                distances.emplace_back(distance, point);
            }
        }

        // Sort distances and keep the m nearest neighbors
        std::sort(distances.begin(), distances.end());
        for (auto k = 0; k < std::min(m, static_cast<int>(distances.size())); ++k) {
            block.nearestNeighbors.push_back(distances[k].second);
        }
    }

    // Debug Save block information for visualization
    if (rank == 1) { // Choose the processor you want to visualize, here 2 is used as an example
        std::ofstream outfile("block_info.txt");
        if (outfile.is_open()) {
            for (const auto& block : blockInfos) {
                outfile << "Block Global Order: " << block.globalOrder << "\n";
                outfile << "Points:\n";
                for (const auto& point : block.points) {
                    outfile << "(" << point.first << ", " << point.second << ")\n";
                }
                outfile << "Block Center: (" << block.center.first << ", " << block.center.second << ")\n";
                outfile << "Nearest Neighbors:\n";
                for (const auto& neighbor : block.nearestNeighbors) {
                    outfile << "(" << neighbor.first << ", " << neighbor.second << ")\n";
                }
                outfile << "\n";
            }
            outfile.close();
        } else {
            std::cerr << "Unable to open file for writing.\n";
        }
    }
}
