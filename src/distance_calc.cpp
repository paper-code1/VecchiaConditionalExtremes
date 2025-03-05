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
        for (const auto &coord : block.center)
        {
            buffer.push_back(coord);
        }
        buffer.push_back(static_cast<double>(block.blocks.size()));
        for (const auto &point : block.blocks)
        {
            for (const auto &coord : point)
            {
                buffer.push_back(coord);
            }
        }
        for (const auto &obs: block.observations_blocks)
        {
            buffer.push_back(obs);
        }
    }
}

// Function to unpack block information from buffers
std::vector<BlockInfo> unpackBlockInfo(const std::vector<double> &buffer, const Opts& opts)
{
    std::vector<BlockInfo> blocks;
    size_t i = 0;
    while (i < buffer.size())
    {
        BlockInfo block;
        block.localOrder = static_cast<int>(buffer[i++]);
        block.globalOrder = static_cast<int>(buffer[i++]);
        block.center.resize(opts.dim);
        for (int j = 0; j < opts.dim; ++j)
        {
            block.center[j] = buffer[i++];
        }
        int numPoints = static_cast<int>(buffer[i++]);
        block.blocks.resize(numPoints, std::vector<double>(opts.dim));
        block.observations_blocks.resize(numPoints);
        for (int j = 0; j < numPoints; ++j)
        {
            for (int k = 0; k < opts.dim; ++k)
            {
                block.blocks[j][k] = buffer[i++];
            }
        }
        for (int j = 0; j < numPoints; ++j)
        {
            block.observations_blocks[j] = buffer[i++];
        }
        blocks.push_back(block);
    }
    return blocks;
}

std::vector<BlockInfo> processAndSendBlocks(std::vector<BlockInfo> &blockInfos, const std::vector<std::vector<double>> &allCenters, int m, double distance_threshold, const Opts& opts)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numBlocks = allCenters.size();

    // Prepare buffers to send blocks to other processors
    std::vector<std::set<int>> blockIndexSets(size);
    std::vector<std::vector<BlockInfo>> sendBuffers(size);

    int m_const = (opts.numBlocksTotal == opts.numPointsTotal) ? 300 : 42;

    for (const auto &blockInfo : blockInfos)
    {
        int globalOrder = blockInfo.globalOrder;

        // Send first 10 blocks to all processors
        if (globalOrder < m_const)
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
                int dest = std::min(static_cast<int>(allCenters[j][0] * size), size - 1);
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
    std::vector<BlockInfo> receivedBlocks = unpackBlockInfo(recvBuffer, opts);

    return receivedBlocks;
}


void nearest_neighbor_search(std::vector<BlockInfo> &blockInfos, std::vector<BlockInfo> &receivedBlocks, const Opts& opts, bool pred_tag)
{
    // Reorder received blocks based on globalOrder
    std::sort(receivedBlocks.begin(), receivedBlocks.end(), [](const BlockInfo& a, const BlockInfo& b) {
        return a.globalOrder < b.globalOrder;
    });
    int m_nn = (pred_tag) ? opts.m_test : opts.m;
    // double distance_threshold = (pred_tag) ? opts.distance_threshold * 10 : opts.distance_threshold;
    double distance_threshold = opts.distance_threshold;
    
    // Perform nearest neighbor search
    #pragma omp parallel for
    for (size_t i = 0; i < blockInfos.size(); ++i) {
        auto& block = blockInfos[i];
        // tuple: distance, point, observation
        std::vector<std::tuple<double, std::vector<double>, double>> distancesMeta;

        for (auto& prevBlock: receivedBlocks) {
            if (prevBlock.globalOrder >= block.globalOrder){
                break;
            }
            for (size_t j = 0; j < prevBlock.blocks.size(); ++j) {
                double distance = calculateDistance(block.center, prevBlock.blocks[j]);
                // Add all points if the block order is small
                if (distance < distance_threshold || block.globalOrder <= 700){
                    distancesMeta.emplace_back(distance, prevBlock.blocks[j], prevBlock.observations_blocks[j]);
                }
            }
        }
        if (distancesMeta.size() < m_nn && block.globalOrder > 0){
            std::cout << "Warning: Not enough neighbors found for block, random added. " << "m: " << distancesMeta.size() << ", block: " << block.globalOrder << std::endl;
            for (auto& prevBlock: receivedBlocks) {
                if (prevBlock.globalOrder >= block.globalOrder) {
                    break;
                }
                for (size_t j = 0; j < prevBlock.blocks.size(); ++j) {
                    // Skip points we've already added
                    bool already_added = false;
                    for (const auto& existing : distancesMeta) {
                        if (std::get<1>(existing) == prevBlock.blocks[j]) {
                            already_added = true;
                            break;
                        }
                    }
                    if (!already_added) {
                        double distance = calculateDistance(block.center, prevBlock.blocks[j]);
                        distancesMeta.emplace_back(distance, prevBlock.blocks[j], prevBlock.observations_blocks[j]);
                        if (distancesMeta.size() >= m_nn) {
                            break;
                        }
                    }
                }
                if (distancesMeta.size() >= m_nn) {
                    break;
                }
            }
        }
        
        // Sort distances and keep the m nearest neighbors
        std::sort(distancesMeta.begin(), distancesMeta.end(), [](const auto& a, const auto& b) {
            return std::get<0>(a) < std::get<0>(b);
        });
        for (auto k = 0; k < std::min(m_nn, static_cast<int>(distancesMeta.size())); ++k) {
            block.nearestNeighbors.push_back(std::get<1>(distancesMeta[k]));
            block.observations_nearestNeighbors.push_back(std::get<2>(distancesMeta[k]));
        }
    }
}

void distanceScale(std::vector<PointMetadata> &localPoints, const std::vector<double>& scale_factor, int dim){
    #pragma omp parallel for
    for (auto& point: localPoints){
        for (int i = 0; i < dim; ++i){
            point.coordinates[i] = point.coordinates[i] / scale_factor[i];
        }
    }
}

void distanceDeScale(std::vector<BlockInfo> &localBlocks, const std::vector<double>& scale_factor, int dim){
    #pragma omp parallel for
    for (auto& block: localBlocks){
        for (auto& point: block.blocks){
            for (int i = 0; i < dim; ++i){
                point[i] = point[i] * scale_factor[i];
            }
        }
    }
    #pragma omp parallel for
    for (auto& block: localBlocks){
        for (auto& point: block.nearestNeighbors){
            for (int i = 0; i < dim; ++i){
                point[i] = point[i] * scale_factor[i];
            }
        }
    }
}

// Add this function before main()
double calculate_distance_threshold(const std::vector<double>& distance_scale, int numBlocksPerProcess, int numPointsTotal, int m, int dim_process, int nn_multiplier) {
    // add a factor to account for the non-uniform distirbution
    int nn_m = m * nn_multiplier;
    // Count dimensions with distance_scale > 1
    int dim = 0;
    double thres_active = 1.0;
    for (double scale : distance_scale) {
        if (scale < thres_active) {
            dim++;
        }
    }
    // Ensure dim is at least 1 to avoid division by zero
    dim = std::max(1, dim);
    // Calculate the volume of the space defined by distance_scale
    double space_volume = 1.0;
    for (double scale : distance_scale) {
        if (scale < thres_active) {
            space_volume /= scale;
        }
    }
    
    // Calculate point density (points per unit volume)
    double point_density = numPointsTotal / space_volume;
    
    // Volume of n-dimensional unit ball (hypersphere)
    double unit_ball_volume;
    if (dim % 2 == 0) {
        // Even dimensions
        unit_ball_volume = pow(M_PI, dim/2) / std::tgamma(dim/2 + 1);
    } else {
        // Odd dimensions
        unit_ball_volume = 2 * pow(M_PI, (dim-1)/2) * std::tgamma((dim+1)/2) / std::tgamma(dim+1);
    }
    
    // Calculate required radius to contain nn_m points on average
    // V = pi^(d/2) * r^d / Gamma(d/2 + 1)
    // Solve for r: r = (V * Gamma(d/2 + 1) / pi^(d/2))^(1/d)
    double required_volume = nn_m / point_density;
    double radius = pow(required_volume / unit_ball_volume, 1.0/dim);
    
    // Add a safety factor (e.g., 1.2) to account for non-uniform distribution
    return radius * 1.2;
}
