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

std::vector<BlockInfo> processAndSendBlocks(std::vector<BlockInfo> &blockInfos, 
    const std::vector<std::pair<std::vector<double>, int>> &CenterRanks, 
    const std::vector<std::pair<std::vector<double>, int>> &CenterRanks_test, 
    double distance_threshold, 
    const std::vector<int>& permutation, 
    const std::vector<int>& localPermutation, 
    const Opts& opts, 
    bool pred_tag)
{
    // returned: receivedBlocks, which is within the MC-NN based range
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Prepare buffers to send blocks to other processors
    std::vector<std::set<int>> blockIndexSets(size);
    std::vector<std::vector<BlockInfo>> sendBuffers(size);
    double distance_threshold_dynamic = (pred_tag) ? distance_threshold : distance_threshold/opts.dim;

    // 42 is a magic number, I like it
    int m_const = (opts.numBlocksTotal == opts.numPointsTotal) ? 300 : 42;
    std::vector<std::pair<std::vector<double>, int>> allCenterRanks;
    // Combine CenterRanks and CenterRanks_test
    if (pred_tag){
        allCenterRanks = CenterRanks_test;
    } else {
        allCenterRanks = CenterRanks;
    }

    // Using OpenMP with private copies of blockIndexSets and sendBuffers to avoid race conditions
    std::vector<std::vector<std::set<int>>> private_blockIndexSets(omp_get_max_threads(), std::vector<std::set<int>>(size));
    std::vector<std::vector<std::vector<BlockInfo>>> private_sendBuffers(omp_get_max_threads(), std::vector<std::vector<BlockInfo>>(size));

    // Parallelize loop over centers with a chunking strategy
    if (pred_tag){
        #pragma omp parallel for schedule(dynamic)
        for (int i=0; i<allCenterRanks.size(); ++i){
            int thread_id = omp_get_thread_num();
            const auto &centerRank = allCenterRanks[i];
            const auto &center = centerRank.first;
            int destRank = centerRank.second;
            
            // For each block, check if it's within distance threshold of this center
            for (const auto &blockInfo : blockInfos) {
                int globalOrder = blockInfo.globalOrder;
                // Calculate distance between block center and the current center
                double distance = calculateDistance(blockInfo.center, center);
                // If within threshold, send to the corresponding rank
                if (distance < distance_threshold_dynamic) {
                    if (private_blockIndexSets[thread_id][destRank].find(globalOrder) == private_blockIndexSets[thread_id][destRank].end()) {
                        private_sendBuffers[thread_id][destRank].push_back(blockInfo);
                        private_blockIndexSets[thread_id][destRank].insert(globalOrder);
                    }
                }
            }
        }
    }else{
        #pragma omp parallel for schedule(dynamic)
        for (int i=0; i<allCenterRanks.size(); ++i){
            int thread_id = omp_get_thread_num();
            const auto &centerRank = allCenterRanks[i];
            const auto &center = centerRank.first;
            int destRank = centerRank.second;
            
            // For each block, check if it's within distance threshold of this center
            for (const auto &blockInfo : blockInfos) {
                int globalOrder = blockInfo.globalOrder;
                if (globalOrder >= permutation[i]){
                    continue;
                }

                // Send first m_const blocks to all processors to ensure enough blocks
                if (globalOrder < m_const) {
                    for (int dest = 0; dest < size; ++dest) {
                        if (private_blockIndexSets[thread_id][dest].find(globalOrder) == private_blockIndexSets[thread_id][dest].end()) {
                            private_sendBuffers[thread_id][dest].push_back(blockInfo);
                            private_blockIndexSets[thread_id][dest].insert(globalOrder);
                        }
                    }
                    continue;
                }
                
                // Calculate distance between block center and the current center
                double distance = calculateDistance(blockInfo.center, center);
                // If within threshold, send to the corresponding rank
                if (distance < distance_threshold_dynamic) {
                    if (private_blockIndexSets[thread_id][destRank].find(globalOrder) == private_blockIndexSets[thread_id][destRank].end()) {
                        private_sendBuffers[thread_id][destRank].push_back(blockInfo);
                        private_blockIndexSets[thread_id][destRank].insert(globalOrder);
                    }
                }
            }
        }
    }

    // Merge private data structures
    for (int t = 0; t < omp_get_max_threads(); t++) {
        for (int dest = 0; dest < size; dest++) {
            for (const auto& globalOrder : private_blockIndexSets[t][dest]) {
                if (blockIndexSets[dest].find(globalOrder) == blockIndexSets[dest].end()) {
                    for (const auto& block : private_sendBuffers[t][dest]) {
                        if (block.globalOrder == globalOrder) {
                            sendBuffers[dest].push_back(block);
                            blockIndexSets[dest].insert(globalOrder);
                            break;
                        }
                    }
                }
            }
        }
    }

    // Directly prepare data for MPI_Alltoallv without using pack/unpack functions
    // First, calculate the size of each buffer to send
    std::vector<int> sendCounts(size, 0);
    for (int dest = 0; dest < size; ++dest) {
        for (const auto& block : sendBuffers[dest]) {
            // Count elements for each block:
            // 2 for localOrder and globalOrder
            // opts.dim for center coordinates
            // 1 for number of points
            // numPoints * opts.dim for block coordinates
            // numPoints for observations
            int numPoints = block.blocks.size();
            sendCounts[dest] += 2 + opts.dim + 1 + (numPoints * opts.dim) + numPoints;
        }
    }

    // Exchange send counts to get receive counts
    std::vector<int> recvCounts(size);
    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Calculate displacements
    std::vector<int> sendDispls(size), recvDispls(size);
    int totalSendCount = 0, totalRecvCount = 0;
    for (int i = 0; i < size; ++i) {
        sendDispls[i] = totalSendCount;
        totalSendCount += sendCounts[i];
        recvDispls[i] = totalRecvCount;
        totalRecvCount += recvCounts[i];
    }

    // Prepare send buffer
    std::vector<double> sendBuffer(totalSendCount);
    int currentPos = 0;
    for (int dest = 0; dest < size; ++dest) {
        for (const auto& block : sendBuffers[dest]) {
            // Add localOrder and globalOrder
            sendBuffer[currentPos++] = static_cast<double>(block.localOrder);
            sendBuffer[currentPos++] = static_cast<double>(block.globalOrder);
            
            // Add center coordinates
            for (const auto& coord : block.center) {
                sendBuffer[currentPos++] = coord;
            }
            
            // Add number of points
            sendBuffer[currentPos++] = static_cast<double>(block.blocks.size());
            
            // Add block coordinates
            for (const auto& point : block.blocks) {
                for (const auto& coord : point) {
                    sendBuffer[currentPos++] = coord;
                }
            }
            
            // Add observations
            for (const auto& obs : block.observations_blocks) {
                sendBuffer[currentPos++] = obs;
            }
        }
    }

    // Prepare receive buffer
    std::vector<double> recvBuffer(totalRecvCount);

    // Perform the all-to-all communication
    MPI_Alltoallv(sendBuffer.data(), sendCounts.data(), sendDispls.data(), MPI_DOUBLE,
                 recvBuffer.data(), recvCounts.data(), recvDispls.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    // Unpack received data directly
    std::vector<BlockInfo> receivedBlocks;
    size_t i = 0;
    while (i < recvBuffer.size()) {
        BlockInfo block;
        block.localOrder = static_cast<int>(recvBuffer[i++]);
        block.globalOrder = static_cast<int>(recvBuffer[i++]);
        
        // Unpack center coordinates
        block.center.resize(opts.dim);
        for (int j = 0; j < opts.dim; ++j) {
            block.center[j] = recvBuffer[i++];
        }
        
        // Unpack number of points
        int numPoints = static_cast<int>(recvBuffer[i++]);
        
        // Unpack block coordinates
        block.blocks.resize(numPoints, std::vector<double>(opts.dim));
        for (int j = 0; j < numPoints; ++j) {
            for (int k = 0; k < opts.dim; ++k) {
                block.blocks[j][k] = recvBuffer[i++];
            }
        }
        
        // Unpack observations
        block.observations_blocks.resize(numPoints);
        for (int j = 0; j < numPoints; ++j) {
            block.observations_blocks[j] = recvBuffer[i++];
        }
        
        receivedBlocks.push_back(block);
    }

    return receivedBlocks;
}


void nearest_neighbor_search(std::vector<BlockInfo> &blockInfos, std::vector<BlockInfo> &receivedBlocks, const Opts& opts, bool pred_tag)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // Reorder received blocks based on globalOrder
    std::sort(receivedBlocks.begin(), receivedBlocks.end(), [](const BlockInfo& a, const BlockInfo& b) {
        return a.globalOrder < b.globalOrder;
    });
    int m_nn = (pred_tag) ? opts.m_test : opts.m;
    double distance_threshold = opts.distance_threshold_finer;
    
    // Perform nearest neighbor search
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < blockInfos.size(); ++i) {
        auto& block = blockInfos[i];
        // tuple: distance, point, observation
        std::vector<std::tuple<double, std::vector<double>, double>> distancesMeta;

        for (auto& prevBlock : receivedBlocks) {
            if (prevBlock.globalOrder >= block.globalOrder){
                break;
            }
            for (size_t j = 0; j < prevBlock.blocks.size(); ++j) {
                double distance = calculateDistance(block.center, prevBlock.blocks[j]);
                // Add all points if the block order is within the range
                if (distance < distance_threshold || block.globalOrder <= 200){
                    distancesMeta.emplace_back(distance, prevBlock.blocks[j], prevBlock.observations_blocks[j]);
                }
            }
        }
        // add cases for the classic Vecchia
        if (block.globalOrder <= m_nn && opts.numBlocksPerProcess == opts.numPointsPerProcess){
            continue;
        }
        if (distancesMeta.size() < m_nn && block.globalOrder > 0){
            std::cout << "Warning: Not enough neighbors found for block, random added. " << "m: " << distancesMeta.size() << ", block: " << block.globalOrder << ", rank: " << rank << std::endl;
            for (auto& prevBlock : receivedBlocks) {
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
    for (int i = 0; i < localPoints.size(); ++i){
        for (int j = 0; j < dim; ++j){
            localPoints[i].coordinates[j] = localPoints[i].coordinates[j] / scale_factor[j];
        }
    }
}

void distanceDeScale(std::vector<BlockInfo> &localBlocks, const std::vector<double>& scale_factor, int dim){
    #pragma omp parallel for
    for (int i = 0; i < localBlocks.size(); ++i){
        for (int j = 0; j < localBlocks[i].blocks.size(); ++j){
            for (int k = 0; k < dim; ++k){
                localBlocks[i].blocks[j][k] = localBlocks[i].blocks[j][k] * scale_factor[k];
            }
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < localBlocks.size(); ++i){
        for (int j = 0; j < localBlocks[i].nearestNeighbors.size(); ++j){
            for (int k = 0; k < dim; ++k){
                localBlocks[i].nearestNeighbors[j][k] = localBlocks[i].nearestNeighbors[j][k] * scale_factor[k];
            }
        }
    }
}

// Add this function before main()
double calculate_distance_threshold(const std::vector<double>& distance_scale, int numPointsTotal, int m, int nn_multiplier) {
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
