#ifndef DISTANCE_CALC_H
#define DISTANCE_CALC_H

#include <vector>
#include <utility>
#include "block_info.h"

// Function to process and send blocks based on distance threshold and special rule for the first 100 blocks
std::vector<BlockInfo> processAndSendBlocks(std::vector<BlockInfo>& blockInfos, const std::vector<std::pair<double, double>>& allCenters, int m, double distance_threshold);

void nearest_neighbor_search(std::vector<BlockInfo> &blockInfos, int m, std::vector<BlockInfo> &receivedBlocks);

#endif // DISTANCE_CALC_H