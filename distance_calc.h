#ifndef DISTANCE_CALC_H
#define DISTANCE_CALC_H

#include <vector>
#include <utility>
#include "block_info.h"

#define DISTANCE_THRESHOLD 0.2  // You can adjust this value as needed

// Function to process and send blocks based on distance threshold and special rule for the first 100 blocks
void processAndSendBlocks(std::vector<BlockInfo>& blockInfos, const std::vector<std::pair<double, double>>& allCenters, int m);

#endif // DISTANCE_CALC_H
