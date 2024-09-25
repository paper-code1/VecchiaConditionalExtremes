#ifndef DISTANCE_CALC_H
#define DISTANCE_CALC_H

#include <vector>
#include <utility>
#include "block_info.h"

// Function to process and send blocks based on distance threshold and special rule for the first 100 blocks
void processAndSendBlocks(std::vector<BlockInfo>& blockInfos, const std::vector<std::pair<double, double>>& allCenters, int m, double distance_threshold);

#endif // DISTANCE_CALC_H
