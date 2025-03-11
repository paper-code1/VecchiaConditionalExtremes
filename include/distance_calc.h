#ifndef DISTANCE_CALC_H
#define DISTANCE_CALC_H

#include <vector>
#include <utility>
#include "block_info.h"

// Function to process and send blocks based on distance threshold and special rule for the first 100 blocks
std::vector<BlockInfo> processAndSendBlocks(std::vector<BlockInfo>& blockInfos, const std::vector<std::pair<std::vector<double>, int>>& allCenters, const std::vector<std::pair<std::vector<double>, int>>& allCenters_test, double distance_threshold, const Opts& opts);

void nearest_neighbor_search(std::vector<BlockInfo> &blockInfos, std::vector<BlockInfo> &receivedBlocks, const Opts& opts, bool pred_tag);

void distanceScale(std::vector<PointMetadata> &localPoints, const std::vector<double>& scale_factor, int dim);

void distanceDeScale(std::vector<BlockInfo> &localPoints, const std::vector<double>& scale_factor, int dim);

double calculate_distance_threshold(const std::vector<double>& distance_scale, int numBlocksPerProcess, int numPointsTotal, int m, int dim, int nn_multiplier);

#endif // DISTANCE_CALC_H