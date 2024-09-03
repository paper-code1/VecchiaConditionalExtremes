#ifndef block_info_H
#define block_info_H

#include <vector>
#include <utility>
#include <cmath>
#include <map>
#include <set>

// Define a distance threshold
constexpr double DISTANCE_THRESHOLD = 0.5;

// Function to calculate the distance between two points
double calculateDistance(const std::pair<double, double>& point1, const std::pair<double, double>& point2);

// Function to build the one-to-one mapping of local centers and the corresponding finer partitions
void buildMapping(const std::vector<std::pair<double, double>>& localCenters, 
                  const std::vector<std::pair<double, double>>& allCenters, 
                  const std::vector<std::vector<std::pair<double, double>>>& finerPartitions, 
                  std::map<int, int>& localToGlobalMap, 
                  std::map<int, int>& globalToLocalMap);

// Function to prepare the candidate pool for all-to-all communication
void prepareCandidatePool(const std::vector<std::pair<double, double>>& localCenters, 
                          const std::vector<std::pair<double, double>>& allCenters, 
                          const std::map<int, int>& localToGlobalMap, 
                          std::set<int>& pool, 
                          int m);

// Function for all-to-all communication to share the candidate pool
void shareCandidatePool(const std::set<int>& pool, 
                        std::vector<std::vector<std::pair<double, double>>>& receivedBlocks);

// Function to find m nearest points for local centers
void findNearestNeighbors(const std::vector<std::pair<double, double>>& localCenters, 
                          const std::vector<std::vector<std::pair<double, double>>>& allBlocks, 
                          const std::map<int, int>& localToGlobalMap, 
                          int m, 
                          std::vector<std::vector<std::pair<double, double>>>& nearestNeighbors);

#endif // block_info_H
