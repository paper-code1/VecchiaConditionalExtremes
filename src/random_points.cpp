#include <mpi.h>
#include <iostream>
#include <numeric>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <random>
#include <limits>
#include <omp.h>
#include <fstream> // Add this line
#include <sstream> // Add this line for std::istringstream
#include "random_points.h"

// Define custom reduction for 2D vector
#pragma omp declare reduction(vec2d_double_plus : std::vector<std::vector<double>> : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(),                           \
                                                                                                        [](std::vector<double> & a, const std::vector<double> &b){                             \
                                                                                                                std::transform(a.begin(), a.end(), b.begin(), a.begin(), std::plus<double>()); \
                                                                                                                    return a;}))                                                               \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size(), std::vector<double>(omp_orig[0].size())))

// Define custom reduction for 1D vector
#pragma omp declare reduction(vec_int_plus : std::vector<int> : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<int>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

// Function to generate random double between 0 and 1
double generateRandomDouble()
{
    return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}

// Function to generate random points
std::vector<PointMetadata> generateRandomPoints(int numPointsPerProcess, const Opts &opts)
{
    std::vector<PointMetadata> pointsMetadata(numPointsPerProcess);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // fix the random seed for each process
    std::srand(rank + 1);

    for (int i = 0; i < numPointsPerProcess; ++i)
    {
        pointsMetadata[i].coordinates.resize(opts.dim);
        for (int j = 0; j < opts.dim; ++j)
        {
            pointsMetadata[i].coordinates[j] = generateRandomDouble();
        }
        pointsMetadata[i].observation = generateRandomDouble();
        // pointsMetadata[i].observation = 0.0;
    }

    return pointsMetadata;
}

// Function to partition points and communicate them to the appropriate processors
void partitionPoints(const std::vector<PointMetadata> &localMetadata, std::vector<PointMetadata> &localMetadata_block, const Opts &opts)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Prepare to send data to the appropriate process based on x value
    std::vector<int> sendCounts(size, 0);
    std::vector<int> sendDisplacements(size, 0);

    std::vector<std::vector<PointMetadata>> sendBuffers(size);

    for (const auto &pointmeta : localMetadata)
    {
        int targetProcess = std::min(static_cast<int>(pointmeta.coordinates[0] * size), size - 1);
        sendBuffers[targetProcess].push_back(pointmeta);
    }

    for (int i = 0; i < size; ++i)
    {
        sendCounts[i] = sendBuffers[i].size() * (opts.dim + 1); // Each point has DIMENSION + 1 doubles
    }

    std::vector<double> sendData;
    for (int i = 0; i < size; ++i)
    {
        for (const auto &point : sendBuffers[i])
        {
            for (int j = 0; j < opts.dim; ++j)
            {
                sendData.push_back(point.coordinates[j]);
            }
            sendData.push_back(point.observation);
        }
    }

    // Calculate displacements
    for (int i = 1; i < size; ++i)
    {
        sendDisplacements[i] = sendDisplacements[i - 1] + sendCounts[i - 1];
    }

    // Prepare receive counts and displacements
    std::vector<int> recvCounts(size, 0);
    std::vector<int> recvDisplacements(size, 0);

    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    for (int i = 1; i < size; ++i)
    {
        recvDisplacements[i] = recvDisplacements[i - 1] + recvCounts[i - 1];
    }

    int totalRecvCount = std::accumulate(recvCounts.begin(), recvCounts.end(), 0);
    std::vector<double> recvData(totalRecvCount);

    MPI_Alltoallv(sendData.data(), sendCounts.data(), sendDisplacements.data(), MPI_DOUBLE,
                  recvData.data(), recvCounts.data(), recvDisplacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    // Convert received data back to PointMetadata
    localMetadata_block.clear();
    localMetadata_block.resize(totalRecvCount / (opts.dim + 1));
    for (int i = 0, index = 0; i < totalRecvCount; i += (opts.dim + 1), ++index)
    {
        localMetadata_block[index].coordinates.resize(opts.dim);
        for (int j = 0; j < opts.dim; ++j)
        {
            localMetadata_block[index].coordinates[j] = recvData[i + j];
        }
        localMetadata_block[index].observation = recvData[i + opts.dim];
    }
}

// Function to perform random clustering
std::vector<int> randomClustering(const std::vector<PointMetadata> &metadata, int k, int dim, int seed)
{
    int numPoints = metadata.size();
    std::vector<int> clusters(numPoints);

    // Initialize random number generator
    std::mt19937 gen(seed);

    // 1. Randomly select k centers without replacement
    std::vector<std::vector<double>> centers(k);
    std::vector<int> centerIndices(numPoints);
    std::iota(centerIndices.begin(), centerIndices.end(), 0);

    // Shuffle and take first k indices
    std::shuffle(centerIndices.begin(), centerIndices.end(), gen);
    for (int i = 0; i < k; ++i)
    {
        centers[i] = metadata[centerIndices[i]].coordinates;
        clusters[centerIndices[i]] = i;
    }

// 2. Assign remaining points to nearest center
#pragma omp parallel for schedule(static)
    for (int i = 0; i < numPoints; ++i)
    {
        // Skip if this point is a center
        if (i < k && i == centerIndices[i])
            continue;

        double minDist = std::numeric_limits<double>::max();
        int nearestCluster = 0;

        // Find nearest center
        for (int j = 0; j < k; ++j)
        {
            double dist = 0.0;
            for (int d = 0; d < dim; ++d)
            {
                double diff = metadata[i].coordinates[d] - centers[j][d];
                dist += diff * diff;
            }

            if (dist < minDist)
            {
                minDist = dist;
                nearestCluster = j;
            }
        }

        clusters[i] = nearestCluster;
    }

    return clusters;
}

// Function to perform finer partitioning within each processor using k-means++
void finerPartition(const std::vector<PointMetadata> &metadata, int numBlocksPerProcess,
                    std::vector<std::vector<PointMetadata>> &finerPartitions, const Opts &opts)
{
    finerPartitions.clear();
    finerPartitions.resize(numBlocksPerProcess);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Perform clustering
    std::vector<int> clusters;
    if (numBlocksPerProcess * 3 < metadata.size())
    {
        // block Vecchia
        if (opts.clustering == "random")
        {
            // Use random clustering for large datasets
            clusters = randomClustering(metadata, numBlocksPerProcess, opts.dim, opts.seed + rank);
        }
        else if (opts.clustering == "kmeans++")
        {
            // Use k-means++ for smaller datasets
            clusters = kMeansPlusPlus(metadata, numBlocksPerProcess, opts.dim, opts.kmeans_max_iter, rank, opts.seed);
        }
        else
        {
            std::cerr << "Invalid clustering method: " << opts.clustering << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    else
    {
        // classic Vecchia
        clusters.resize(metadata.size());
        std::iota(clusters.begin(), clusters.end(), 0);
    }

    // Assign points to clusters
    for (size_t i = 0; i < metadata.size(); ++i)
    {
        finerPartitions[clusters[i]].push_back(metadata[i]);
    }
}

// Function to calculate centers of gravity for each block with OpenMP parallelization
std::vector<std::vector<double>> calculateCentersOfGravity(const std::vector<std::vector<PointMetadata>> &finerPartitions, const Opts &opts)
{
    int numBlocks = finerPartitions.size();
    std::vector<std::vector<double>> centers(numBlocks, std::vector<double>(opts.dim));

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < numBlocks; ++i)
    {
        auto &blockmetadata = finerPartitions[i];
        if (blockmetadata.empty())
        {
            continue;
        }
        std::vector<double> sum(opts.dim, 0.0);
        for (auto &pointmeta : blockmetadata)
        {
            for (int j = 0; j < opts.dim; ++j)
            {
                sum[j] += pointmeta.coordinates[j];
            }
        }
        for (int j = 0; j < opts.dim; ++j)
        {
            centers[i][j] = sum[j] / blockmetadata.size();
        }
    }

    return centers;
}

// Function to send centers of gravity to processor 0
void sendCentersOfGravityToRoot(const std::vector<std::vector<double>> &centers, std::vector<std::pair<std::vector<double>, int>> &allCenters, const Opts &opts)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numCenters = centers.size();
    std::vector<int> recvCounts(size, 0);
    MPI_Gather(&numCenters, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displacements(size, 0);
    int totalCenters = 0;
    if (rank == 0)
    {
        for (int i = 0; i < size; ++i)
        {
            displacements[i] = totalCenters * (opts.dim + 1); // +1 for rank
            totalCenters += recvCounts[i];
            recvCounts[i] *= (opts.dim + 1); // Each center has DIMENSION doubles + rank
        }
    }

    // Create send buffer with center coordinates and rank
    std::vector<double> sendBuffer(numCenters * (opts.dim + 1));
    for (int i = 0; i < numCenters; ++i)
    {
        for (int j = 0; j < opts.dim; ++j)
        {
            sendBuffer[i * (opts.dim + 1) + j] = centers[i][j];
        }
        // Add rank information
        sendBuffer[i * (opts.dim + 1) + opts.dim] = static_cast<double>(rank);
    }

    std::vector<double> recvBuffer;
    if (rank == 0)
    {
        recvBuffer.resize(totalCenters * (opts.dim + 1));
    }
    MPI_Gatherv(sendBuffer.data(), numCenters * (opts.dim + 1), MPI_DOUBLE, 
                recvBuffer.data(), recvCounts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        allCenters.clear();
        allCenters.resize(totalCenters);
        
        for (int i = 0; i < totalCenters; ++i)
        {
            std::vector<double> centerCoords(opts.dim);
            for (int j = 0; j < opts.dim; ++j)
            {
                centerCoords[j] = recvBuffer[i * (opts.dim + 1) + j];
            }
            int centerRank = static_cast<int>(recvBuffer[i * (opts.dim + 1) + opts.dim]);
            allCenters[i] = std::make_pair(centerCoords, centerRank);
        }
    }
}

// Function to randomly reorder centers at processor 0
void reorderCenters(std::vector<std::pair<std::vector<double>, int>> &centers, const Opts &opts)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        // Use a seeded random number generator for reproducibility
        std::mt19937 gen(opts.seed);
        // Shuffle non-empty centers using std::shuffle instead of deprecated std::random_shuffle
        std::shuffle(centers.begin(), centers.end(), gen);
    }
}

// Function to broadcast reordered centers to all processors
void broadcastCenters(std::vector<std::pair<std::vector<double>, int>> &allCenters, int numCenters, const Opts &opts)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Prepare data for broadcast
    std::vector<double> centersData(numCenters * (opts.dim + 1)); // +1 for rank information
    if (rank == 0)
    {
        for (int i = 0; i < numCenters; ++i)
        {
            for (int j = 0; j < opts.dim; ++j)
            {
                centersData[i * (opts.dim + 1) + j] = allCenters[i].first[j];
            }
            // Add rank information
            centersData[i * (opts.dim + 1) + opts.dim] = static_cast<double>(allCenters[i].second);
        }
    }

    // Broadcast the reordered centers to all processes
    MPI_Bcast(centersData.data(), numCenters * (opts.dim + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Resize the allCenters vector on all processes
    allCenters.resize(numCenters);

    // Convert data back to vector of pairs (coordinates, rank)
    for (int i = 0; i < numCenters; ++i)
    {
        std::vector<double> centerCoords(opts.dim);
        for (int j = 0; j < opts.dim; ++j)
        {
            centerCoords[j] = centersData[i * (opts.dim + 1) + j];
        }
        int centerRank = static_cast<int>(centersData[i * (opts.dim + 1) + opts.dim]);
        allCenters[i] = std::make_pair(centerCoords, centerRank);
    }
}

// Function to perform k-means++ clustering with iterations and parallelization
std::vector<int> kMeansPlusPlus(const std::vector<PointMetadata> &metadata, int k, int dim, int maxIterations, int rank, int seed)
{
    std::vector<int> clusters(metadata.size());
    std::vector<std::vector<double>> centroids(k, std::vector<double>(dim));
    std::mt19937 gen(seed + rank); // Use fixed seed + rank for reproducibility
    // Choose the first centroid randomly
    std::uniform_int_distribution<> dis(0, metadata.size() - 1);
    int firstCentroidIndex = dis(gen);
    centroids[0] = metadata[firstCentroidIndex].coordinates;

    // Choose the remaining centroids
    for (int i = 1; i < k; ++i)
    {
        std::vector<double> distances(metadata.size(), std::numeric_limits<double>::max());

#pragma omp parallel for
        for (size_t j = 0; j < metadata.size(); ++j)
        {
            for (int c = 0; c < i; ++c)
            {
                double dist = 0;
                for (int d = 0; d < dim; ++d)
                {
                    double diff = metadata[j].coordinates[d] - centroids[c][d];
                    dist += diff * diff;
                }
                distances[j] = std::min(distances[j], dist);
            }
        }

        // Choose the next centroid with probability proportional to distance squared
        std::discrete_distribution<> d(distances.begin(), distances.end());
        int nextCentroidIndex = d(gen);
        centroids[i] = metadata[nextCentroidIndex].coordinates;
    }

    // K-means iterations
    for (int iter = 0; iter < maxIterations; ++iter)
    {
        // print info for every 10 iterations
        if (iter % 30 == 0 && rank == 0)
        {
            std::cout << "K-means iteration: " << iter << std::endl;
        }
// Assign points to the nearest centroid
#pragma omp parallel for
        for (size_t i = 0; i < metadata.size(); ++i)
        {
            double minDist = std::numeric_limits<double>::max();
            int nearestCentroid = 0;
            for (int j = 0; j < k; ++j)
            {
                double dist = 0;
                for (int d = 0; d < dim; ++d)
                {
                    double diff = metadata[i].coordinates[d] - centroids[j][d];
                    dist += diff * diff;
                }
                if (dist < minDist)
                {
                    minDist = dist;
                    nearestCentroid = j;
                }
            }
            clusters[i] = nearestCentroid;
        }

        // Recalculate centroids
        std::vector<std::vector<double>> newCentroids(k, std::vector<double>(dim, 0.0));
        std::vector<int> clusterSizes(k, 0);

#pragma omp parallel for reduction(vec2d_double_plus : newCentroids) reduction(vec_int_plus : clusterSizes)
        for (size_t i = 0; i < metadata.size(); ++i)
        {
            int cluster = clusters[i];
            for (int d = 0; d < dim; ++d)
            {
                newCentroids[cluster][d] += metadata[i].coordinates[d];
            }
            clusterSizes[cluster]++;
        }

        // Update centroids
        for (int i = 0; i < k; ++i)
        {
            if (clusterSizes[i] > 0)
            {
                for (int d = 0; d < dim; ++d)
                {
                    centroids[i][d] = newCentroids[i][d] / clusterSizes[i];
                }
            }
            else
            {
                // Assign a random point as the centroid for empty clusters
                int randomIndex = dis(gen);
                centroids[i] = metadata[randomIndex].coordinates;
                // Update the cluster assignment for the randomly chosen point
                clusters[randomIndex] = i;
            }
        }
    }

    return clusters;
}

// Function to read points concurrently, with each processor reading a specific chunk of rows
std::vector<PointMetadata> readPointsConcurrently(const std::string &filename, int numBlocks, const Opts &opts)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numPointsTotal = numBlocks;

    // Calculate the number of rows each process should read
    int rows_per_process = numPointsTotal / size;
    int remainder = numPointsTotal % size;

    // Calculate the start and end rows for this process
    int start_row = rank * rows_per_process + std::min(rank, remainder);
    int end_row = start_row + rows_per_process + (rank < remainder ? 1 : 0);

    // Open the file
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return {};
    }

    // Skip to the start row for this process
    std::string line;
    for (int i = 0; i < start_row; ++i)
    {
        std::getline(file, line);
    }

    // Read the assigned chunk of rows
    std::vector<PointMetadata> points;
    for (int i = start_row; i < end_row && std::getline(file, line); ++i)
    {
        std::istringstream lineStream(line);
        PointMetadata point;
        point.coordinates.resize(opts.dim);
        std::string value;

        // Read coordinates
        for (int j = 0; j < opts.dim; ++j)
        {
            if (std::getline(lineStream, value, ',') || std::getline(lineStream, value, ' '))
            {
                point.coordinates[j] = std::stod(value);
            }
            else
            {
                std::cerr << "Error: Invalid data format in file at row " << i << std::endl;
                continue;
            }
        }

        // Read observation (last column)
        if (!std::getline(lineStream, value))
        {
            std::cerr << "Error: Invalid data format in file at row " << i << std::endl;
            continue;
        }
        point.observation = std::stod(value);

        points.push_back(point);
    }
    // // print first ten observations and coordinates
    // std::cout << "First ten observations and coordinates:" << std::endl;
    // for (size_t i = 0; i < std::min(points.size(), static_cast<size_t>(10)); ++i) {
    //     std::cout << "Observation: " << points[i].observation << ", Coordinates: ";
    //     for (const auto& coord : points[i].coordinates) {
    //         std::cout << coord << " ";
    //     }
    //     std::cout << std::endl;
    // }

    file.close();
    return points;
}

// New combined function that replaces partitionPoints and finerPartition
void partitionPointsDirectly(
    const std::vector<PointMetadata> &localPoints,
    std::vector<std::vector<PointMetadata>> &finerPartitions,
    int numBlocks,
    const Opts &opts)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int mm = numBlocks;

    // Check if we should use classic Vecchia (each point is its own cluster)
    if (localPoints.size() <= 2 * mm)
    {
        // Classic Vecchia case - each point becomes its own cluster
        std::cout << "Classic Vecchia case - each point becomes its own cluster" << std::endl;
        finerPartitions.clear();
        finerPartitions.resize(localPoints.size());

        for (size_t i = 0; i < localPoints.size(); ++i)
        {
            finerPartitions[i].push_back(localPoints[i]);
        }

        return; // No need for further processing
    }

    // Block Vecchia case - continue with the algorithm
    // Initialize finerPartitions with exactly opts.numBlocksPerProcess clusters
    finerPartitions.clear();
    finerPartitions.resize(mm);

    // 1. Randomly choose opts.numBlocksPerProcess points as centers on each node
    std::vector<PointMetadata> localCenters;
    std::vector<int> centerIndices;

    // Initialize random number generator with seed + rank for reproducibility
    std::mt19937 gen(opts.seed + rank);

    // Randomly select opts.numBlocksPerProcess points as centers
    std::vector<int> indices(localPoints.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);

    centerIndices.resize(mm);
    localCenters.resize(mm);

    // Assign centers as the first point in each cluster
    for (int i = 0; i < mm; ++i)
    {
        centerIndices[i] = indices[i];
        localCenters[i] = localPoints[indices[i]];
        // Add the center as the first point in its cluster
        // finerPartitions[i].push_back(localCenters[i]);
    }

    // 2. Gather all centers with their node labels and cluster indices
    // Prepare data for gathering: coordinates + observation + rank (as node label) + cluster index
    std::vector<double> localCentersData;
    for (int i = 0; i < mm; ++i)
    {
        for (int j = 0; j < opts.dim; ++j)
        {
            localCentersData.push_back(localCenters[i].coordinates[j]);
        }
        localCentersData.push_back(localCenters[i].observation);
        localCentersData.push_back(static_cast<double>(rank)); // Node label
        localCentersData.push_back(static_cast<double>(i));    // Cluster index within the node
    }

    // Gather counts from all processes
    int localCenterCount = localCenters.size();
    std::vector<int> centerCounts(size);
    MPI_Allgather(&localCenterCount, 1, MPI_INT, centerCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Calculate displacements and total centers
    std::vector<int> displacements(size, 0);
    int totalCenters = 0;
    for (int i = 0; i < size; ++i)
    {
        displacements[i] = totalCenters * (opts.dim + 3); // +3 for observation, rank, and cluster index
        totalCenters += centerCounts[i];
        centerCounts[i] *= (opts.dim + 3); // Each center has dim + observation + rank + cluster index
    }

    // Gather all centers
    std::vector<double> allCentersData(totalCenters * (opts.dim + 3));
    MPI_Allgatherv(localCentersData.data(), localCenterCount * (opts.dim + 3), MPI_DOUBLE,
                   allCentersData.data(), centerCounts.data(), displacements.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    // Convert gathered data to centers with node labels and cluster indices
    std::vector<PointMetadata> allCenters(totalCenters);
    std::vector<int> centerNodeLabels(totalCenters);
    std::vector<int> centerClusterIndices(totalCenters);

    for (int i = 0; i < totalCenters; ++i)
    {
        allCenters[i].coordinates.resize(opts.dim);
        for (int j = 0; j < opts.dim; ++j)
        {
            allCenters[i].coordinates[j] = allCentersData[i * (opts.dim + 3) + j];
        }
        allCenters[i].observation = allCentersData[i * (opts.dim + 3) + opts.dim];
        centerNodeLabels[i] = static_cast<int>(allCentersData[i * (opts.dim + 3) + opts.dim + 1]);
        centerClusterIndices[i] = static_cast<int>(allCentersData[i * (opts.dim + 3) + opts.dim + 2]);
    }

    // 3. Assign remaining local points to nearest centers and prepare to send them
    std::vector<std::vector<PointMetadata>> pointsToSend(size);
    std::vector<std::vector<int>> clusterIndicesForPoints(size);

#pragma omp parallel for
    for (size_t i = 0; i < localPoints.size(); ++i)
    {
        // // Skip points that are already centers
        // bool isCenter = false;
        // for (int j = 0; j < opts.numBlocksPerProcess; ++j)
        // {
        //     if (i == centerIndices[j])
        //     {
        //         isCenter = true;
        //         break;
        //     }
        // }
        // if (isCenter)
        //     continue;

        // find the min distance between the point and all centers
        double minDist = std::numeric_limits<double>::max();
        int nearestCenterIndex = 0;

        // Find nearest center
        for (int j = 0; j < totalCenters; ++j)
        {
            double dist = 0.0;
            for (int d = 0; d < opts.dim; ++d)
            {
                double diff = localPoints[i].coordinates[d] - allCenters[j].coordinates[d];
                dist += diff * diff;
            }

            if (dist < minDist)
            {
                minDist = dist;
                nearestCenterIndex = j;
            }
        }

        // Determine target node and cluster index
        int targetNode = centerNodeLabels[nearestCenterIndex];
        int clusterIndex = centerClusterIndices[nearestCenterIndex];

#pragma omp critical
        {
            pointsToSend[targetNode].push_back(localPoints[i]);
            clusterIndicesForPoints[targetNode].push_back(clusterIndex);
        }
    }

    // 4. Send points to their assigned nodes
    // Prepare counts for Alltoall
    std::vector<int> sendCounts(size, 0);
    for (int i = 0; i < size; ++i)
    {
        sendCounts[i] = pointsToSend[i].size() * (opts.dim + 1 + 1); // +1 for observation, +1 for cluster index
    }

    std::vector<int> recvCounts(size);
    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Calculate displacements
    std::vector<int> sendDisplacements(size, 0);
    std::vector<int> recvDisplacements(size, 0);
    for (int i = 1; i < size; ++i)
    {
        sendDisplacements[i] = sendDisplacements[i - 1] + sendCounts[i - 1];
        recvDisplacements[i] = recvDisplacements[i - 1] + recvCounts[i - 1];
    }

    // Prepare send buffer
    int totalSendSize = std::accumulate(sendCounts.begin(), sendCounts.end(), 0);
    std::vector<double> sendBuffer(totalSendSize);

    int bufferIndex = 0;
    for (int i = 0; i < size; ++i)
    {
        for (size_t j = 0; j < pointsToSend[i].size(); ++j)
        {
            for (int d = 0; d < opts.dim; ++d)
            {
                sendBuffer[bufferIndex++] = pointsToSend[i][j].coordinates[d];
            }
            sendBuffer[bufferIndex++] = pointsToSend[i][j].observation;
            sendBuffer[bufferIndex++] = static_cast<double>(clusterIndicesForPoints[i][j]);
        }
    }

    // Receive buffer
    int totalRecvSize = std::accumulate(recvCounts.begin(), recvCounts.end(), 0);
    std::vector<double> recvBuffer(totalRecvSize);

    // Exchange data
    MPI_Alltoallv(sendBuffer.data(), sendCounts.data(), sendDisplacements.data(), MPI_DOUBLE,
                  recvBuffer.data(), recvCounts.data(), recvDisplacements.data(), MPI_DOUBLE,
                  MPI_COMM_WORLD);

    // 5. Process received points and add them to their respective clusters
    int pointsPerRecord = opts.dim + 2; // Size of each point record in the buffer
    // int expectedPoints = totalRecvSize / pointsPerRecord;

    // Process points with bounds checking
    for (int i = 0; i < totalRecvSize; i += pointsPerRecord)
    {   
        PointMetadata point;
        point.coordinates.resize(opts.dim);

        for (int d = 0; d < opts.dim; ++d)
        {
            point.coordinates[d] = recvBuffer[i + d];
        }
        point.observation = recvBuffer[i + opts.dim];
        int clusterIndex = static_cast<int>(recvBuffer[i + opts.dim + 1]);
        // Add point to its cluster
        finerPartitions[clusterIndex].push_back(point);
    }
}
