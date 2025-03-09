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
#include <fstream>  // Add this line
#include <sstream>  // Add this line for std::istringstream
#include "random_points.h"

// Define custom reduction for 2D vector
#pragma omp declare reduction(vec2d_double_plus : std::vector<std::vector<double>> : \
    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), \
        [](std::vector<double>& a, const std::vector<double>& b) { \
            std::transform(a.begin(), a.end(), b.begin(), a.begin(), std::plus<double>()); \
            return a; \
        })) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size(), std::vector<double>(omp_orig[0].size())))

// Define custom reduction for 1D vector
#pragma omp declare reduction(vec_int_plus : std::vector<int> : \
    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<int>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

// Function to generate random double between 0 and 1
double generateRandomDouble()
{
    return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}

// Function to generate random points
std::vector<PointMetadata> generateRandomPoints(int numPointsPerProcess, const Opts& opts)
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
void partitionPoints(const std::vector<PointMetadata> &localMetadata, std::vector<PointMetadata> &localMetadata_block, const Opts& opts)
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
std::vector<int> randomClustering(const std::vector<PointMetadata>& metadata, int k, int dim, int seed) {
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
    for (int i = 0; i < k; ++i) {
        centers[i] = metadata[centerIndices[i]].coordinates;
        clusters[centerIndices[i]] = i;
    }
    
    // 2. Assign remaining points to nearest center
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < numPoints; ++i) {
        // Skip if this point is a center
        if (i < k && i == centerIndices[i]) continue;
        
        double minDist = std::numeric_limits<double>::max();
        int nearestCluster = 0;
        
        // Find nearest center
        for (int j = 0; j < k; ++j) {
            double dist = 0.0;
            for (int d = 0; d < dim; ++d) {
                double diff = metadata[i].coordinates[d] - centers[j][d];
                dist += diff * diff;
            }
            
            if (dist < minDist) {
                minDist = dist;
                nearestCluster = j;
            }
        }
        
        clusters[i] = nearestCluster;
    }
    
    return clusters;
}

// Function to perform finer partitioning within each processor using k-means++
void finerPartition(const std::vector<PointMetadata>& metadata, int numBlocksPerProcess, 
                   std::vector<std::vector<PointMetadata>>& finerPartitions, const Opts& opts)
{
    finerPartitions.clear();
    finerPartitions.resize(numBlocksPerProcess);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Perform clustering
    std::vector<int> clusters;
    if (numBlocksPerProcess * 3 < metadata.size()) {
        // block Vecchia
        if (opts.clustering == "random") {
            // Use random clustering for large datasets
            clusters = randomClustering(metadata, numBlocksPerProcess, opts.dim, opts.seed + rank);
        } else if (opts.clustering == "kmeans++") {
            // Use k-means++ for smaller datasets
            clusters = kMeansPlusPlus(metadata, numBlocksPerProcess, opts.dim, opts.kmeans_max_iter, rank, opts.seed);
        } else {
            std::cerr << "Invalid clustering method: " << opts.clustering << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    } else {
        // classic Vecchia
        clusters.resize(metadata.size());
        std::iota(clusters.begin(), clusters.end(), 0);
    }

    // Assign points to clusters
    for (size_t i = 0; i < metadata.size(); ++i) {
        finerPartitions[clusters[i]].push_back(metadata[i]);
    }
}

// Function to calculate centers of gravity for each block (specific for 2D)
std::vector<std::vector<double>> calculateCentersOfGravity(const std::vector<std::vector<PointMetadata>> &finerPartitions, const Opts& opts)
{
    int numBlocks = finerPartitions.size();
    std::vector<std::vector<double>> centers(numBlocks, std::vector<double>(opts.dim));

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
void sendCentersOfGravityToRoot(const std::vector<std::vector<double>> &centers, std::vector<std::vector<double>> &allCenters, const Opts& opts)
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
            displacements[i] = totalCenters * opts.dim;
            totalCenters += recvCounts[i];
            recvCounts[i] *= opts.dim; // Each center has DIMENSION doubles
        }
    }

    std::vector<double> sendBuffer(numCenters * opts.dim);
    for (int i = 0; i < numCenters; ++i)
    {
        for (int j = 0; j < opts.dim; ++j)
        {
            sendBuffer[i * opts.dim + j] = centers[i][j];
        }
    }

    std::vector<double> recvBuffer;
    if (rank == 0)
    {
        recvBuffer.resize(totalCenters * opts.dim);
    }
    MPI_Gatherv(sendBuffer.data(), numCenters * opts.dim, MPI_DOUBLE, recvBuffer.data(), recvCounts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        allCenters.clear();
        allCenters.resize(totalCenters, std::vector<double>(opts.dim));
        for (int i = 0; i < totalCenters; ++i)
        {
            for (int j = 0; j < opts.dim; ++j)
            {
                allCenters[i][j] = recvBuffer[i * opts.dim + j];
            }
        }
    }
}

// Function to randomly reorder centers at processor 0
void reorderCenters(std::vector<std::vector<double>> &centers, const Opts& opts)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        // Shuffle non-empty centers
        std::random_shuffle(centers.begin(), centers.end());
    }
}

// Function to broadcast reordered centers to all processors
void broadcastCenters(std::vector<std::vector<double>>& allCenters, int numCenters, const Opts& opts)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Prepare data for broadcast
    std::vector<double> centersData(numCenters * opts.dim);
    if (rank == 0)
    {
        for (int i = 0; i < numCenters; ++i)
        {
            for (int j = 0; j < opts.dim; ++j)
            {
                centersData[i * opts.dim + j] = allCenters[i][j];
            }
        }
    }

    // Broadcast the reordered centers to all processes
    MPI_Bcast(centersData.data(), numCenters * opts.dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Resize the allCenters vector on all processes
    allCenters.resize(numCenters, std::vector<double>(opts.dim));

    // Convert data back to vector of vectors
    for (int i = 0; i < numCenters; ++i)
    {
        for (int j = 0; j < opts.dim; ++j)
        {
            allCenters[i][j] = centersData[i * opts.dim + j];
        }
    }
}

// Function to perform k-means++ clustering with iterations and parallelization
std::vector<int> kMeansPlusPlus(const std::vector<PointMetadata>& metadata, int k, int dim, int maxIterations, int rank, int seed)
{
    std::vector<int> clusters(metadata.size());
    std::vector<std::vector<double>> centroids(k, std::vector<double>(dim));
    std::mt19937 gen(seed + rank);  // Use fixed seed + rank for reproducibility
    // Choose the first centroid randomly
    std::uniform_int_distribution<> dis(0, metadata.size() - 1);
    int firstCentroidIndex = dis(gen);
    centroids[0] = metadata[firstCentroidIndex].coordinates;

    // Choose the remaining centroids
    for (int i = 1; i < k; ++i) {
        std::vector<double> distances(metadata.size(), std::numeric_limits<double>::max());
        
        #pragma omp parallel for
        for (size_t j = 0; j < metadata.size(); ++j) {
            for (int c = 0; c < i; ++c) {
                double dist = 0;
                for (int d = 0; d < dim; ++d) {
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
    for (int iter = 0; iter < maxIterations; ++iter) {
        // print info for every 10 iterations
        if (iter % 30 == 0 && rank == 0){
            std::cout << "K-means iteration: " << iter << std::endl;
        }
        // Assign points to the nearest centroid
        #pragma omp parallel for
        for (size_t i = 0; i < metadata.size(); ++i) {
            double minDist = std::numeric_limits<double>::max();
            int nearestCentroid = 0;
            for (int j = 0; j < k; ++j) {
                double dist = 0;
                for (int d = 0; d < dim; ++d) {
                    double diff = metadata[i].coordinates[d] - centroids[j][d];
                    dist += diff * diff;
                }
                if (dist < minDist) {
                    minDist = dist;
                    nearestCentroid = j;
                }
            }
            clusters[i] = nearestCentroid;
        }

        // Recalculate centroids
        std::vector<std::vector<double>> newCentroids(k, std::vector<double>(dim, 0.0));
        std::vector<int> clusterSizes(k, 0);

        #pragma omp parallel for reduction(vec2d_double_plus:newCentroids) reduction(vec_int_plus:clusterSizes)
        for (size_t i = 0; i < metadata.size(); ++i) {
            int cluster = clusters[i];
            for (int d = 0; d < dim; ++d) {
                newCentroids[cluster][d] += metadata[i].coordinates[d];
            }
            clusterSizes[cluster]++;
        }

        // Update centroids
        for (int i = 0; i < k; ++i) {
            if (clusterSizes[i] > 0) {
                for (int d = 0; d < dim; ++d) {
                    centroids[i][d] = newCentroids[i][d] / clusterSizes[i];
                }
            } else {
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
std::vector<PointMetadata> readPointsConcurrently(const std::string& filename, const Opts& opts) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numPointsTotal = opts.numPointsTotal;

    // Calculate the number of rows each process should read
    int rows_per_process = numPointsTotal / size;
    int remainder = numPointsTotal % size;

    // Calculate the start and end rows for this process
    int start_row = rank * rows_per_process + std::min(rank, remainder);
    int end_row = start_row + rows_per_process + (rank < remainder ? 1 : 0);

    // Open the file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return {};
    }

    // Skip to the start row for this process
    std::string line;
    for (int i = 0; i < start_row; ++i) {
        std::getline(file, line);
    }

    // Read the assigned chunk of rows
    std::vector<PointMetadata> points;
    for (int i = start_row; i < end_row && std::getline(file, line); ++i) {
        std::istringstream lineStream(line);
        PointMetadata point;
        point.coordinates.resize(opts.dim);
        std::string value;

        // Read coordinates
        for (int j = 0; j < opts.dim; ++j) {
            if (std::getline(lineStream, value, ',') || std::getline(lineStream, value, ' ')) {
                point.coordinates[j] = std::stod(value);
            } else {
                std::cerr << "Error: Invalid data format in file at row " << i << std::endl;
                continue;
            }
        }

        // Read observation (last column)
        if (!std::getline(lineStream, value)) {
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
