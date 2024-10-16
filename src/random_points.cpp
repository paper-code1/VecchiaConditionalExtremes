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

// Function to perform finer partitioning within each processor using k-means++
void finerPartition(const std::vector<PointMetadata>& metadata, int numBlocksPerProcess, std::vector<std::vector<PointMetadata>>& finerPartitions, const Opts& opts)
{
    finerPartitions.clear();
    finerPartitions.resize(numBlocksPerProcess);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Perform k-means++ clustering
    std::vector<int> clusters = kMeansPlusPlus(metadata, numBlocksPerProcess, opts.dim, opts.kmeans_max_iter, rank, opts.seed);

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

// Add this function to read points concurrently
std::vector<PointMetadata> readPointsConcurrently(const std::string& filename, const Opts& opts) {
    MPI_File fh;
    MPI_Status status;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Open the file
    MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    // Get the file size
    MPI_Offset filesize;
    MPI_File_get_size(fh, &filesize);

    // Calculate the chunk size and offset for each process
    MPI_Offset chunk_size = filesize / size;
    MPI_Offset offset = rank * chunk_size;

    // Adjust the last process to read any remaining bytes
    if (rank == size - 1) {
        chunk_size = filesize - offset;
    }

    // Allocate buffer and read the chunk
    std::vector<char> buffer(chunk_size);
    MPI_File_read_at(fh, offset, buffer.data(), chunk_size, MPI_CHAR, &status);

    // Close the file
    MPI_File_close(&fh);

    // Process the buffer to extract PointMetadata
    std::vector<PointMetadata> points;
    std::istringstream iss(std::string(buffer.begin(), buffer.end()));
    std::string line;

    while (std::getline(iss, line)) {
        // Skip partial lines at the beginning and end of the chunk
        if ((rank > 0 && iss.tellg() == 0) || 
            (rank < size - 1 && iss.eof() && !line.empty())) {
            continue;
        }

        std::istringstream lineStream(line);
        PointMetadata point;
        point.coordinates.resize(opts.dim);
        std::string value;

        // Read coordinates
        for (int i = 0; i < opts.dim; ++i) {
            if (!std::getline(lineStream, value, ',')) {
                std::cerr << "Error: Invalid data format in file" << std::endl;
                continue;
            }
            point.coordinates[i] = std::stod(value);
        }

        // Read observation (last column)
        if (!std::getline(lineStream, value)) {
            std::cerr << "Error: Invalid data format in file" << std::endl;
            continue;
        }
        point.observation = std::stod(value);

        points.push_back(point);
    }

    return points;
}