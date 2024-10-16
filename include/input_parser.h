#ifndef INPUT_PARSER_H
#define INPUT_PARSER_H

#include <iostream>
#include <string>
#include <vector>
#include <magma_v2.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include "cxxopts.hpp"

// Structure to store command line options
struct Opts
{
    int numPointsPerProcess;
    int numPointsTotal;
    int numBlocksPerProcess;
    int numBlocksTotal;
    int m; // the number of nearest neighbor
    bool print;
    int gpu_id;
    int seed;
    int dim;
    double distance_threshold;
    int kmeans_max_iter;
    std::string train_metadata_path;
    std::string test_metadata_path;
    std::vector<double> theta;
    cudaStream_t stream;
    magma_queue_t queue;
};

inline bool parse_args(int argc, char **argv, Opts &opts)
{
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cxxopts::Options options(argv[0], "Block Vecchia approximation");
    options.add_options()
    ("num_total_points", "Total number of points", cxxopts::value<int>()->default_value("20000"))
    ("num_total_blocks", "Total number of blocks", cxxopts::value<int>()->default_value("100"))
    ("print", "Print additional information", cxxopts::value<bool>()->default_value("false"))
    ("m", "Special rule for the first 100 blocks", cxxopts::value<int>()->default_value("30"))
    ("distance_threshold", "Distance threshold for blocks", cxxopts::value<double>()->default_value("0.2"))
    ("theta", "Parameters for the covariance function", cxxopts::value<std::vector<double>>()->default_value("1.0,0.01,0.0001"))
    ("train_metadata_path", "Path to the training metadata file", cxxopts::value<std::string>()->default_value(""))
    ("test_metadata_path", "Path to the testing metadata file", cxxopts::value<std::string>()->default_value(""))
    ("gpu_id", "GPU ID", cxxopts::value<int>()->default_value("0"))
    ("dim", "Dimension of the problem", cxxopts::value<int>()->default_value("2"))
    ("seed", "Seed for random number generator", cxxopts::value<int>()->default_value("0"))
    ("kmeans_max_iter", "Maximum number of iterations for k-means++", cxxopts::value<int>()->default_value("30"))
    ("help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        return false;
    }

    opts.numPointsTotal = result["num_total_points"].as<int>();
    opts.numPointsPerProcess = opts.numPointsTotal / size + (rank < opts.numPointsTotal % size);
    opts.numBlocksTotal = result["num_total_blocks"].as<int>();
    opts.numBlocksPerProcess = opts.numBlocksTotal / size + (rank < opts.numBlocksTotal % size);
    opts.print = result["print"].as<bool>();
    opts.m = result["m"].as<int>();
    opts.distance_threshold = result["distance_threshold"].as<double>();
    opts.theta = result["theta"].as<std::vector<double>>();
    // gpu_id is used for personal server, hen/swan/..., each server has 2 GPUs
    opts.gpu_id = (rank < (size / 2)) ? 0 : 1;
    cudaSetDevice(opts.gpu_id);
    magma_queue_create(opts.gpu_id, &opts.queue);
    opts.stream = magma_queue_get_cuda_stream(opts.queue);
    opts.seed = result["seed"].as<int>();
    opts.dim = result["dim"].as<int>();
    opts.kmeans_max_iter = result["kmeans_max_iter"].as<int>();
    opts.train_metadata_path = result["train_metadata_path"].as<std::string>();
    opts.test_metadata_path = result["test_metadata_path"].as<std::string>();
    return true;
}

#endif