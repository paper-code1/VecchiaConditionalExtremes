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
    int numBlocksX;
    int numBlocksY;
    int m; // the number of nearest neighbor
    bool print;
    int gpu_id;
    int seed;
    int dim;
    double distance_threshold;
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
    ("num_loc_per_process", "Number of locations for each processor", cxxopts::value<int>()->default_value("2000"))
    ("sub_partition", "Number of blocks in x and y directions (format: x,y)", cxxopts::value<std::vector<int>>()->default_value("2,40"))
    ("print", "Print additional information", cxxopts::value<bool>()->default_value("false"))
    ("m", "Special rule for the first 100 blocks", cxxopts::value<int>()->default_value("30"))
    ("distance_threshold", "Distance threshold for blocks", cxxopts::value<double>()->default_value("0.2"))
    ("theta", "Parameters for the covariance function", cxxopts::value<std::vector<double>>()->default_value("1.0,0.1,0.5"))
    ("gpu_id", "GPU ID", cxxopts::value<int>()->default_value("0"))
    ("dim", "Dimension of the problem", cxxopts::value<int>()->default_value("2"))
    ("seed", "Seed for random number generator", cxxopts::value<int>()->default_value("0"))
    ("help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        return false;
    }

    opts.numPointsPerProcess = result["num_loc_per_process"].as<int>();
    
    auto sub_partition = result["sub_partition"].as<std::vector<int>>();
    if (sub_partition.size() == 2)
    {
        opts.numBlocksX = sub_partition[0];
        opts.numBlocksY = sub_partition[1];
    }
    else
    {
        std::cerr << "Error: --sub_partition requires exactly two values and its format should be (x,y)" << std::endl;
        return false;
    }
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
    return true;
}

#endif