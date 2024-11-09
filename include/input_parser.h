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
    int numPointsPerProcess_test;
    int numPointsTotal_test;
    int numBlocksPerProcess_test;
    int numBlocksTotal_test;
    int m; // the number of nearest neighbor
    int m_test; // the number of nearest neighbor for testing
    bool print;
    int gpu_id;
    int seed;
    int dim;
    double distance_threshold;
    // kmeans and optimization
    int kmeans_max_iter;
    int current_iter;
    int maxeval;
    double xtol_rel;
    int num_simulations;
    std::string mode;
    std::vector<double> lower_bounds;
    std::vector<double> upper_bounds;
    std::vector<double> theta_init;

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
    ("num_total_blocks", "Total number of blocks", cxxopts::value<int>()->default_value("1000"))
    ("print", "Print additional information", cxxopts::value<bool>()->default_value("false"))
    ("m", "Special rule for the first 100 blocks", cxxopts::value<int>()->default_value("200"))
    ("num_total_points_test", "Total number of points for testing", cxxopts::value<int>()->default_value("2000"))
    ("num_total_blocks_test", "Total number of blocks for testing", cxxopts::value<int>()->default_value("100"))
    ("m_test", "Special rule for the first 100 blocks for testing", cxxopts::value<int>()->default_value("120"))
    ("distance_threshold", "Distance threshold for blocks", cxxopts::value<double>()->default_value("0.2"))
    ("distance_scale", "Distance scale for blocks", cxxopts::value<std::vector<double>>()->default_value(""))
    ("lower_bounds", "Lower bounds for optimization", cxxopts::value<std::vector<double>>()->default_value("0.01,0.01,0.01,0.1"))
    ("upper_bounds", "Upper bounds for optimization", cxxopts::value<std::vector<double>>()->default_value("3,3,3,1"))
    ("theta_init", "Initial parameters for optimization", cxxopts::value<std::vector<double>>()->default_value("1.0,0.01,0.5,0.1"))
    ("train_metadata_path", "Path to the training metadata file", cxxopts::value<std::string>()->default_value(""))
    ("test_metadata_path", "Path to the testing metadata file", cxxopts::value<std::string>()->default_value(""))
    ("dim", "Dimension of the problem", cxxopts::value<int>()->default_value("2"))
    ("seed", "Seed for random number generator", cxxopts::value<int>()->default_value("0"))
    ("kmeans_max_iter", "Maximum number of iterations for k-means++", cxxopts::value<int>()->default_value("100"))
    ("current_iter", "Current iteration for optimization", cxxopts::value<int>()->default_value("0"))
    ("maxeval", "Maximum number of function evaluations", cxxopts::value<int>()->default_value("5000"))
    ("xtol_rel", "Relative tolerance for optimization", cxxopts::value<double>()->default_value("1e-5"))
    ("mode", "Mode type (estimation or prediction or performance)", cxxopts::value<std::string>()->default_value("estimation"))
    ("num_simulations", "Number of simulations for evaluation", cxxopts::value<int>()->default_value("1000"))
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
    opts.numPointsTotal_test = result["num_total_points_test"].as<int>();
    opts.numPointsPerProcess_test = opts.numPointsTotal_test / size + (rank < opts.numPointsTotal_test % size);
    opts.numBlocksTotal_test = result["num_total_blocks_test"].as<int>();
    opts.numBlocksPerProcess_test = opts.numBlocksTotal_test / size + (rank < opts.numBlocksTotal_test % size);
    opts.print = result["print"].as<bool>();
    opts.m = result["m"].as<int>();
    opts.m_test = result["m_test"].as<int>();
    opts.distance_threshold = result["distance_threshold"].as<double>();
    // opts.theta = result["theta"].as<std::vector<double>>();
    opts.lower_bounds = result["lower_bounds"].as<std::vector<double>>();
    opts.upper_bounds = result["upper_bounds"].as<std::vector<double>>();
    opts.theta_init = result["theta_init"].as<std::vector<double>>();
    // Get the total number of GPUs available on the current node
    int local_gpu_count = 0;
    cudaGetDeviceCount(&local_gpu_count);
    if (local_gpu_count == 0) {
        std::cerr << "No GPUs found on this node for rank " << rank << std::endl;
        MPI_Finalize();
        return 0;
    }
    opts.gpu_id = rank % local_gpu_count;
    cudaSetDevice(opts.gpu_id);
    magma_queue_create(opts.gpu_id, &opts.queue);
    opts.stream = magma_queue_get_cuda_stream(opts.queue);
    opts.seed = result["seed"].as<int>();
    opts.dim = result["dim"].as<int>();
    if (result.count("distance_scale")) {
        opts.distance_scale = result["distance_scale"].as<std::vector<double>>();
    }else{
        opts.distance_scale = std::vector<double>(opts.dim, 1.0);
    }
    opts.kmeans_max_iter = result["kmeans_max_iter"].as<int>();
    opts.train_metadata_path = result["train_metadata_path"].as<std::string>();
    opts.test_metadata_path = result["test_metadata_path"].as<std::string>();
    opts.maxeval = result["maxeval"].as<int>();
    opts.xtol_rel = result["xtol_rel"].as<double>();
    opts.current_iter = result["current_iter"].as<int>();
    opts.mode = result["mode"].as<std::string>();
    if (opts.mode == "performance"){
        opts.print = true;
        opts.maxeval = 1;
    }
    opts.num_simulations = result["num_simulations"].as<int>();
    return true;
}

#endif