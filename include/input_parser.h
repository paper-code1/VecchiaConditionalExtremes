#ifndef INPUT_PARSER_H
#define INPUT_PARSER_H

#include <iostream>
#include <string>
#include <vector>
#include <magma_v2.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include "cxxopts.hpp"
#include "input_parser_helper.h"


inline bool parse_args(int argc, char **argv, Opts &opts)
{
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cxxopts::Options options(argv[0], "Block Vecchia approximation");
    options.add_options()
    ("num_total_points", "Total number of points", cxxopts::value<long long>()->default_value("20000"))
    ("num_total_blocks", "Total number of blocks", cxxopts::value<long long >()->default_value("1000"))
    ("print", "Print additional information", cxxopts::value<bool>()->default_value("true")->implicit_value("true"))
    ("m", "Special rule for the first 100 blocks", cxxopts::value<int>()->default_value("200"))
    ("num_total_points_test", "Total number of points for testing", cxxopts::value<long long>()->default_value("2000"))
    ("num_total_blocks_test", "Total number of blocks for testing", cxxopts::value<long long>()->default_value("100"))
    ("m_test", "Special rule for the first 100 blocks for testing", cxxopts::value<int>()->default_value("120"))
    ("distance_threshold_coarse", "Distance threshold for blocks", cxxopts::value<double>()->default_value("0.2"))
    ("distance_threshold_finer", "Distance threshold for blocks", cxxopts::value<double>()->default_value("0.05"))
    ("distance_scale", "Distance scale for blocks, used for scaling distance", cxxopts::value<std::vector<double>>())
    ("distance_scale_init", "Initial distance scale for blocks, used for optimization", cxxopts::value<std::vector<double>>())
    ("kernel_type", "Kernel type", cxxopts::value<std::string>()->default_value("Matern72"))
    ("precision", "Floating point precision (double|float)", cxxopts::value<std::string>()->default_value("double"))
    ("theta_init", "Initial parameters for optimization", cxxopts::value<std::vector<double>>())
    ("lower_bounds", "Lower bounds for optimization", cxxopts::value<std::vector<double>>())
    ("upper_bounds", "Upper bounds for optimization", cxxopts::value<std::vector<double>>())
    ("train_metadata_path", "Path to the training metadata file", cxxopts::value<std::string>()->default_value(""))
    ("test_metadata_path", "Path to the testing metadata file", cxxopts::value<std::string>()->default_value(""))
    ("dim", "Dimension of the problem", cxxopts::value<int>()->default_value("8"))
    ("seed", "Seed for random number generator", cxxopts::value<int>()->default_value("0"))
    ("kmeans_max_iter", "Maximum number of iterations for k-means++", cxxopts::value<int>()->default_value("10"))
    ("clustering", "Use random/kmeans++ clustering for large datasets", cxxopts::value<std::string>()->default_value("random"))
    ("current_iter", "Current iteration for optimization", cxxopts::value<int>()->default_value("0"))
    ("maxeval", "Maximum number of function evaluations", cxxopts::value<int>()->default_value("5000"))
    ("xtol_rel", "Relative tolerance for optimization", cxxopts::value<double>()->default_value("1e-5"))
    ("ftol_rel", "Relative tolerance of function for optimization", cxxopts::value<double>()->default_value("1e-5"))
    ("mode", "Mode type (estimation or prediction)", cxxopts::value<std::string>()->default_value("estimation"))
    ("partition", "Partition type (linear or none)", cxxopts::value<std::string>()->default_value("linear"))
    ("num_simulations", "Number of simulations for evaluation", cxxopts::value<int>()->default_value("1000"))
    ("omp_num_threads", "Number of threads for OpenMP", cxxopts::value<int>()->default_value("20"))
    ("log_append", "Append to the log file", cxxopts::value<std::string>()->default_value(""))
    ("nn_multiplier", "Number of nearest neighbors multiplier", cxxopts::value<int>()->default_value("400"))
    ("perf", "Enable performance warmup (0/1)", cxxopts::value<int>()->default_value("1"))
    ("help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        return false;
    }
    try
    {
        opts.kernel_type = parse_kernel_type(result["kernel_type"].as<std::string>());
    }
    catch(const std::runtime_error& e)
    {
        if (rank == 0) {
            std::cerr << e.what() << '\n';
        }
        return false;
    }
    // precision
    {
        std::string prec = result["precision"].as<std::string>();
        if (prec == "double" || prec == "fp64" || prec == "64") {
            opts.precision = PrecisionType::Double;
        } else if (prec == "float" || prec == "single" || prec == "fp32" || prec == "32") {
            opts.precision = PrecisionType::Float;
        } else {
            if (rank == 0) {
                std::cerr << "Invalid precision: " << prec << ". Use 'double' or 'float'." << std::endl;
            }
            return false;
        }
    }
    opts.numPointsTotal = result["num_total_points"].as<long long>();
    opts.numPointsPerProcess = opts.numPointsTotal / size + (rank < opts.numPointsTotal % size);
    opts.numBlocksTotal = result["num_total_blocks"].as<long long>();
    opts.numBlocksPerProcess = opts.numBlocksTotal / size + (rank < opts.numBlocksTotal % size);
    opts.numPointsTotal_test = result["num_total_points_test"].as<long long>();
    opts.numPointsPerProcess_test = opts.numPointsTotal_test / size + (rank < opts.numPointsTotal_test % size);
    opts.numBlocksTotal_test = result["num_total_blocks_test"].as<long long>();
    opts.numBlocksPerProcess_test = opts.numBlocksTotal_test / size + (rank < opts.numBlocksTotal_test % size);
    opts.print = result["print"].as<bool>();
    opts.m = result["m"].as<int>();
    opts.m_test = result["m_test"].as<int>();
    opts.distance_threshold_coarse = result["distance_threshold_coarse"].as<double>();
    opts.distance_threshold_finer = result["distance_threshold_finer"].as<double>();
    opts.log_append = result["log_append"].as<std::string>();
    opts.nn_multiplier = result["nn_multiplier"].as<int>();
    // Get local rank within the node using MPI
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    int local_rank, local_size;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_size(node_comm, &local_size);

    // Get the total number of GPUs available on the current node
    int local_gpu_count = 0;
    cudaGetDeviceCount(&local_gpu_count); // slurm, it's always 1
    if (local_gpu_count == 0) {
        std::cerr << "No GPUs found on this node for rank " << rank << std::endl;
        MPI_Comm_free(&node_comm);
        MPI_Finalize();
        return false;
    }

    // If there are more ranks than GPUs, multiple ranks will share each GPU
    if (local_size > local_gpu_count && rank == 0) {
        std::cout << "Warning: " << local_size << " ranks will share " << local_gpu_count 
                 << " GPUs on each node (approximately " 
                 << local_size / local_gpu_count + (local_size % local_gpu_count > 0) 
                 << " ranks per GPU)" << std::endl;
    }

    // Assign GPU based on local rank instead of global rank
    opts.gpu_id = local_rank % local_gpu_count;
    cudaSetDevice(opts.gpu_id);
    magma_queue_create(opts.gpu_id, &opts.queue);
    opts.stream = magma_queue_get_cuda_stream(opts.queue);

    MPI_Comm_free(&node_comm);

    opts.seed = result["seed"].as<int>();
    opts.dim = result["dim"].as<int>();
    opts.omp_num_threads = result["omp_num_threads"].as<int>();
    // optimized parameters
    if (result.count("distance_scale")) {
        opts.distance_scale = result["distance_scale"].as<std::vector<double>>();
    }else{
        opts.distance_scale = std::vector<double>(opts.dim, 1.0);
    }
    if (result.count("distance_scale_init")){
        opts.distance_scale_init = result["distance_scale_init"].as<std::vector<double>>();
    }else{
        opts.distance_scale_init = opts.distance_scale;
    }
    if (result.count("theta_init")){
        opts.theta_init = result["theta_init"].as<std::vector<double>>();
        opts.range_offset = opts.theta_init.size();
    }else{
        switch (opts.kernel_type)
        {
        case KernelType::PowerExponential:
            opts.theta_init = {1.0, 0.5, 0.00001}; // variance, smoothness, nugget;
            opts.range_offset = 3;
            break;
        case KernelType::Matern12:
            opts.theta_init = {1.0, 0.00001}; // variance, nugget;
            opts.range_offset = 2;
            break;
        case KernelType::Matern32:
            opts.theta_init = {1.0, 0.00001}; // variance, nugget;
            opts.range_offset = 2;
            break;
        case KernelType::Matern52:
            opts.theta_init = {1.0, 0.00001}; // variance, nugget;
            opts.range_offset = 2;
            break;
        case KernelType::Matern72:
            opts.theta_init = {1.0, 0.00001}; // variance, nugget;
            opts.range_offset = 2;
            break;
        default:
            break;
        }
    }
    // append the distance scale to the theta_init
    opts.theta_init.insert(opts.theta_init.end(), opts.distance_scale_init.begin(), opts.distance_scale_init.end());
    // add the theta_init to the bounds
    for (int i=0; i < opts.theta_init.size(); i++) {
        opts.lower_bounds.push_back(opts.theta_init[i] * 0.001);
        opts.upper_bounds.push_back(opts.theta_init[i] * 10);
    }
    // other options
    opts.kmeans_max_iter = result["kmeans_max_iter"].as<int>();
    opts.train_metadata_path = result["train_metadata_path"].as<std::string>();
    opts.test_metadata_path = result["test_metadata_path"].as<std::string>();
    opts.maxeval = result["maxeval"].as<int>();
    opts.xtol_rel = result["xtol_rel"].as<double>();
    opts.ftol_rel = result["ftol_rel"].as<double>();
    opts.current_iter = result["current_iter"].as<int>();
    opts.clustering = result["clustering"].as<std::string>();
    opts.partition = result["partition"].as<std::string>();
    opts.mode = result["mode"].as<std::string>();
    // if (opts.mode == "performance"){
    //     opts.print = true;
    //     opts.maxeval = 1;
    // }
    opts.num_simulations = result["num_simulations"].as<int>();
    opts.time_cholesky_trsm_gemm = 0;
    opts.time_cholesky_trsm = 0;
    opts.time_covgen = 0;
    opts.time_gpu_total = 0;
    opts.perf = result["perf"].as<int>();
    return true;
}

#endif