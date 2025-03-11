#ifndef INPUT_PARSER_HELPER_H
#define INPUT_PARSER_HELPER_H

#include <string>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>
#include <magma_v2.h>

enum class KernelType {
    PowerExponential,
    Matern12,
    Matern32,
    Matern52,
    Matern72
};

inline KernelType parse_kernel_type(const std::string &kernel_type_str) {
    if (kernel_type_str == "PowerExponential") return KernelType::PowerExponential;
    if (kernel_type_str == "Matern12") return KernelType::Matern12;
    if (kernel_type_str == "Matern32") return KernelType::Matern32;
    if (kernel_type_str == "Matern52") return KernelType::Matern52;
    if (kernel_type_str == "Matern72") return KernelType::Matern72;
    throw std::invalid_argument("Invalid kernel type: " + kernel_type_str);
}


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
    double distance_threshold_coarse;
    double distance_threshold_finer;
    // kmeans and optimization
    int kmeans_max_iter;
    std::string clustering;
    int current_iter;
    int maxeval;
    int nn_multiplier;
    // log_append
    std::string log_append;
    // optimization
    double xtol_rel;
    double ftol_rel;
    int num_simulations;
    std::string mode;
    std::vector<double> lower_bounds;
    std::vector<double> upper_bounds;
    std::vector<double> theta_init;
    std::vector<double> distance_scale;
    std::vector<double> distance_scale_init;
    KernelType kernel_type;
    int range_offset;

    std::string train_metadata_path;
    std::string test_metadata_path;
    std::vector<double> theta;
    int omp_num_threads;
    cudaStream_t stream;
    magma_queue_t queue;
};

#endif