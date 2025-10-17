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

enum class PrecisionType {
    Double,
    Float
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
    long long numPointsPerProcess;
    long long numPointsTotal;
    long long numBlocksPerProcess;
    long long numBlocksTotal;
    long long numPointsPerProcess_test;
    long long numPointsTotal_test;
    long long numBlocksPerProcess_test;
    long long numBlocksTotal_test;
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
    std::string partition;
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
    PrecisionType precision;

    // mixed precision controls (effective only when precision==Float)
    // Back-compat grouped toggles
    bool mp_cov_double;        // legacy: covariance generation
    bool mp_schur_double;      // legacy: potrf_neighbors + trsm + gemm
    bool mp_final_double;      // legacy: batched add + potrf_final + final trsm/loglike
    bool mp_all_double_ops;    // convenience: enable all operations in double
    // Fine-grained toggles
    bool mp_covgen_double;         // covariance generation
    bool mp_trsm_double;           // TRSM (both conditioning and final TRSM)
    bool mp_gemm_double;           // GEMM in correction
    bool mp_potrf_neighbors_double;// conditioning CHOL (neighbors)
    bool mp_potrf_final_double;    // final CHOL (corrected covariance)
    bool mp_batched_add_double;    // batched_matrix_add and batched_vector_add
    bool mp_core_ops_double;       // preset: enable covgen, trsm, gemm, potrf_final, batched_add

    // timing
    float time_covgen; // legacy aggregate
    float time_cholesky_trsm_gemm; // legacy aggregate
    float time_cholesky_trsm; // legacy aggregate
    float time_gpu_total;
    // fine-grained stage timings (ms)
    float t_cov_self;     // cov(X,X)
    float t_cov_cross;    // cov(NN,X)
    float t_cov_cond;     // cov(NN,NN)
    float t_potrf_neighbors;
    float t_trsm_cross;
    float t_trsm_obs;
    float t_gemm_covcorr;
    float t_gemm_mucorr;
    float t_batched_matadd;
    float t_batched_vecadd;
    float t_potrf_final;
    float t_trsm_final;
    float t_norm_det;

    // paths
    std::string train_metadata_path;
    std::string test_metadata_path;
    std::vector<double> theta;
    int omp_num_threads;
    cudaStream_t stream;
    magma_queue_t queue;
    // performance/testing controls
    int perf;
};

#endif