#ifndef VECCHIA_HELPER_H
#define VECCHIA_HELPER_H
#include "input_parser.h"

void saveTimeAndGflops(
    double duration_RAC_partitioning, 
    double duration_centers_of_gravity, double duration_send_centers_of_gravity, 
    double duration_reorder_centers, double duration_broadcast_centers, 
    double duration_create_block_info, double duration_block_sending, 
    double duration_nn_searching, double duration_gpu_copy, 
    double duration_computation, double duration_cleanup_gpu,
    double duration_total, 
    double total_gflops, int numPointsPerProcess, int numPointsTotal, 
    int numBlocksPerProcess, int numBlocksTotal, int m, int seed, 
    double mspe, double rmspe, double ci_coverage, double *new_theta, 
    double optimized_log_likelihood, const Opts &opts);

#endif