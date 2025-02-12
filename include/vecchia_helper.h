#ifndef VECCHIA_HELPER_H
#define VECCHIA_HELPER_H
#include "input_parser.h"

void saveTimeAndGflops(double duration_outer_partitioning, double duration_finer_partitioning, double duration_candidate_preparation, double duration_nn_searching, double duration_computation, double duration_total, double total_gflops, int numPointsPerProcess, int numPointsTotal, int numBlocksPerProcess, int numBlocksTotal, int m, int seed, double mspe, double rmspe, double ci_coverage, double *new_theta, double optimized_log_likelihood, const Opts &opts);

#endif