#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include "input_parser.h"

void saveTimeAndGflops(
    double duration_RAC_partitioning, 
    double duration_centers_of_gravity, double duration_send_centers_of_gravity, 
    double duration_reorder_centers, 
    double duration_create_block_info, double duration_block_sending, 
    double duration_nn_searching, double duration_gpu_copy, 
    double duration_computation, 
    double duration_gpu_total, // GPU total time is less than computation time, excluding the warmup time
    double duration_cleanup_gpu,
    double duration_total, 
    double total_gflops, int numPointsPerProcess, int numPointsTotal, 
    int numBlocksPerProcess, int numBlocksTotal, int m, int seed, 
    double mspe, double rmspe, double ci_coverage, double *new_theta, 
    double optimized_log_likelihood, const Opts &opts
    ){
    // check if the distance is scaled
    bool is_scaled = opts.distance_scale[0] != 1.0;
    // create the log directory if not exists
    std::string logDir = "log";
    if (!std::filesystem::exists(logDir)) {
        std::filesystem::create_directory(logDir);
    }
    // save the time and gflops to a file with append mode for csv format
    std::string logFileName = logDir + "/logFile_numPointsTotal" + std::to_string(opts.numPointsTotal) + "_numBlocksTotal" + std::to_string(opts.numBlocksTotal) + "_m" + std::to_string(opts.m) + "_seed" + std::to_string(opts.seed) + "_isScaled" + std::to_string(is_scaled) + "_" + opts.log_append + ".csv";
    std::ofstream file(logFileName, std::ios::app);
    // save the mspe in 16 digits after the decimal point
    if (file.tellp() == 0) {
        file << "RAC_partitioning,centers_of_gravity_calculation,send_centers_of_gravity,reorder_centers,create_block_info,block_sending,nn_searching,gpu_copy,computation,gpu_total,cleanup_gpu,total,total_gflops,numPointsPerProcess,numPointsTotal,numBlocksPerProcess,numBlocksTotal,m,seed,mspe,rmspe,ci_coverage,optimized_log_likelihood,iters" <<std::endl;
    }
    file << std::setprecision(15) << duration_RAC_partitioning << "," << duration_centers_of_gravity << "," << duration_send_centers_of_gravity << "," << duration_reorder_centers << "," << duration_create_block_info << "," << duration_block_sending << "," << duration_nn_searching << "," << duration_gpu_copy << "," << duration_computation << "," << duration_gpu_total << "," << duration_cleanup_gpu << "," << duration_total << "," << total_gflops << "," << numPointsPerProcess << "," << numPointsTotal << "," << numBlocksPerProcess << "," << numBlocksTotal << "," << m << "," << seed << "," << mspe << "," << rmspe << "," << ci_coverage << "," << optimized_log_likelihood << "," << opts.current_iter << std::endl;
    file.close();

    // save the theta to a file
    std::string thetaFileName = logDir + "/theta_numPointsTotal" + std::to_string(opts.numPointsTotal) + "_numBlocksTotal" + std::to_string(opts.numBlocksTotal) + "_m" + std::to_string(opts.m) + "_seed" + std::to_string(opts.seed) + "_isScaled" + std::to_string(is_scaled) + "_" + opts.log_append + ".csv";
    // append the theta to the file
    std::ofstream thetaFile(thetaFileName, std::ios::app);
    for (int i = 0; i < opts.dim + opts.range_offset; i++) {
        thetaFile << new_theta[i] << ",";
    }
    thetaFile << std::endl;
    thetaFile.close();
}


