#include <fstream>
#include <iostream>
#include <filesystem>
#include "input_parser.h"

void saveTimeAndGflops(double duration_outer_partitioning, double duration_finer_partitioning, 
double duration_computation, double duration_candidate_preparation, double duration_nn_searching, double duration_total, double total_gflops, int numPointsPerProcess, int numPointsTotal, int numBlocksPerProcess, int numBlocksTotal, int m, int seed, double mspe, double rmspe, double ci_coverage, double *new_theta, double optimized_log_likelihood, const Opts &opts){
    // check if the distance is scaled
    bool is_scaled = opts.distance_scale[0] != 1.0;
    // create the log directory if not exists
    std::string logDir = "log";
    if (!std::filesystem::exists(logDir)) {
        std::filesystem::create_directory(logDir);
    }
    // save the time and gflops to a file with append mode for csv format
    std::string logFileName = logDir + "/logFile_numPointsTotal" + std::to_string(opts.numPointsTotal) + "_numBlocksPerProcess" + std::to_string(opts.numBlocksPerProcess) + "_m" + std::to_string(opts.m) + "_seed" + std::to_string(opts.seed) + "_isScaled" + std::to_string(is_scaled) + "_" + opts.log_append + ".csv";
    std::ofstream file(logFileName, std::ios::app);
    // save the mspe in 16 digits after the decimal point
    // std::string mspe_str = std::to_string(mspe);
    // mspe_str = mspe_str.substr(0, mspe_str.find('.') + 17);
    if (file.tellp() == 0) {
        file << "duration_outer_partitioning,duration_finer_partitioning,duration_candidate_preparation,duration_nn_searching,duration_computation,duration_total,total_gflops,numPointsPerProcess,numPointsTotal,numBlocksPerProcess,numBlocksTotal,m,seed,mspe,rmspe,ci_coverage,optimized_log_likelihood" << std::endl;
    }
    file << std::setprecision(15) << duration_outer_partitioning << "," << duration_finer_partitioning << "," << duration_candidate_preparation << "," << duration_nn_searching << "," << duration_computation << "," << duration_total << "," << total_gflops << "," << numPointsPerProcess << "," << numPointsTotal << "," << numBlocksPerProcess << "," << numBlocksTotal << "," << m << "," << seed << "," << mspe << "," << rmspe << "," << ci_coverage << "," << optimized_log_likelihood << std::endl;

    // save the theta to a file
    std::string thetaFileName = logDir + "/theta_numPointsTotal" + std::to_string(opts.numPointsTotal) + "_numBlocksPerProcess" + std::to_string(opts.numBlocksPerProcess) + "_m" + std::to_string(opts.m) + "_seed" + std::to_string(opts.seed) + "_isScaled" + std::to_string(is_scaled) + ".csv";
    // append the theta to the file
    std::ofstream thetaFile(thetaFileName, std::ios::app);
    for (int i = 0; i < opts.dim + opts.range_offset; i++) {
        thetaFile << new_theta[i] << ",";
    }
    thetaFile << std::endl;
}


