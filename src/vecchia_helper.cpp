#include <fstream>
#include <iostream>

void saveTimeAndGflops(double duration_outer_partitioning, double duration_finer_partitioning, 
double duration_computation, double duration_candidate_preparation, double duration_nn_searching, double duration_total, double total_gflops, int numPointsPerProcess, int numPointsTotal, int numBlocksPerProcess, int numBlocksTotal, int m, int seed, double mspe, double ci_coverage){
    // save the time and gflops to a file with append mode for csv format
    std::ofstream file("time_and_gflops.csv", std::ios::app);
    // write the data to the file
    // write the header if the file is empty
    if (file.tellp() == 0) {
        file << "duration_outer_partitioning,duration_finer_partitioning,duration_candidate_preparation,duration_nn_searching,duration_computation,duration_total,total_gflops,numPointsPerProcess,numPointsTotal,numBlocksPerProcess,numBlocksTotal,m,seed,mspe,ci_coverage" << std::endl;
    }
    file << duration_outer_partitioning << "," << duration_finer_partitioning << "," << duration_candidate_preparation << "," << duration_nn_searching << "," << duration_computation << "," << duration_total << "," << total_gflops << "," << numPointsPerProcess << "," << numPointsTotal << "," << numBlocksPerProcess << "," << numBlocksTotal << "," << m << "," << seed << "," << mspe << "," << ci_coverage << std::endl;
}


