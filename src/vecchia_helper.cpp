#include <fstream>
#include <iostream>

void saveTimeAndGflops(double duration_preprocessing, double duration_computation, double total_gflops, int numPointsPerProcess, int numBlocksX, int numBlocksY, int m, int seed){
    // save the time and gflops to a file with append mode for csv format
    std::ofstream file("time_and_gflops.csv", std::ios::app);
    // write the data to the file
    // write the header if the file is empty
    if (file.tellp() == 0) {
        file << "duration_preprocessing,duration_computation,total_gflops,numPointsPerProcess,numBlocksX,numBlocksY,m,seed" << std::endl;
    }
    file << duration_preprocessing << "," << duration_computation << "," << total_gflops << "," << numPointsPerProcess << "," << numBlocksX << "," << numBlocksY << "," << m << "," << seed << std::endl;
}
