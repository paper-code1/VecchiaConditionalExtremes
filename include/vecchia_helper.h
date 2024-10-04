#ifndef VECCHIA_HELPER_H
#define VECCHIA_HELPER_H

void saveTimeAndGflops(double duration_preprocessing, double duration_computation, double duration_block_sending, double total_gflops, int numPointsPerProcess, int numBlocksX, int numBlocksY, int m, int seed);

#endif