#ifndef PREDICTION_H
#define PREDICTION_H

// Function to perform prediction on the GPU
void performPredictionOnGPU(const GpuData &gpuData, const std::vector<double> &theta, const Opts &opts);

#endif // PREDICTION_H