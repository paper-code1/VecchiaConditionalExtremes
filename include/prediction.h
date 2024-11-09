#ifndef PREDICTION_H
#define PREDICTION_H

// Function to perform prediction on the GPU
std::pair<double, double> performPredictionOnGPU(const GpuData &gpuData, const std::vector<double> &theta, const Opts &opts);

#endif // PREDICTION_H