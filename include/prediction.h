#ifndef PREDICTION_H
#define PREDICTION_H

// Function to perform prediction on the GPU
template <typename Real>
std::tuple<double, double, double> performPredictionOnGPU(const GpuDataT<Real> &gpuData, const std::vector<double> &theta, const Opts &opts);

#endif // PREDICTION_H