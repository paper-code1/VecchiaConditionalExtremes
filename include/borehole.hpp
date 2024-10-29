#ifndef BOREHOLE_HPP
#define BOREHOLE_HPP

#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include "block_info.h"

// Structure to store scaling information for each parameter
struct ScaleInfo {
    std::vector<double> min_values;   // Minimum values for each parameter
    std::vector<double> max_values;   // Maximum values for each parameter
};

// Namespace to avoid name collisions
namespace Borehole {

// Borehole function implementation
inline double borehole_function(double rw, double r, double Tu, double Hu, double Tl, double Hl, double L, double Kw) {
    double numerator = 2 * M_PI * Tu * (Hu - Hl);
    double logTerm = std::log(r / rw);
    double denominator = logTerm * (1 + (2 * L * Tu) / (logTerm * rw * rw * Kw) + Tu / Tl);
    return numerator / denominator;
}

// Helper function to generate a random double within a range [min, max]
inline double random_double(double min, double max, std::mt19937& gen) {
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

// Helper function to scale a value to the range [0, 1]
inline double scale_to_unit(double value, double min_val, double max_val) {
    return (value - min_val) / (max_val - min_val);
}

// Function to sample from the Borehole function, scale parameters, and return scaling info
inline std::pair<std::vector<PointMetadata>, std::pair<double, double>> sample_borehole(int num_samples, int seed, bool is_training = true, double train_mean = 0.0, double train_variance = 1.0) {
    // Set up the random number generator
    std::mt19937 gen(seed);

    // Define the parameter ranges
    const std::vector<double> min_vals = {0.05, 100.0, 63070.0, 990.0, 63.1, 700.0, 1120.0, 1500.0};
    const std::vector<double> max_vals = {0.15, 50000.0, 115600.0, 1110.0, 116.0, 820.0, 1680.0, 15000.0};

    // Vector to store the sampled results
    std::vector<PointMetadata> samples;
    samples.reserve(num_samples);

    // Reserve space for scaling info
    ScaleInfo scale_info;
    scale_info.min_values = min_vals;
    scale_info.max_values = max_vals;

    // Variables for calculating mean and variance
    double sum = 0.0;
    double sum_sq = 0.0;

    // Sampling loop
    for (int i = 0; i < num_samples; ++i) {
        // Generate random parameters within the specified ranges
        std::vector<double> params(8);
        for (size_t j = 0; j < 8; ++j) {
            params[j] = random_double(min_vals[j], max_vals[j], gen);
        }

        // Evaluate the Borehole function with the sampled parameters
        double result = borehole_function(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7]);

        // Update sum and sum of squares for mean and variance calculation
        if (is_training) {
            sum += result;
            sum_sq += result * result;
        }

        // Scale the parameters to the range [0, 1]
        for (size_t j = 0; j < 8; ++j) {
            params[j] = scale_to_unit(params[j], min_vals[j], max_vals[j]);
        }

        // Store the scaled parameters and the function value in PointMetadata
        PointMetadata point;
        point.coordinates = params;
        point.observation = result;
        samples.push_back(point);
    }

    double mean, variance;
    if (is_training) {
        // Calculate mean and variance for training data
        mean = sum / num_samples;
        variance = (sum_sq / num_samples) - (mean * mean);
    } else {
        // Use provided mean and variance for test data
        mean = train_mean;
        variance = train_variance;
    }

    // Normalize observations
    for (auto& sample : samples) {
        sample.observation = (sample.observation - mean) / std::sqrt(variance);
    }

    // // Calculate MSE using normalized mean (0) as predictor for training data
    // if (is_training) {
    //     double mse_sum = 0.0;
    //     for (const auto& sample : samples) {
    //         double error = sample.observation - 0.0;  // 0.0 is the mean of normalized data
    //         mse_sum += error * error;
    //     }
    //     double avg_mse = mse_sum / num_samples;
    //     std::cout << "Average MSE using normalized mean as predictor: " << avg_mse << std::endl;
    // }

    // Return the vector of sampled results, scaling info, and the mean and variance
    return {samples, {mean, variance}};
}

} // namespace Borehole

#endif // BOREHOLE_HPP
