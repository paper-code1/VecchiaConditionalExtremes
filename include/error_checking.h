#ifndef ERROR_CHECKING_H
#define ERROR_CHECKING_H

#include <cuda_runtime.h>
#include <magma_v2.h>
#include <iostream>

// Updated function to check CUDA errors with file and line information
#define checkCudaError(error) _checkCudaError(error, __FILE__, __LINE__)
#define checkMagmaError(error) _checkMagmaError(error, __FILE__, __LINE__)

inline void _checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in " << file << ":" << line 
                  << ": " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

inline void _checkMagmaError(magma_int_t error, const char* file, int line) {
    if (error != MAGMA_SUCCESS) {
        std::cerr << "MAGMA error in " << file << ":" << line 
                  << ": " << magma_strerror(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

#endif // ERROR_CHECKING_H