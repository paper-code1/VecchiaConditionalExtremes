// magma_dprint_gpu.h

#ifndef MAGMA_DPRINT_GPU_H
#define MAGMA_DPRINT_GPU_H

#include <stdio.h>
#include <magma_v2.h>

// Function to print a double-precision matrix from GPU with more decimals
static inline void magma_dprint_gpu_custom(
    magma_int_t m,
    magma_int_t n,
    magmaDouble_const_ptr dA,
    magma_int_t ldda,
    magma_queue_t queue,
    int decimals)
{
    double* hA;
    magma_int_t lda = m;

    // Allocate host memory for the matrix
    hA = (double*)malloc(m * n * sizeof(double));
    if (hA == NULL) {
        fprintf(stderr, "Error: Unable to allocate host memory.\n");
        return;
    }

    // Copy the matrix from GPU to CPU
    magma_dgetmatrix(m, n, dA, ldda, hA, lda, queue);

    // Create a format string to set the desired decimal precision
    char format_str[20];
    snprintf(format_str, sizeof(format_str), "%%.%df  ", decimals);

    // Print the matrix with the specified precision
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            // Print each element using the custom format string
            printf(format_str, hA[i + j * lda]);
        }
        printf("\n");
    }

    // Free host memory
    free(hA);
}

#endif // MAGMA_DPRINT_GPU_H
