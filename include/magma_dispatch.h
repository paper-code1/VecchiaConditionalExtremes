#ifndef MAGMA_DISPATCH_H
#define MAGMA_DISPATCH_H

#include <magma_v2.h>
#include <type_traits>

template <typename Real>
struct MagmaOps;

template <>
struct MagmaOps<double> {
    static inline void potrf_neighbors(magma_uplo_t uplo, const int* d_n, double** d_A_array, const int* d_ldda, magma_int_t* dinfo, magma_int_t batchCount, magma_int_t max_n, magma_queue_t queue) {
        magma_int_t* n_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_n));
        magma_int_t* ldda_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda));
        magma_dpotrf_vbatched_max_nocheck(uplo, n_nc, d_A_array, ldda_nc, dinfo, batchCount, max_n, queue);
    }
    static inline void trsm_max(magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
                                magma_int_t max_m, magma_int_t max_n,
                                const int* d_m, const int* d_n,
                                double alpha,
                                double** d_A_array, const int* d_ldda_A,
                                double** d_B_array, const int* d_ldda_B,
                                magma_int_t batchCount, magma_queue_t queue) {
        magma_int_t* m_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_m));
        magma_int_t* n_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_n));
        magma_int_t* lddaA_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda_A));
        magma_int_t* lddb_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda_B));
        magmablas_dtrsm_vbatched_max_nocheck(side, uplo, transA, diag, max_m, max_n, m_nc, n_nc, alpha, d_A_array, lddaA_nc, d_B_array, lddb_nc, batchCount, queue);
    }
    static inline void gemm_max(magma_trans_t transA, magma_trans_t transB,
                                const int* d_m, const int* d_n, const int* d_k,
                                double alpha,
                                double const* const* d_A_array, const int* d_ldda_A,
                                double const* const* d_B_array, const int* d_ldda_B,
                                double beta,
                                double** d_C_array, const int* d_ldda_C,
                                magma_int_t batchCount,
                                magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
                                magma_queue_t queue) {
        magma_int_t* m_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_m));
        magma_int_t* n_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_n));
        magma_int_t* k_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_k));
        magma_int_t* lddaA_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda_A));
        magma_int_t* lddb_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda_B));
        magma_int_t* lddc_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda_C));
        magmablas_dgemm_vbatched_max_nocheck(transA, transB, m_nc, n_nc, k_nc, alpha, d_A_array, lddaA_nc, d_B_array, lddb_nc, beta, d_C_array, lddc_nc, batchCount, max_m, max_n, max_k, queue);
    }
    static inline void potrf_final(magma_uplo_t uplo, const int* d_n, double** d_A_array, const int* d_ldda, magma_int_t* dinfo, magma_int_t batchCount, magma_queue_t queue) {
        magma_int_t* n_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_n));
        magma_int_t* ldda_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda));
        magma_dpotrf_vbatched(uplo, n_nc, d_A_array, ldda_nc, dinfo, batchCount, queue);
    }
    static inline void trsm_final(magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
                                  const int* d_m, const int* d_n, double alpha,
                                  double** d_A_array, const int* d_ldda_A,
                                  double** d_B_array, const int* d_ldda_B,
                                  magma_int_t batchCount, magma_queue_t queue) {
        magma_int_t* m_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_m));
        magma_int_t* n_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_n));
        magma_int_t* lddaA_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda_A));
        magma_int_t* lddb_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda_B));
        magmablas_dtrsm_vbatched(side, uplo, transA, diag, m_nc, n_nc, alpha, d_A_array, lddaA_nc, d_B_array, lddb_nc, batchCount, queue);
    }
};

template <>
struct MagmaOps<float> {
    static inline void potrf_neighbors(magma_uplo_t uplo, const int* d_n, float** d_A_array, const int* d_ldda, magma_int_t* dinfo, magma_int_t batchCount, magma_int_t max_n, magma_queue_t queue) {
        magma_int_t* n_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_n));
        magma_int_t* ldda_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda));
        magma_spotrf_vbatched_max_nocheck(uplo, n_nc, d_A_array, ldda_nc, dinfo, batchCount, max_n, queue);
    }
    static inline void trsm_max(magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
                                magma_int_t max_m, magma_int_t max_n,
                                const int* d_m, const int* d_n,
                                float alpha,
                                float** d_A_array, const int* d_ldda_A,
                                float** d_B_array, const int* d_ldda_B,
                                magma_int_t batchCount, magma_queue_t queue) {
        magma_int_t* m_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_m));
        magma_int_t* n_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_n));
        magma_int_t* lddaA_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda_A));
        magma_int_t* lddb_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda_B));
        magmablas_strsm_vbatched_max_nocheck(side, uplo, transA, diag, max_m, max_n, m_nc, n_nc, alpha, d_A_array, lddaA_nc, d_B_array, lddb_nc, batchCount, queue);
    }
    static inline void gemm_max(magma_trans_t transA, magma_trans_t transB,
                                const int* d_m, const int* d_n, const int* d_k,
                                float alpha,
                                float const* const* d_A_array, const int* d_ldda_A,
                                float const* const* d_B_array, const int* d_ldda_B,
                                float beta,
                                float** d_C_array, const int* d_ldda_C,
                                magma_int_t batchCount,
                                magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
                                magma_queue_t queue) {
        magma_int_t* m_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_m));
        magma_int_t* n_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_n));
        magma_int_t* k_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_k));
        magma_int_t* lddaA_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda_A));
        magma_int_t* lddb_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda_B));
        magma_int_t* lddc_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda_C));
        magmablas_sgemm_vbatched_max_nocheck(transA, transB, m_nc, n_nc, k_nc, alpha, d_A_array, lddaA_nc, d_B_array, lddb_nc, beta, d_C_array, lddc_nc, batchCount, max_m, max_n, max_k, queue);
    }
    static inline void potrf_final(magma_uplo_t uplo, const int* d_n, float** d_A_array, const int* d_ldda, magma_int_t* dinfo, magma_int_t batchCount, magma_queue_t queue) {
        magma_int_t* n_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_n));
        magma_int_t* ldda_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda));
        magma_spotrf_vbatched(uplo, n_nc, d_A_array, ldda_nc, dinfo, batchCount, queue);
    }
    static inline void trsm_final(magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
                                  const int* d_m, const int* d_n, float alpha,
                                  float** d_A_array, const int* d_ldda_A,
                                  float** d_B_array, const int* d_ldda_B,
                                  magma_int_t batchCount, magma_queue_t queue) {
        magma_int_t* m_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_m));
        magma_int_t* n_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_n));
        magma_int_t* lddaA_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda_A));
        magma_int_t* lddb_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda_B));
        magmablas_strsm_vbatched(side, uplo, transA, diag, m_nc, n_nc, alpha, d_A_array, lddaA_nc, d_B_array, lddb_nc, batchCount, queue);
    }
};

#endif // MAGMA_DISPATCH_H
