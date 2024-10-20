#include <assert.h>
#include "mm_cpu.h"

template<idx_t N>
struct matric {
  idx_t M;
  real * a;
  matric(idx_t _M) {
    M = _M;
    a = (real *)aligned_alloc(vwidth, sizeof(real) * M * N);
  }
  /* return a scalar A(i,j) */
  real& operator() (idx_t i, idx_t j) {
#if CHECK_IDX
    assert(i < M);
    assert(j < N);
    assert(i >= 0);
    assert(j >= 0);
#endif
    return a[i * N + j];
  }
  /* return a vector at A(i,j) (i.e., A(i,j:j+L) */
  realv& V(idx_t i, idx_t j) {
#if CHECK_IDX
    assert(i < M);
    assert(j + L <= N);
    assert(i >= 0);
    assert(j >= 0);
#endif
    return ::V(a[i * N + j]);
  }
  void copy_from(matrix A, idx_t i0, idx_t j0) {
    for (idx_t i = 0; i < M; i++) {
      for (idx_t j = 0; j < N; j += L) {
        V(i, j) = A.V(i0 + i, j0 + j);
      }
    }
  }
  void copy_to(matrix A, idx_t i0, idx_t j0) {
    for (idx_t i = 0; i < M; i++) {
      for (idx_t j = 0; j < N; j += L) {
        A.V(i0 + i, j0 + j) = V(i, j);
      }
    }
  }
};

template<idx_t N, idx_t K>
void gemmc(matric<K>& A, matric<N>& B, matric<N>& C) {
  idx_t M = C.M;
  idx_t bM = 8;
  idx_t bN = 2;
  assert(M % bM == 0);
  assert(N % (bN * L) == 0);
  for (idx_t i = 0; i < M; i += bM) {
    for (idx_t j = 0; j < N; j += bN * L) {
      real C_[bM][bN * L] __attribute__((aligned(vwidth)));
      for (idx_t ii = 0; ii < bM; ii++) {
        for (idx_t jj = 0; jj < bN * L; jj += L) {
          V(C_[ii][jj]) = U(0);
        }
      }
      asm volatile("# loop begins");
      for (idx_t k = 0; k < K; k++) {
        for (idx_t ii = 0; ii < bM; ii++) {
          for (idx_t jj = 0; jj < bN * L; jj += L) {
            V(C_[ii][jj]) += A(i+ii,k) * B.V(k,j+jj);
          }
        }
      }
      asm volatile("# loop ends");
      for (idx_t ii = 0; ii < bM; ii++) {
        for (idx_t jj = 0; jj < bN * L; jj += L) {
          C.V(i+ii,j+jj) += V(C_[ii][jj]);
        }
      }
    }
  }
}

void gemm(matrix A, matrix B, matrix C) {
  idx_t M = C.M;
  idx_t N = C.N;
  idx_t K = A.N;
  const idx_t bN = 288;
  const idx_t bK = 544;
  matric<bK> sA(M);             //  M x bK
  matric<bN> sB(bK);            // bK x bN
  matric<bN> sC(M);             //  M x bN
  assert(N % bN == 0);
  assert(K % bK == 0);
  for (idx_t j = 0; j < N; j += bN) {
    sC.copy_from(C, 0, j);
    for (idx_t k = 0; k < K; k += bK) {
      sA.copy_from(A, 0, k);
      sB.copy_from(B, k, j);
      gemmc(sA, sB, sC);
    }
    sC.copy_to(C, 0, j);
  }
}


