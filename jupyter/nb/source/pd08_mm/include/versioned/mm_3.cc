#include <assert.h>
#include "mm_cpu.h"

void gemm(matrix A, matrix B, matrix C) {
  idx_t M = C.M;
  idx_t N = C.N;
  idx_t K = A.N;
  idx_t bM = 8;
  assert(M % bM == 0);
  assert(N % L == 0);
  for (idx_t i = 0; i < M; i += bM) {
    for (idx_t j = 0; j < N; j += L) {
      real C_[bM][L] __attribute__((aligned(vwidth)));
      for (idx_t ii = 0; ii < bM; ii++) {
        V(C_[ii][0]) = U(0);
      }
      asm volatile("# loop begins");
      for (idx_t k = 0; k < K; k++) {
        for (idx_t ii = 0; ii < bM; ii++) {
          V(C_[ii][0]) += A(i+ii,k) * B.V(k,j);
        }
      }
      asm volatile("# loop ends");
      for (idx_t ii = 0; ii < bM; ii++) {
        C.V(i+ii,j) += V(C_[ii][0]);
      }
    }
  }
}


