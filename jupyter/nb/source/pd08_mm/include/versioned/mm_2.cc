#include <assert.h>
#include "mm_cpu.h"

void gemm(matrix A, matrix B, matrix C) {
  idx_t M = C.M;
  idx_t N = C.N;
  idx_t K = A.N;
  assert(N % L == 0);
  for (idx_t i = 0; i < M; i++) {
    for (idx_t j = 0; j < N; j += L) {
      realv c = U(0);
      asm volatile("# loop begins");
      for (idx_t k = 0; k < K; k++) {
        c += A(i,k) * B.V(k,j);
      }
      asm volatile("# loop ends");
      C.V(i,j) += c;
    }
  }
}


