#include <assert.h>
#include "mm_cuda.h"

__global__ void gemm_cuda(matrix A, matrix B, matrix C) {
  idx_t M = C.M;
  idx_t N = C.N;
  idx_t K = A.N;
  for (idx_t i = 0; i < M; i++) {
    for (idx_t j = 0; j < N; j++) {
      real c = 0;
      for (idx_t k = 0; k < K; k++) {
        c += A(i,k) * B(k,j);
      }
      C(i,j) += c;
    }
  }
}

void gemm(matrix A, matrix B, matrix C) {
  check_launch_error((gemm_cuda<<<1,1>>>(A, B, C)));
}


