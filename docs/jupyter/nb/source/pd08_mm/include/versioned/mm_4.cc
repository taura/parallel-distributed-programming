#include <assert.h>
#include "mm_cpu.h"

void gemm(matrix A, matrix B, matrix C) {
  idx_t M = C.M;
  idx_t N = C.N;
  idx_t K = A.N;
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


