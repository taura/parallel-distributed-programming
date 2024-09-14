#com 1
#include <assert.h>
#include <stdio.h>

#include "cuda_util.h"

typedef long long int llint;
typedef struct {
  double x;
  llint t0;
  llint t1;
  uint sm0;
  uint sm1;
} record_t;

__global__ void cuda_thread_fun(double a, double b, llint nt, record_t * c, int nthreads) {
  int idx      = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < nthreads) {
    double x = idx;
    c[idx].sm0 = get_smid();
    c[idx].t0 = clock64();
    for (long k = 0; k < nt; k++) {
      x = a * x + b;
    }
    c[idx].t1 = clock64();
    c[idx].sm1 = get_smid();
    c[idx].x = x;
  }
}

int main(int argc, char ** argv) {
  int i = 1;
  int nthreads        = (argc > i ? atoi(argv[i]) : 100);  i++;
  int thread_block_sz = (argc > i ? atoi(argv[i]) : 64);   i++;
  llint n             = (argc > i ? atoll(argv[i]) : 100 * 1000 * 1000); i++;
  double a            = (argc > i ? atof(argv[i]) : 0.99); i++;
  double b            = (argc > i ? atof(argv[i]) : 1.00); i++;
  int n_thread_blocks = (nthreads + thread_block_sz - 1) / thread_block_sz;
  printf("%d threads/block * %d blocks\n", thread_block_sz, n_thread_blocks);

  long rec_sz = sizeof(record_t) * nthreads;
  record_t * r = (record_t *)calloc(rec_sz, 1);
  record_t * r_dev;
  check_api_error(cudaMalloc(&r_dev, rec_sz));
  check_api_error(cudaMemcpy(r_dev, r, rec_sz, cudaMemcpyHostToDevice));
  check_launch_error((cuda_thread_fun<<<n_thread_blocks,thread_block_sz>>>(a, b, n, r_dev, nthreads)));
  cudaDeviceSynchronize();
  check_api_error(cudaMemcpy(r, r_dev, rec_sz, cudaMemcpyDeviceToHost));
  for (long k = 0; k < nthreads; k++) {
    printf("%ld : %f %lld %lld %u %u\n", k, r[k].x, r[k].t0, r[k].t1, r[k].sm0, r[k].sm1);
  }
  return 0;
}
