#include <assert.h>
#include <stdio.h>

#include "cuda_util.h"

__global__ void cuda_thread_fun(long long * p, int n) {
  int i        = blockDim.x * blockIdx.x + threadIdx.x;
  p[i] = clock64();
}

int main(int argc, char ** argv) {
  int n               = (argc > 1 ? atoi(argv[1]) : 10);
  int thread_block_sz = (argc > 2 ? atoi(argv[2]) : 3);
  int n_thread_blocks = (n + thread_block_sz - 1) / thread_block_sz;

  long long * c = (long long *)malloc(sizeof(long long) * n);
  long long * c_dev;
  check_api_error(cudaMalloc(&c_dev, sizeof(long long) * n));
  check_launch_error((cuda_thread_fun<<<n_thread_blocks,thread_block_sz>>>(c_dev, n)));
  check_api_error(cudaDeviceSynchronize());
  check_api_error(cudaMemcpy(c, c_dev, sizeof(long long) * n, cudaMemcpyDeviceToHost));
  check_api_error(cudaFree(c_dev));
  for (int i = 0; i < n; i++) {
    printf("c[%d] = %lld\n", i, c[i]);
  }
  return 0;
}
