#include <assert.h>
#include <stdio.h>
/*** if False */
/*
  1 : c=malloc;                launch(c);              show c;  free(c);               (segfault on device)
  2 :           c'=cudamalloc; launch(c');             show c';          cudafree(c'); (segfault on host)
  3 : c=malloc; c'=cudamalloc; launch(c');             show c;  free(c); cudafree(c'); (show wrong result)
  4 : c=malloc; c'=cudamalloc; launch(c'); cudamemcpy; show c;  free(c); cudafree(c'); (answer)
  5 : c=mallocmanaged; launch(c); show c;                                cudafree(c);  (answer)
 */
/*** endif */
#include "cuda_util.h"

__global__ void cuda_thread_fun(long * p, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    p[i] = i * i;
  }
}

int main(int argc, char ** argv) {
  int n               = (argc > 1 ? atoi(argv[1]) : 10);
  int thread_block_sz = (argc > 2 ? atoi(argv[2]) : 3);
  int n_thread_blocks = (n + thread_block_sz - 1) / thread_block_sz;
/*** if VER in [1,3,4] */
  long * c = (long *)malloc(sizeof(long) * n);
/*** endif */
/*** if VER in [2,3,4] */
  long * c_dev;
  check_api_error(cudaMalloc(&c_dev, sizeof(long) * n));
/*** endif */
/*** if VER in [5] */
  long * c;
  check_api_error(cudaMallocManaged(&c, sizeof(long) * n));
/*** endif */
/*** if VER in [1,5] */
  check_launch_error((cuda_thread_fun<<<n_thread_blocks,thread_block_sz>>>(c, n)));
/*** elif VER in [2,3,4] */
  check_launch_error((cuda_thread_fun<<<n_thread_blocks,thread_block_sz>>>(c_dev, n)));
/*** endif */
  check_api_error(cudaDeviceSynchronize());
/*** if VER in [4] */
  check_api_error(cudaMemcpy(c, c_dev, sizeof(long) * n, cudaMemcpyDeviceToHost));
/*** endif */
/*** if VER in [2] */
  for (int i = 0; i < n; i++) {
    printf("c[%d] = %ld\n", i, c_dev[i]);
  }
/*** elif VER in [1,3,4,5] */
  for (int i = 0; i < n; i++) {
    printf("c[%d] = %ld\n", i, c[i]);
  }
/*** endif */
/*** if VER in [1,3,4] */
  free(c);
/*** endif */
/*** if VER in [2,3,4] */
  check_api_error(cudaFree(c_dev));
/*** endif */
/*** if VER in [5] */
  check_api_error(cudaFree(c));
/*** endif */
  return 0;
}
