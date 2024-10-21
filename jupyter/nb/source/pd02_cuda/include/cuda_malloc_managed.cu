/*** com 2 */
#include <assert.h>
#include <stdio.h>

#include "cuda_util.h"

__global__ void cuda_thread_fun(long long * p, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  p[i] = i * i;
}

int main(int argc, char ** argv) {
  int n               = (argc > 1 ? atoi(argv[1]) : 10);
  int thread_block_sz = (argc > 2 ? atoi(argv[2]) : 3);
  int n_thread_blocks = (n + thread_block_sz - 1) / thread_block_sz;

/*** if VER == 1 */
  long long * c = (long long *)malloc(sizeof(long long) * n);
  long long * c_dev;
  check_api_error(cudaMalloc(&c_dev, sizeof(long long) * n));
  check_launch_error((cuda_thread_fun<<<n_thread_blocks,thread_block_sz>>>(c_dev, n)));
/*** elif VER == 2 */
  long long * c;
  check_api_error(cudaMallocManaged(&c, sizeof(long long) * n));
  check_launch_error((cuda_thread_fun<<<n_thread_blocks,thread_block_sz>>>(c, n)));
/*** endif */
  check_api_error(cudaDeviceSynchronize());
/*** if VER == 1 */
  check_api_error(cudaMemcpy(c, c_dev, sizeof(long long) * n, cudaMemcpyDeviceToHost));
/*** endif */
  for (int i = 0; i < n; i++) {
    printf("c[%d] = %ld\n", i, c[i]);
  }
/*** if VER == 1 */
  free(c);
  check_api_error(cudaFree(c_dev));
/*** elif VER == 2 */
  check_api_error(cudaFree(c));
/*** endif */
  return 0;
}
