/*** com 2 */
#include <assert.h>
#include <stdio.h>

/*** if VER == 1 */
/*
  you'd better spend time on making sure you always check errors ...
*/

void check_api_error_(cudaError_t e,
                      const char * msg, const char * file, int line) {
  if (e) {
    fprintf(stderr, "%s:%d:error: %s %s\n",
            file, line, msg, cudaGetErrorString(e));
    exit(1);
  }
}

#define check_api_error(e) check_api_error_(e, #e, __FILE__, __LINE__)

void check_launch_error_(const char * msg, const char * file, int line) {
  cudaError_t e = cudaGetLastError();
  if (e) {
    fprintf(stderr, "%s:%d:error: %s %s\n",
            file, line, msg, cudaGetErrorString(e));
    exit(1);
  }
}

#define check_launch_error(exp) do { exp; check_launch_error_(#exp, __FILE__, __LINE__); } while (0)

/*** elif VER == 2 */
#include "cuda_util.h"
/*** endif */

__global__ void cuda_thread_fun(int n) {
  int i        = blockDim.x * blockIdx.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;
  if (i < n) {
    printf("hello I am CUDA thread %d out of %d\n", i, nthreads);
  }
}

int main(int argc, char ** argv) {
  int n               = (argc > 1 ? atoi(argv[1]) : 100);
  int thread_block_sz = (argc > 2 ? atoi(argv[2]) : 64);
  int n_thread_blocks = (n + thread_block_sz - 1) / thread_block_sz;
  printf("%d threads/block * %d blocks\n", thread_block_sz, n_thread_blocks);

  check_launch_error((cuda_thread_fun<<<n_thread_blocks,thread_block_sz>>>(n)));
  check_api_error(cudaDeviceSynchronize());
  return 0;
}
