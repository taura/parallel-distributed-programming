#com 2
#include <assert.h>
#include <stdio.h>

#include "cuda_util.h"

#include <cooperative_groups.h>
#ifpy VER == 3
#include <cooperative_groups/reduce.h>
#endifpy

//using namespace cooperative_groups;
// Alternatively use an alias to avoid polluting the namespace with collective algorithms
namespace cg = cooperative_groups;

__global__ void sum_array(double * c, long n) {
  // should return c[0] + c[1] + ... + c[n-1] in c[0]
  // you can destroy other elements of the array
  cg::grid_group g = cg::this_grid();
  unsigned long long i = g.thread_rank();
#ifpy VER == 1
  
#elifpy VER == 2
  unsigned long long h;
  for (int m = n; m > 1; m = h) {
    h = (m + 1) / 2;
    if (i + h < m) {
      c[i] += c[i + h];
    }
    g.sync();
  }
#endifpy
}

int main(int argc, char ** argv) {
  long n                = (argc > 1 ? atoi(argv[1]) : 10000);
  int threads_per_block = (argc > 2 ? atoi(argv[2]) : 64);
  int n_thread_blocks = (n + threads_per_block - 1) / threads_per_block;

  double * c = (double *)malloc(sizeof(double) * n);
  for (long i = 0; i < n; i++) {
    c[i] = 1.0;
  }
  double * c_dev;
  check_api_error(cudaMalloc(&c_dev, sizeof(double) * n));
  check_api_error(cudaMemcpy(c_dev, c, sizeof(double) * n, cudaMemcpyHostToDevice));
  void * args[2] = { (void *)&c_dev, (void *)&n };
  check_api_error(cudaLaunchCooperativeKernel((void*)sum_array,
                                              n_thread_blocks,
                                              threads_per_block,
                                              args));
  check_api_error(cudaDeviceSynchronize());
  check_api_error(cudaMemcpy(c, c_dev, sizeof(double) * n, cudaMemcpyDeviceToHost));
  check_api_error(cudaFree(c_dev));
  printf("sum = %f\n", c[0]);
  assert(c[0] == n);
  return 0;
}
