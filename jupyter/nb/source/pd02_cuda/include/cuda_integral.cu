#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <time.h>

#include "cuda_util.h"

double cur_time() {
  struct timespec tp[1];
  clock_gettime(CLOCK_REALTIME, tp);
  return tp->tv_sec + tp->tv_nsec * 1.0e-9;
}

__global__ void cuda_thread_fun(int n, double xa, double ya, double dx, double dy, double * sp) {
  int i          = blockDim.x * blockIdx.x + threadIdx.x;
  int j          = blockDim.y * blockIdx.y + threadIdx.y;
  if (i < n && j < n) {
    double x = xa + i * dx;
    double y = ya + j * dy;
    double z2 = 1 - x * x - y * y;
    if (z2 > 0) {
      atomicAdd(sp, sqrt(z2) * dx * dy);
    }
  }
}

int main(int argc, char ** argv) {
  double xa = 0.0;
  double xb = 1.0;
  double ya = 0.0;
  double yb = 1.0;
  int n = 10000;
  double dx = (xb - xa) / n;
  double dy = (yb - ya) / n;

  // thread configuration
  int nx                = n;
  int ny                = n;
  int thread_block_sz_x = (argc > 1 ? atoi(argv[1]) : 8);
  int thread_block_sz_y = thread_block_sz_x;
  int n_thread_blocks_x = (nx + thread_block_sz_x - 1) / thread_block_sz_x;
  int n_thread_blocks_y = (ny + thread_block_sz_y - 1) / thread_block_sz_y;

  double s = 0.0;
  double * s_dev;
  double t0 = cur_time();
  check_api_error(cudaMalloc(&s_dev, sizeof(double)));
  double t1 = cur_time();
  check_api_error(cudaMemcpy(s_dev, &s, sizeof(double), cudaMemcpyHostToDevice));
  double t2 = cur_time();
  
  dim3 nb(n_thread_blocks_x, n_thread_blocks_y);
  dim3 tpb(thread_block_sz_x, thread_block_sz_y);
  check_launch_error((cuda_thread_fun<<<nb,tpb>>>(n, xa, ya, dx, dy, s_dev)));
  check_api_error(cudaDeviceSynchronize());
  double t3 = cur_time();
  
  check_api_error(cudaMemcpy(&s, s_dev, sizeof(double), cudaMemcpyDeviceToHost));
  double t4 = cur_time();
  
  printf("s = %.9f (err = %e)\n", s, fabs(s - M_PI/6));
  printf(" cudaMalloc  : %f sec\n", t1 - t0);
  printf(" host -> dev : %f sec\n", t2 - t1);
  printf(" kernel      : %f sec\n", t3 - t2);
  printf(" host <- dev : %f sec\n", t4 - t3);
  printf("---------------------------\n");
  printf("total        : %f sec\n", t4 - t0);
  return 0;
}
