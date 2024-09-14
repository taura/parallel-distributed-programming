#com 1
#include <stdio.h>
#include <unistd.h>
#include <omp.h>

int main() {
  double t0 = omp_get_wtime();
#pragma omp parallel master
  printf("I am thread %d of a team of %d threads\n",
         omp_get_thread_num(), omp_get_num_threads());
  double t1 = omp_get_wtime();
  printf("%f sec\n", t1 - t0);
  return 0;
}

