#include <stdio.h>
#include <unistd.h>
#include <omp.h>

int main() {
  double t0 = omp_get_wtime();
#pragma omp parallel
#pragma omp master
#pragma omp taskloop
  for (int i = 0; i < 5; i++) {
    printf("i = %d starts\n", i);
    fflush(stdout);
#pragma omp taskloop
    for (int j = 0; j < 5; j++) {
      usleep(100 * 1000 * (i + j));
      printf("iteration (%d, %d) executed by thread %d\n", i, j, omp_get_thread_num());
      fflush(stdout);
    }
  }
  double t1 = omp_get_wtime();
  printf("%f sec\n", t1 - t0);
  return 0;
}
