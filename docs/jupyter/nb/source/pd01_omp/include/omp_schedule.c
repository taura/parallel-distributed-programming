#com 1
#include <stdio.h>
#include <unistd.h>
#include <omp.h>

int main() {
  double t0 = omp_get_wtime();
  /* ----- add schedule clause below ----- */
#pragma omp parallel for
  for (int i = 0; i < 24; i++) {
    usleep(100 * 1000 * i);     /* sleep 100 x i milliseconds */
    printf("iteration %d executed by thread %d\n", i, omp_get_thread_num());
    fflush(stdout);
  }
  double t1 = omp_get_wtime();
  printf("%f sec\n", t1 - t0);
  return 0;
}

