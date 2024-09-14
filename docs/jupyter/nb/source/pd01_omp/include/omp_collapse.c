#com 2

#include <stdio.h>
#include <unistd.h>
#include <omp.h>

#ifpy VER == 1
int main() {
  double t0 = omp_get_wtime();
  /* apply collapse and schedule */
#pragma omp parallel for
  for (int i = 0; i < 5; i++) {
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
#elifpy VER == 2
int main() {
  double t0 = omp_get_wtime();
  /* apply collapse and schedule */
#pragma omp parallel for collapse(2) schedule(runtime)
  for (int i = 0; i < 5; i++) {
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
#endifpy


