#include <stdio.h>
#include <unistd.h>
#include <omp.h>

void recursive_tasks(int n, int tid) {
  printf("task %d by %d of %d\n",
         tid, omp_get_thread_num(), omp_get_num_threads());
  fflush(stdout);
  if (n == 0) {
    usleep(300 * 1000);
  } else {
#pragma omp task
    recursive_tasks(n - 1, 2 * tid + 1);
#pragma omp task
    recursive_tasks(n - 1, 2 * tid + 2);
#pragma omp taskwait
  }
}
int main() {
  double t0 = omp_get_wtime();
#pragma omp parallel
#pragma omp master
  {
    recursive_tasks(5, 0);
  }
  double t1 = omp_get_wtime();
  printf("%f sec\n", t1 - t0);
  return 0;
}
