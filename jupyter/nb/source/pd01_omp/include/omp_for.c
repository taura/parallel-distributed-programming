#com 1
#include <stdio.h>
#include <unistd.h>
#include <omp.h>

int main() {
#pragma omp parallel
  {
    printf("I am thread %d in a team of %d threads\n",
           omp_get_thread_num(), omp_get_num_threads());
#pragma omp for
    for (int i = 0; i < 24; i++) {
      usleep(100 * 1000 * i);
      printf("iteration %d executed by thread %d\n", i, omp_get_thread_num());
      fflush(stdout);
    }
  }
  return 0;
}

