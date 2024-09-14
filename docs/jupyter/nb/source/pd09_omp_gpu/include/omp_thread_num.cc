#include <stdio.h>
#include <omp.h>
int main() {
  printf("hello on host\n");
#pragma omp target
#pragma omp parallel
  printf("hello, I am thread %d of %d threads in team %d of %d teams\n",
         omp_get_thread_num(), omp_get_num_threads(),
         omp_get_team_num(), omp_get_num_teams());
  printf("back on host\n");
  return 0;
}
