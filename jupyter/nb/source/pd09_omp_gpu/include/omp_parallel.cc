#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int main(int argc, char ** argv) {
  int n_threads= atoi(getenv("OMP_NUM_THREADS"));
  if (n_threads % 32) {
    fprintf(stderr, "OMP_NUM_THREADS (%d) must be a multiple of 32\n", n_threads);
    exit(1);
  }
  printf("hello on host\n");
#pragma omp target teams
  {
    printf("in teams: %03d/%03d\n", omp_get_team_num(), omp_get_num_teams());
#pragma omp parallel num_threads(n_threads)
    printf("in parallel: %03d/%03d %03d/%03d\n",
           omp_get_team_num(), omp_get_num_teams(),
           omp_get_thread_num(), omp_get_num_threads());
  }
  printf("back on host\n");
  return 0;
}
