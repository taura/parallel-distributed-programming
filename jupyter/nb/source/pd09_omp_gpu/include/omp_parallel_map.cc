#include <stdio.h>
#include <omp.h>
#include "omp_gpu_util.h"
enum { N = 1024 };
int main(int argc, char ** argv) {
  int i = 1;
  int n_teams = (argc > i ? atoi(argv[i]) : 10); i++;
  int n_threads = (argc > i ? atoi(argv[i]) : 20); i++;
  int a[N][N];
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      a[i][j] = -1;
    }
  }
  printf("hello on host\n");
#pragma omp target teams map(tofrom: a[0:N][0:N]) num_teams(n_teams)
#pragma omp parallel num_threads(n_threads)
  {
    int i = omp_get_team_num();
    int j = omp_get_thread_num();
    if (i >= N) printf("team_num   = %d >= %d\n", i, N);
    if (j >= N) printf("thread_num = %d >= %d\n", j, N);
    if (i < N && j < N) {
      a[i][j] = get_smid();
    }
#if 0
    printf("hello, I am in team %03d of %d teams, thread %03d of %d threads, on SM %d\n",
           omp_get_team_num(), omp_get_num_teams(),
           omp_get_thread_num(), omp_get_num_threads(),
           get_smid());
#endif
  }
  printf("back on host\n");
  printf("size: %d %d\n", N, N);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("i=%d j=%d sm=%d\n", i, j, a[i][j]);
    }
  }
  return 0;
}
