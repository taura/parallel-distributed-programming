#include <stdio.h>
#include <omp.h>
int main() {
  printf("hello on host\n");
#pragma omp target
#pragma omp teams
  printf("in teams: %03d/%03d\n", omp_get_team_num(), omp_get_num_teams());
  printf("back on host\n");
  return 0;
}
