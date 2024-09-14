#include <stdio.h>
#include <omp.h>
#include "omp_gpu_util.h"
int main(int argc, char ** argv) {
  int i = 1;
  int m = (argc > i ? atoi(argv[i]) : 5); i++;
  int n = (argc > i ? atoi(argv[i]) : 5); i++;
  long x[m][n];
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      x[i][j] = 0;
    }
  }
#pragma omp target teams distribute map(tofrom: x)
  for (int i = 0; i < m; i++) {
#pragma omp parallel for
    for (int j = 0; j < n; j++) {
      x[i][j] = 123; // get_smid();
    }
  }
  printf("size: %d %d\n", m, n);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("x[%d][%d] = %ld\n", i, j, x[i][j]);
    }
  }
  return 0;
}
