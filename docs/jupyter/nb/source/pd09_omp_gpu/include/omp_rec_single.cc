#include <stdio.h>
#include <omp.h>
#include "omp_gpu_util.h"
int main(int argc, char ** argv) {
  int i = 1;
  int m = (argc > i ? atoi(argv[i]) : 5); i++;
  long x[m];
  for (int i = 0; i < m; i++) {
    x[i] = 0;
  }
#pragma omp target teams distribute parallel for map(tofrom: x)
  for (int i = 0; i < m; i++) {
    x[i] = get_smid();
  }
  printf("size: %d\n", m);
  for (int i = 0; i < m; i++) {
    printf("x[%d] = %ld\n", i, x[i]);
  }
  return 0;
}
