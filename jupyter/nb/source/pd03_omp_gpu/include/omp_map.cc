#include <stdio.h>
#include <omp.h>
int main(int argc, char ** argv) {
  int i = 1;
  int m = (argc > i ? atoi(argv[i]) : 5); i++;
  float x[m];
  for (int i = 0; i < m; i++) {
    x[i] = i + 1;
  }
#pragma omp target teams distribute parallel for map(tofrom: x[0:m])
  for (int i = 0; i < m; i++) {
    x[i] = x[i] * x[i];
  }
  for (int i = 0; i < m; i++) {
    printf("x[%d] = %.1f\n", i, x[i]);
  }
  return 0;
}
