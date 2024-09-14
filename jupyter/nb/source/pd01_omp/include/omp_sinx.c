#com 4
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <omp.h>

int main() {
  double a = 0.0;
  double b = M_PI / 2.0;
  int n = 10000000;
  double dx = (b - a) / n;
  double s = 0.0;
  double t0 = omp_get_wtime();
#ifpy VER == 2
#pragma omp parallel for reduction(+:s)
#else
#pragma omp parallel for
#endifpy
  for (int i = 0; i < n; i++) {
    double x = a + i * dx;
#ifpy VER == 3
#pragma omp critical
#elifpy VER == 4
#pragma omp atomic
#endifpy
    s += 0.5 * (sin(x) + sin(x + dx)) * dx;
  }
  double t1 = omp_get_wtime();
  printf("ans = %.9f in %f sec\n", s, t1 - t0);
  return 0;
}

