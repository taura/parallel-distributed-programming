#com 2
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifpy VER == 1
/* serial version */
double int_sqrt_one_minus_x2_y2(long n) {
  double h = 1.0 / n;
  double s = 0.0;
  for (long i = 0; i < n; i++) {
    for (long j = 0; j < n; j++) {
      double x = i * h ;
      double y = j * h;
      double z = 1 - x * x - y * y;
      if (z > 0.0) {
        s += sqrt(z);
      } else {
        break;
      }
    }
  }
  return s * h * h;
}
#elifpy VER == 2
/* serial version */
double int_sqrt_one_minus_x2_y2(long n) {
  double h = 1.0 / n;
  double s = 0.0;
#pragma omp parallel for reduction(+:s) schedule(dynamic)
  for (long i = 0; i < n; i++) {
    for (long j = 0; j < n; j++) {
      double x = i * h ;
      double y = j * h;
      double z = 1 - x * x - y * y;
      if (z > 0.0) {
        s += sqrt(z);
      } else {
        break;
      }
    }
  }
  return s * h * h;
}
#endifpy

int main(int argc, char ** argv) {
  int i = 1;
  long n      = (argc > i ? atof(argv[i]) : 30L * 1000L); i++;
  double s = int_sqrt_one_minus_x2_y2(n);
  printf("s = %.9f (err = %e)\n", s, fabs(s - M_PI/6));
  return 0;
}
