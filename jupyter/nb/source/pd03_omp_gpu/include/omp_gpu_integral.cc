/*** com 2 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*** if VER == 1 */
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
/*** elif VER == 2 */
/* serial version */
double int_sqrt_one_minus_x2_y2(long n, int n_teams, int n_threads_per_team) {
  double h = 1.0 / n;
  double s = 0.0;
#pragma omp target teams distribute parallel for collapse(2) num_teams(n_teams) num_threads(n_threads_per_team) reduction(+:s)
  for (long i = 0; i < n; i++) {
    for (long j = 0; j < n; j++) {
      double x = i * h ;
      double y = j * h;
      double z = 1 - x * x - y * y;
      if (z > 0.0) {
        s += sqrt(z);
      }
    }
  }
  return s * h * h;
}
/*** endif */

int getenv_int(const char * v) {
  char * s = getenv(v);
  if (!s) {
    fprintf(stderr, "specify environment variable %s\n", v);
    exit(1);
  }
  return atoi(s);
}

int main(int argc, char ** argv) {
  int i = 1;
  long n                  = (argc > i ? atol(argv[i]) : 30L * 1000L); i++;
  int n_teams = getenv_int("OMP_NUM_TEAMS");
  int n_threads_per_team = getenv_int("OMP_NUM_THREADS");
  printf("n = %ld (%ld points to evaluate integrand on)\n", n, n * n);
  printf("n_teams = %d\n", n_teams);
  printf("n_threads_per_team = %d\n", n_threads_per_team);
/*** if VER == 1 */
  double s = int_sqrt_one_minus_x2_y2(n);
/*** elif VER == 2 */
  double s = int_sqrt_one_minus_x2_y2(n, n_teams, n_threads_per_team);
/*** endif */
  printf("s = %.9f (err = %e)\n", s, fabs(s - M_PI/6));
  return 0;
}
