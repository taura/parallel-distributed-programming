/*** com 9 */
#include <err.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

/*** if False */
/*
  1 : serial
  2 : outerloop
  3 : outerloop + record
  4 : collapse
  5 : collapse + record
  6 : taskloop
  7 : taskloop + record
  8 : task
  9 : task + record
 */
/*** endif */

long cur_time() {
  struct timespec ts[1];
  clock_gettime(CLOCK_REALTIME, ts);
  return ts->tv_sec * 1000000000L + ts->tv_nsec;
}

typedef struct {
  long t;
  int thread;
} record_t;

enum { A = 32 };

/*** if VER > 1 and VER % 2 == 1 */
void rec(record_t * r) {
  r->t = cur_time();
  r->thread = omp_get_thread_num();
}

void dump(const char * filename, long n, record_t R[(n+A-1)/A][(n+A-1)/A]) {
  FILE * wp = fopen(filename, "w");
  if (!wp) err(1, "fopen");
  for (long i = 0; i < n; i += A) {
    for (long j = 0; j < n; j += A) {
      fprintf(wp, "i = %ld j = %ld t = %ld thread = %d\n",
              i, j, R[i/A][j/A].t, R[i/A][j/A].thread);
    }
  }
  fclose(wp);
}
/*** endif */

/*** if VER == 1 */
/* serial version */
double int_sqrt_one_minus_x2_y2(long n, record_t R[(n+A-1)/A][(n+A-1)/A]) {
  (void)R;
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
/*** elif VER in [2,3] */
/* parallel for outerloop */
double int_sqrt_one_minus_x2_y2(long n, record_t R[(n+A-1)/A][(n+A-1)/A]) {
  (void)R;
  double h = 1.0 / n;
  double s = 0.0;
#pragma omp parallel for reduction(+:s) schedule(runtime)
  for (long i = 0; i < n; i++) {
    for (long j = 0; j < n; j++) { 
/*** if VER % 2 == 1 */
      if (i % A == 0 && j % A == 0) { rec(&R[i/A][j/A]); }
/*** endif */
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
/*** elif VER in [4,5] */
/* parallel for both loops */
double int_sqrt_one_minus_x2_y2(long n, record_t R[(n+A-1)/A][(n+A-1)/A]) {
  (void)R;
  double h = 1.0 / n;
  double s = 0.0;
#pragma omp parallel for collapse(2) reduction(+:s) schedule(runtime)
  for (long i = 0; i < n; i++) {
    for (long j = 0; j < n; j++) {
/*** if VER % 2 == 1 */
      if (i % A == 0 && j % A == 0) { rec(&R[i/A][j/A]); }
/*** endif */
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
/*** elif VER in [6,7] */
/* taskloops */
double int_sqrt_one_minus_x2_y2(long n, record_t R[(n+A-1)/A][(n+A-1)/A]) {
  (void)R;
  double h = 1.0 / n;
  double s = 0.0;
#pragma omp parallel
#pragma omp master
#pragma omp taskloop collapse(2) reduction(+:s)
  for (long i = 0; i < n; i++) {
    for (long j = 0; j < n; j++) {
/*** if VER % 2 == 1 */
      if (i % A == 0 && j % A == 0) { rec(&R[i/A][j/A]); }
/*** endif */
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
/*** elif VER in [8,9] */
/* task */
typedef struct {
  long x0;
  long y0;
  long dx;
  long dy;
} reg_t;
enum { threshold = 10000 };

double int_sqrt_one_minus_x2_y2_rec(long n, reg_t r,
                                    record_t R[(n+A-1)/A][(n+A-1)/A]) {
  (void)R;
  if (r.dx * r.dy < threshold) {
    double h = 1.0 / n;
    double s = 0.0;
    for (long i = r.x0; i < r.x0 + r.dx; i++) {
      for (long j = r.y0; j < r.y0 + r.dy; j++) {
/*** if VER % 2 == 1 */
        if (i % A == 0 && j % A == 0) { rec(&R[i/A][j/A]); }
/*** endif */
        double x = i * h ;
        double y = j * h;
        double z = 1 - x * x - y * y;
        if (z > 0.0) {
          s += sqrt(z);
        }
      }
    }
    return s * h * h;
  } else if (r.dy < r.dx) {
    long dx = r.dx;
    reg_t r0 = { r.x0,          r.y0, dx / 2,          r.dy };
    reg_t r1 = { r.x0 + dx / 2, r.y0, r.dx - dx / 2, r.dy };
    double s0 = 0.0;
    double s1 = 0.0;
#pragma omp task shared(s0)
    s0 = int_sqrt_one_minus_x2_y2_rec(n, r0, R);
#pragma omp task shared(s1)
    s1 = int_sqrt_one_minus_x2_y2_rec(n, r1, R);
#pragma omp taskwait
    return s0 + s1;
  } else {
    long dy = r.dy;
    reg_t r0 = { r.x0, r.y0,          r.dx, dy / 2 };
    reg_t r1 = { r.x0, r.y0 + dy / 2, r.dx, r.dy - dy / 2 };
    double s0 = 0.0;
    double s1 = 0.0;
#pragma omp task shared(s0)
    s0 = int_sqrt_one_minus_x2_y2_rec(n, r0, R);
#pragma omp task shared(s1)
    s1 = int_sqrt_one_minus_x2_y2_rec(n, r1, R);
#pragma omp taskwait
    return s0 + s1;
  }
}

double int_sqrt_one_minus_x2_y2(long n, record_t R[(n+A-1)/A][(n+A-1)/A]) {
  reg_t r = { 0, 0, n, n };
  double s = 0.0;
#pragma omp parallel
#pragma omp master
  s = int_sqrt_one_minus_x2_y2_rec(n, r, R);
  return s;
}
/*** endif */

int main(int argc, char ** argv) {
  int i = 1;
  long n      = (argc > i ? atof(argv[i]) : 30L * 1000L); i++;
  printf("n = %ld (%ld points to evaluate integrand on)\n", n, n * n);
/*** if VER > 1 and VER % 2 == 1 */
  record_t * R_ = (record_t *)calloc((n+A-1)/A * (n+A-1)/A, sizeof(record_t));
  record_t (*R)[] = (record_t (*)[])R_;
/*** else*/
  record_t (*R)[] = (record_t (*)[])0;
/*** endif */
  long t0 = cur_time();
  double s = int_sqrt_one_minus_x2_y2(n, R);
  long t1 = cur_time();
  long dt = t1 - t0;
  printf("%.3f sec\n", dt * 1.0e-9);
  printf("s = %.9f (err = %e)\n", s, fabs(s - M_PI/6));
/*** if VER > 1 and VER % 2 == 1 */
  dump("sched.dat", n, R);
/*** endif */
  return 0;
}
