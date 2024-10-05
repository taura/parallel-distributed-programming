/*** com 8 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
/*** if 4 <= VER <= 8 */
#include <omp.h>
/*** endif */

/*** if VER == 1 */
/* scalar version */
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
/*** elif VER == 3 */
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
/*** elif VER == 4 */
double int_sqrt_one_minus_x2_y2(long n) {
  double h = 1.0 / n;
  double s = 0.0;
#pragma omp simd reduction(+:s)
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
/*** elif VER == 5 */
double int_sqrt_one_minus_x2_y2(long n) {
  double h = 1.0 / n;
  double s = 0.0;
  for (long i = 0; i < n; i++) {
    double t = 0.0;
#pragma omp simd reduction(+:t)
    for (long j = 0; j < n; j++) {
      double x = i * h ;
      double y = j * h;
      double z = 1 - x * x - y * y;
      if (z > 0.0) {
        s += sqrt(z);
      }
    }
    s += t;
  }
  return s * h * h;
}
/*** elif VER == 6 */
double int_sqrt_one_minus_x2_y2(long n) {
  double h = 1.0 / n;
  double s = 0.0;
#pragma omp simd collapse(2) reduction(+:s)
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
/*** elif 7 <= VER < 9 */
#include <x86intrin.h>
#if defined(__AVX512F__)
enum { simd_width = 64 };       /* 512 bit = 64 bytes */
#else
#error "sorry, you must have either __AVX512F__"
#endif
typedef double doublev __attribute__((vector_size(simd_width),__may_alias__,aligned(simd_width)));
enum { L = sizeof(doublev) / sizeof(double) };

#define V(p) (*((doublev*)&p))

/* {x, x+1, ..., x+L-1} */
doublev Li(double x) {
  double a[L];
  for (int i = 0; i < L; i++) a[i] = x + i;
  return V(a[0]);
}

doublev U(double x) {
  double a[L];
  for (int i = 0; i < L; i++) a[i] = x;
  return V(a[0]);
}

doublev vmax(doublev a, doublev b) {
  return _mm512_max_pd(a, b);
}

doublev vsqrt(doublev x) {
  return _mm512_sqrt_pd(x);
}

double vsum(doublev x) {
  return ((x[0] + x[1]) + (x[2] + x[3])) + ((x[4] + x[5]) + (x[6] + x[7]));
}

/*** if VER == 7 */
double int_sqrt_one_minus_x2_y2(long n) {
  double h = 1.0 / n;
  doublev s = U(0);
  for (long i = 0; i < n; i += L) {
    doublev x = Li(i) * h ;
    for (long j = 0; j < n; j++) {
      double  y = j * h;
      doublev z = 1 - x * x - y * y;
      doublev pz = vmax(z, U(0));
      if (pz[0] > 0.0) {
        s += vsqrt(pz);
      } else {
        break;
      }
    }
  }
  return vsum(s) * h * h;
}
/*** elif VER == 8 */
/* user-defined reduction on vec_t */
#pragma omp declare reduction (vplus : doublev : omp_out += omp_in) initializer(omp_priv = U(0))

double int_sqrt_one_minus_x2_y2(long n) {
  double h = 1.0 / n;
  doublev s = U(0);
#pragma omp parallel for schedule(dynamic) reduction(vplus:s)
  for (long i = 0; i < n; i += L) {
    doublev x = Li(i) * h ;
    for (long j = 0; j < n; j++) {
      double  y = j * h;
      doublev z = 1 - x * x - y * y;
      doublev pz = vmax(z, U(0));
      if (pz[0] > 0.0) {
        s += vsqrt(pz);
      } else {
        break;
      }
    }
  }
  return vsum(s) * h * h;
}
/*** elif VER == 9 */
/* GPU */

#include "cuda_util.h"

__global__ void int_sqrt_one_minus_x2_y2_kernel(long n, double * s) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  if (i < n && j < n) {
    double h = 1.0 / (double)n;
    double x = i * h;
    double y = j * h;
    double z = 1 - x * x - y * y;
    if (z > 0.0) {
      atomicAdd(s, sqrt(z) * h * h);
    }
  }
}

double int_sqrt_one_minus_x2_y2(long n) {
  int thread_block_sz_x = 32;
  int thread_block_sz_y = 32;
  int n_thread_blocks_x = (n + thread_block_sz_x - 1) / thread_block_sz_x;
  int n_thread_blocks_y = (n + thread_block_sz_y - 1) / thread_block_sz_y;
  double s = 0.0;
  double * s_dev;
  check_api_error(cudaMalloc(&s_dev, sizeof(double)));
  check_api_error(cudaMemcpy(s_dev, &s, sizeof(double), cudaMemcpyHostToDevice));
  
  dim3 nb(n_thread_blocks_x, n_thread_blocks_y);
  dim3 tpb(thread_block_sz_x, thread_block_sz_y);
  check_launch_error((int_sqrt_one_minus_x2_y2_kernel<<<nb,tpb>>>(n, s_dev)));
  check_api_error(cudaDeviceSynchronize());
  check_api_error(cudaMemcpy(&s, s_dev, sizeof(double), cudaMemcpyDeviceToHost));
  return s;
}
/*** endif */
/*** endif */

// wall clock time
double get_wtime() {
  struct timespec ts[1];
  clock_gettime(CLOCK_REALTIME, ts);
  return ts->tv_sec + ts->tv_nsec * 1.0e-9;
}

int main(int argc, char ** argv) {
  int i = 1;
  long n      = (argc > i ? atof(argv[i]) : 30L * 1000L); i++;
  long repeat = (argc > i ? atof(argv[i]) : 5);           i++;
  n += 15;
  n = (n / 16) * 16;
  double n_sqrts = (M_PI / 4) * n * n;
  for (long r = 0; r < repeat; r++) {
    printf("===== repeat %ld =====\n", r);
    fflush(stdout);
    double t0 = get_wtime();
    double s = int_sqrt_one_minus_x2_y2(n);
    double t1 = get_wtime();
    double dt = t1 - t0;
    printf("s = %.9f (err = |s - pi/6| = %e)\n", s, fabs(s - M_PI/6));
    printf("%f sec\n", dt);
    printf("%f points/sec\n", n_sqrts / dt);
    fflush(stdout);
  }
  return 0;
}
