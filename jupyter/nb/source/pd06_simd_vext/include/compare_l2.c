#com 3
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifpy VER <= 2
double l2_norm(double * a, long n) {
  double s = 0.0;
  for (long i = 0; i < n; i++) {
    s += a[i] * a[i];
  }
  return sqrt(s);
}
#elifpy VER == 3
#include <x86intrin.h>
#if defined(__AVX512F__)
#define SIMD_WIDTH 64
enum { simd_width = 64 };       /* 512 bit = 64 bytes */
#elif defined(__AVX2__) || defined(__AVX__)
#define SIMD_WIDTH 32
enum { simd_width = 32 };       /* 256 bit = 32 bytes */
#else
#error "sorry, you must have either __AVX__, __AVX2__, or __AVX512F__"
#endif
typedef double doublev __attribute__((vector_size(simd_width),aligned(sizeof(double))));
const int L = sizeof(doublev) / sizeof(double);

#define V(p) (*((doublev*)(&p)))

#if SIMD_WIDTH == 64
doublev vzero() {
  return _mm512_set1_pd(0.0);
}

double hadd(doublev v) {
  return ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]));
}
#elif SIMD_WIDTH == 32
doublev vzero() {
  return _mm256_set1_pd(0.0);
}

double hadd(doublev v) {
  return ((v[0] + v[1]) + (v[2] + v[3]));
}
#endif

double l2_norm(double * a, long n) {
  doublev sv = vzero();
  long i = 0;
  for (i = 0; i + L <= n; i += L) {
    sv += V(a[i]) * V(a[i]);
  }
  double s = hadd(sv);
  for (; i < n; i++) {
    s += a[i] * a[i];
  }
  return sqrt(s);
}
#endifpy

int main(int argc, char ** argv) {
  int i = 1;
  long n      = (argc > i ? atof(argv[i]) : 1000L); i++;
  double * a = (double *)malloc(sizeof(double) * n);
  for (long i = 0; i < n; i++) { a[i] = 1.0; }
  double l = l2_norm(a, n);
  printf("|a| = %f\n", l);
  printf("|a| - sqrt(n) = %f\n", l - sqrt(n));
  free(a);
  return 0;
}
