#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <x86intrin.h>

typedef long longv __attribute__((vector_size(64), __may_alias__, aligned(sizeof(long))));
enum { n_lanes = sizeof(longv) / sizeof(long) };

longv& V(long& a) {
  return *((longv *)&a);
}

long f(long x) {
  if (x % 2 == 0) {
    return x / 2;
  } else {
    return 3 * x + 1;
  }
}

longv set1(long x) {
  return _mm512_set1_epi64(x);
}

__mmask8 eq(longv a, longv b) {
  return _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_EQ);
}

longv if_then_else(__mmask8 k, longv a, longv b) {
  return _mm512_mask_blend_epi64(k, b, a);  
}

longv f(longv x) {
  __mmask8 k = eq(x & 1L, set1(0));
  return if_then_else(k, x / 2, 3 * x + 1);
}

void collatz_no_simd(long * a, long m, long n) {
  for (long i = 0; i < m; i += n_lanes) {
    for (long j = 0; j < n; j++) {
      a[i] = f(a[i]);
    }
  }
}

void collatz_simd(long * a, long m, long n) {
  for (long i = 0; i < m; i += n_lanes) {
    for (long j = 0; j < n; j++) {
      V(a[i]) = f(V(a[i]));
    }
  }
}

void check_v(long * a, long n) {
  long ng = 0;
  for (long i = 0; i < n; i++) {
    if (a[i] != 1 && a[i] != 2 && a[i] != 4) {
      ng++;
      printf("NG: a[%ld] = %ld\n", i, a[i]);
    }
  }
  if (ng == 0) printf("OK\n");
}

int main(int argc, char ** argv) {
  int i = 1;
  long m = (argc > i ? atol(argv[i]) : 4096); i++;
  long n = (argc > i ? atol(argv[i]) : 1024 * 1024); i++;
  long seed = (argc > i ? atol(argv[i]) : 123456789L); i++;
  unsigned short rg[3] = {
    (unsigned short)((seed >> 16) & 0xFFFF),
    (unsigned short)((seed >>  8) & 0xFFFF),
    (unsigned short)((seed >>  0) & 0xFFFF) };

  m = (m / n_lanes) * n_lanes;
  long a[m];
  for (long i = 0; i < m; i++) {
    a[i] = 2 * nrand48(rg) + 1;
  }
  double t0 = omp_get_wtime();
  collatz_simd(a, m, n);
  double t1 = omp_get_wtime();
  printf("%f sec\n", t1 - t0);
  check_v(a, m);
  return 0;
}

