/*** com 2 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>

/*** if VER == 1 */
long collatz(long a, long b, long n) {
  long s = 0;
  for (long i = a; i < b; i++) {
    long x = i;
    for (long j = 0; j < n; j++) {
      x = (x % 2 == 0 ? x / 2 : 3 * x + 1);
    }
    if (x != 1 && x != 2 && x != 4) {
      s++;
    }
  }
  return s;
}
/*** elif VER == 2 */
typedef long _longv __attribute__((vector_size(64), __may_alias__, aligned(sizeof(long))));
struct longv {
  _longv v;
  longv(_longv _v) { v = _v; }
  longv(long _v) { v = _mm512_set1_epi64(_v); }
};
enum { n_lanes = sizeof(_longv) / sizeof(long) };

__mmask8 operator==(longv a, longv b) {
  return _mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_EQ);
}

__mmask8 operator==(longv a, long b) {
  return a == longv(b);
}

__mmask8 operator!=(longv a, longv b) {
  return _mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NE);
}

__mmask8 operator!=(longv a, long b) {
  return a != longv(b);
}

__mmask8 operator<(longv a, longv b) {
  return _mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_LE);
}

__mmask8 operator<(longv a, long b) {
  return a < longv(b);
}

longv operator&(longv a, longv b) {
  return longv(a.v & b.v);
}

longv operator+(longv a, longv b) {
  return longv(a.v + b.v);
}

longv operator*(longv a, longv b) {
  return longv(a.v * b.v);
}

longv operator/(longv a, longv b) {
  return longv(a.v / b.v);
}

longv blend(__mmask8 k, longv a, longv b) {
  return longv(_mm512_mask_blend_epi64(k, b.v, a.v));
}

longv lin(long a) {
  long v[n_lanes];
  for (long i = 0; i < n_lanes; i++) {
    v[i] = a + i;
  }
  return *((_longv*)v);
}

long count_one(__mmask8 k) {
  return _mm_popcnt_u32((unsigned int)k);
}

long collatz(long a, long b, long n) {
  long s = 0;
  for (long i = a; i < b; i += n_lanes) {
    longv x = lin(i);
    for (long j = 0; j < n; j++) {
      x = blend((x & 1L) == 0, x / 2, 3 * x + 1);
    }
    __mmask8 k = (x != 1) & (x != 2) & (x != 4) & (lin(i) < b);
    s += count_one(k);
  }
  return s;
}
/*** endif */

int main(int argc, char ** argv) {
  int i = 1;
  long a = (argc > i ? atol(argv[i]) : 1); i++;
  long b = (argc > i ? atol(argv[i]) : 1024 * 1024); i++;
  long n = (argc > i ? atol(argv[i]) : 256); i++;

  double t0 = omp_get_wtime();
  long c = collatz(a, b, n);
  double t1 = omp_get_wtime();
  printf("%f sec\n", t1 - t0);
  printf("answer = %ld\n", c);
  return 0;
}
