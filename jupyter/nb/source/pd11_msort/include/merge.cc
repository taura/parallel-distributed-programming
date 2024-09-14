#com 2
#include <assert.h>
#include "msort.h"

#ifpy VER == 1
void merge(float * a, float * b, long p, long q, long s, long t, long d, long th) {
  (void)th;
  long i = p;
  long j = s;
  long k = d;
  while (i < q && j < t) {
    if (a[i] < a[j]) {
      b[k++] = a[i++];
    } else {
      b[k++] = a[j++];
    }
  }
  while (i < q) {
    b[k++] = a[i++];
  }
  while (j < t) {
    b[k++] = a[j++];
  }
}

/* merge, called from main */
void merge_from_main(float * a, float * b, long p, long q, long s, long t, long d, long th) {
  merge(a, b, p, q, s, t, d, th);
}

#elsepy

/* merge a[p:q] and a[r:s] --> b[d:] */
static void merge_s(float * a, float * b, long p, long q, long s, long t, long d) {
  long i = p;
  long j = s;
  long k = d;
  while (i < q && j < t) {
    if (a[i] < a[j]) {
      b[k++] = a[i++];
    } else {
      b[k++] = a[j++];
    }
  }
  while (i < q) {
    b[k++] = a[i++];
  }
  while (j < t) {
    b[k++] = a[j++];
  }
}

/* find r s.t. a[r-1] <= piv < a[r] */
static long find_piv(float * a, long b, long e, float piv) {
  if (piv < a[b]) return b;
  if (a[e - 1] <= piv) return e;
  long p = b;
  long q = e - 1;
  /* a[p] <= piv < a[q] */
  while (q - p > 1) {
    assert(a[p] <= piv);
    assert(piv < a[q]);
    long r = (p + q) / 2;
    if (a[r] <= piv) {
      p = r;                    // a[p] <= piv
    } else {
      q = r;                    // piv < a[q]
    }
  }
  assert(a[q - 1] <= piv);
  assert(piv < a[q]);
  /* a[q - 1] <= piv < a[q] */
  return q;
}

/* merge a[p:q] and a[r:s] --> b[d:] */
void merge(float * a, float * b, long p, long q, long s, long t, long d, long th) {
  if (q - p + t - s < th) {
    merge_s(a, b, p, q, s, t, d);
  } else if (q - p > t - s) {
    long r = (p + q) / 2;
    float piv = a[r];
    long u = find_piv(a, s, t, piv);
#pragma omp task
    merge(a, b, p, r, s, u, d, th);
#pragma omp task
    merge(a, b, r, q, u, t, d + (r - p) + (u - s), th);
#pragma omp taskwait
  } else {
    long u = (s + t) / 2;
    float piv = a[u];
    long r = find_piv(a, p, q, piv);
#pragma omp task
    merge(a, b, p, r, s, u, d, th);
#pragma omp task
    merge(a, b, r, q, u, t, d + (r - p) + (u - s), th);
#pragma omp taskwait
  }
}

/* merge, called from main */
void merge_from_main(float * a, float * b, long p, long q, long s, long t, long d, long th) {
#pragma omp parallel
#pragma omp master
  merge(a, b, p, q, s, t, d, th);
}

#endifpy
