#include "msort.h"

/* insertion sort for leaf cases */
/* return m a[m] = min {a[begin:end]} */ 
static long choose_min_idx(float * a, long begin, long end) {
  long m = begin;
  for (long i = begin + 1; i < end; i++) {
    if (a[i] < a[m]) {
      m = i;
    }
  }
  return m;
}

/* sort a[p:q] by insertion sort */
static void insertion_sort(float * a, long p, long q) {
  for (long i = p; i < q; i++) {
    long j = choose_min_idx(a, i, q);
    float t = a[i];
    a[i] = a[j];
    a[j] = t;
  }
}

/* sort a[p:q] into g[p:q], using b[p:q] as a temporary space */
static void msort_r(float * a, float * b, float * g, long p, long q, long th0, long th1) {
  if (q - p < th0) {
    /* the array is small -> switch to insertion sort */
    if (g != a) {
      for (long i = p; i < q; i++) {
        g[i] = a[i];
      }
    }
    insertion_sort(g, p, q);
  } else {
    long r = (p + q) / 2;
    /* get partial results into the other array != g */
    float * h = (g == a ? b : a);
#pragma omp task
    msort_r(a, b, h, p, r, th0, th1);
#pragma omp task
    msort_r(a, b, h, r, q, th0, th1);
#pragma omp taskwait
    /* merge h[p:r] and h[r:q] -> g[p:]*/
    merge(h, g, p, r, r, q, p, th1);
  }
}

/* merge sort, called from the main */
void msort_from_main(float * a, float * b, float * g, long p, long q, long th0, long th1) {
#pragma omp parallel
#pragma omp master
  msort_r(a, b, g, p, q, th0, th1);
}

