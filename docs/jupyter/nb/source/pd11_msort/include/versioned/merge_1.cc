#include <assert.h>
#include "msort.h"

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

