#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "event.h"
#include "msort.h"

float * random_sorted_two_arrays(long n, long seed) {
  float * a = (float *)malloc(sizeof(float) * n);
  unsigned short rg[3] = { (unsigned short)((seed >> 32) & 65535),
                           (unsigned short)((seed >> 16) & 65535),
                           (unsigned short)((seed >> 0 ) & 65535) };
  long h = n / 2;
  float s = 0.0;
  for (long i = 0; i < h; i++) {
    s += erand48(rg) / h;
    a[i] = s;
  }
  s = 0.0;
  for (long i = h; i < n; i++) {
    s += erand48(rg) / h;
    a[i] = s;
  }
  return a;
}

/* make a random array of n elements */
float * const_array(long n, float c) {
  float * a = (float *)malloc(sizeof(float) * n);
  for (long i = 0; i < n; i++) {
    a[i] = c;
  }
  return a;
}

/* check if a[0:n] is sorted */
int count_unsorted(float * a, long n) {
  int err = 0;
  for (long i = 0; i < n - 1; i++) {
    if (a[i] > a[i+1]) {
      fprintf(stderr, "a[%ld] = %f > a[%ld] = %f\n",
              i, a[i], i + 1, a[i + 1]);
      err++;
    }
    assert(a[i] <= a[i+1]);
  }
  return err;
}

int main(int argc, char ** argv) {
  int i = 1;
  long n              = (argc > i ? atol(argv[i]) : 100 * 1000 * 1000); i++;
  long mg_threshold   = (argc > i ? atol(argv[i]) :  10 * 1000);        i++;
  long seed           = (argc > i ? atol(argv[i]) : 12345678); i++;
  const char * events = getenv("EV");
  if (!events) events = strdup("ref-cycles,cycles,instructions,L1-dcache-load-misses,cache-misses");
  long h = n / 2;
  float * a = random_sorted_two_arrays(n, seed);
  float * b = const_array(n, 0);
  perf_event_counters_t pc = mk_perf_event_counters(events);
  printf("merge %ld + %ld elements\n", h, n - h);
  perf_event_values_t v0 = perf_event_counters_get(pc);
  /* real thing happens here */
  merge_from_main(a, b, 0, h, h, n, 0, mg_threshold);
  perf_event_values_t v1 = perf_event_counters_get(pc);
  for (int i = 0; i < pc.n; i++) {
    printf("%s : %lld\n", pc.events[i], v1.values[i] - v0.values[i]);
  }
  perf_event_counters_destroy(pc);
  long us = count_unsorted(b, n);
  if (us == 0) {
    printf("OK\n");
  } else {
    printf("NG\n");
  }
  return 0;
}
