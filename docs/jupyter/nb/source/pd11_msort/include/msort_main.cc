#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "event.h"
#include "msort.h"

/* make a random array of n elements */
float * random_array(long n, long seed) {
  float * a = (float *)malloc(sizeof(float) * n);
  unsigned short rg[3] = { (unsigned short)((seed >> 32) & 65535),
                           (unsigned short)((seed >> 16) & 65535),
                           (unsigned short)((seed >> 0 ) & 65535) };
  for (long i = 0; i < n; i++) {
    a[i] = erand48(rg);
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
  /* n <= ms_threshold -> insertion sort */
  long ms_threshold   = (argc > i ? atol(argv[i]) : 50);        i++;
  long mg_threshold   = (argc > i ? atol(argv[i]) : 10 * 1000); i++;
  long seed           = (argc > i ? atol(argv[i]) : 12345678);  i++;
  const char * exe = argv[0];
  const char * th_s = getenv("OMP_NUM_THREADS");
  long th = (th_s ? atol(th_s) : 1);
  const char * events = getenv("EV");
  if (!events) events = strdup("ref-cycles,cycles,instructions,L1-dcache-load-misses,cache-misses");
  float * a = random_array(n, seed);
  float * b = const_array(n, 0);
  perf_event_counters_t pc = mk_perf_event_counters(events);
  printf("exe: %s\n", exe);
  printf("threads: %ld\n", th);
  printf("elements: %ld\n", n);
  printf("threshold from merge to insertion: %ld\n", ms_threshold);
  printf("threshold from parallel merge to serial merge: %ld\n", mg_threshold);
  perf_event_values_t v0 = perf_event_counters_get(pc);
  /* real thing happens here */
  msort_from_main(a, b, a, 0, n, ms_threshold, mg_threshold);
  perf_event_values_t v1 = perf_event_counters_get(pc);
  for (int i = 0; i < pc.n; i++) {
    printf("%s : %lld\n", pc.events[i], v1.values[i] - v0.values[i]);
  }
  perf_event_counters_destroy(pc);
  long us = count_unsorted(a, n);
  if (us == 0) {
    printf("OK\n");
  } else {
    printf("NG\n");
  }
  return 0;
}
