#com 1
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <x86intrin.h>

#include "binsearch_nosimd.h"
#include "binsearch_simd_util.h"
#include "binsearch_simd.h"

// make a random array of n elements, each element in [0:max_val)
int * make_random_array(int n, int max_val, unsigned long seed) {
  int * a = (int *)malloc(sizeof(int) * n);
  assert(a);
  unsigned short rg[3] = {
    (unsigned short)(seed >> 32),
    (unsigned short)(seed >> 16),
    (unsigned short)(seed >>  0)
  };
  for (int i = 0; i < n; i++) {
    a[i] = nrand48(rg) % max_val;
  }
  return a;
}

// print a[0:n] (don't call it with large n)
void print_array(int * a, int n) {
  for (int i = 0; i < n; i++) {
    printf("%d ", a[i]);
  }
  printf("\n");
}

// wall clock time
double get_wtime() {
  struct timespec ts[1];
  clock_gettime(CLOCK_REALTIME, ts);
  return ts->tv_sec + ts->tv_nsec * 1.0e-9;
}

// aux function to sort the searched array
int cmp_int(const void * _a, const void * _b) {
  const int * a = (const int *)_a;
  const int * b = (const int *)_b;
  return *a - *b;
}

int main(int argc, char ** argv) {
  int i = 1;
  int simd = (argc > i ? atoi(argv[i]) : 0); i++;
  int n = (argc > i ? atoi(argv[i]) : 10 * 1000 * 1000); i++;
  int m = (argc > i ? atoi(argv[i]) :  5 * 1000 * 1000); i++;
  // max_val = 3n => approximately 1/3 of values will be found
  int max_val = (argc > i ? atoi(argv[i]) : 3 * n); i++;
  unsigned long seed0 = (argc > i ? atol(argv[i]) : 1234567890L); i++;
  unsigned long seed1 = (argc > i ? atol(argv[i]) : 2345678909L); i++;

  // generate array A (to be searched)
  printf("generating array A to be searched (%d elements in range 0..%d) ...\n", n, max_val); fflush(stdout);
  int * a = make_random_array(n, max_val, seed0);
  printf("sorting ...\n"); fflush(stdout);
  qsort(a, n, sizeof(int), cmp_int);
  //print_array(a, n);

  // generate array X (values to search for)
  printf("generating values X to search for (%d elements in range 0..%d) ...\n", m, max_val); fflush(stdout);
  int * x = make_random_array(m, max_val, seed1);
  //print_array(x, m);

  // start the real thing
  printf("start binary search ...\n"); fflush(stdout);
  double t0 = get_wtime();
  int c = (simd == 0 ?
           binsearch_many(a, n, x, m) :
           binsearch_many_simd(a, n, x, m));
  double t1 = get_wtime();
  printf("done in %f sec\n", t1 - t0);
  printf("%d out of %d elements found in the array\n", c, m);
  free(a);
  free(x);
  return 0;
}
