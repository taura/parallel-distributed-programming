#define _GNU_SOURCE
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <omp.h>
#include <x86intrin.h>

void dump_result(record_t * R, llint * T, int nthreads, llint n) {
  // dump for visualization
  long k = 0;
  for (long idx = 0; idx < nthreads; idx++) {
    printf("thread=%ld vcore0=%u vcore1=%u x=%f", idx, R[idx].vcore0, R[idx].vcore1, R[idx].x[0]);
    for (long i = 1; i < K; i++) {
      printf(",%f", R[idx].x[i]);
    }
    for (long i = 0; i < n; i++) {
      printf(" %lld", T[k]);
      k++;
    }
    printf("\n");
  }
}

void show_performance(record_t * R, llint * T, int nthreads, llint n, llint m) {
  // dump for visualization
  llint min_t = -1;
  llint max_t = -1;
  for (long idx = 0, start_idx = 0, end_idx = n - 1;
       idx < nthreads;
       idx++, start_idx += n, end_idx += n) {
    if (min_t == -1 || T[start_idx] < min_t) {
      min_t = T[start_idx];
    }
    if (max_t == -1 || T[end_idx] > max_t) {
      max_t = T[end_idx];
    }
    double cycles = T[end_idx] - T[start_idx];
    double cycles_per_iter = cycles / ((n - 1) * m);
    double fmas = (n - 1) * m * K;
    fprintf(stderr, "thread %ld : cycles/iter = %f, fmas/cycle = %f\n",
            idx, cycles_per_iter, fmas / cycles);
  }
  double fmas = (n - 1) * m * K * nthreads;
  fprintf(stderr, "fmas/cycle = %f\n", fmas / (double)(max_t - min_t));
}

void thread_fun(double a, double b, record_t * R,
                llint * T, llint n, llint m);

/* usage
   ./ilp_rec NTHREADS N M A B

   creates about NTHREADS threads, with THREAD_BLOCK_SZ
   threads in each thread block. 
   each thread repeats x = A x + B (N * M) times.

   S is the shared memory allocated for each thread block
   (just to control the number of thread blocks simultaneously
   scheduled on an SM). shared memory is not actually used at all.
 */
int main(int argc, char ** argv) {
  int i = 1;
  llint n             = (argc > i ? atoll(argv[i]) : 100);  i++;
  llint m             = (argc > i ? atoll(argv[i]) : 1000000);  i++;
  double a            = (argc > i ? atof(argv[i])  : 0.99); i++;
  double b            = (argc > i ? atof(argv[i])  : 1.00); i++;

  int nthreads        = omp_get_max_threads();
  // allocate record_t array (both on host and device)
  long R_sz = sizeof(record_t) * nthreads;
  record_t * R = (record_t *)calloc(R_sz, 1);

  // allocate clock array (both on host and device)
  long T_sz = sizeof(llint) * n * nthreads;
  llint * T = (llint *)calloc(T_sz, 1);

#pragma omp parallel
  {
    thread_fun(a, b, R, T, n, m);
  }
  dump_result(R, T, nthreads, n);
  show_performance(R, T, nthreads, n, m);
  return 0;
}
