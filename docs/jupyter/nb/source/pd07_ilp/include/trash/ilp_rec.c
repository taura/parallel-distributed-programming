#com 5
#define _GNU_SOURCE
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <omp.h>
#include <x86intrin.h>
#ifpy VER == 0
/* 
   VER ==
   1 : scalar
   2 : simd
   3 : simd, using perf_event
   4 : 2 simds
   5 : C simds
 */
#endifpy
#ifpy VER >= 3
#include "perf.h"
#endifpy
// record of execution
typedef long long int llint;
#ifpy VER >= 2
typedef double doublev __attribute__((vector_size(64), __may_alias__, aligned(sizeof(double))));
enum { L = sizeof(doublev) / sizeof(double) };
#endifpy

#ifpy VER == 1
enum { K = 1 };
#elifpy VER <= 3
enum { K = L };
#elifpy VER <= 4
enum { K = 2 * L };
#elifpy VER <= 5
#ifndef C
#define C 2
#endif
enum { K = C * L };
#endifpy

typedef struct {
#ifpy VER == 1
  double x[1];
#elsepy
  double x[K];                     // a (meaningless) answer
#endifpy
  int vcore0; // a virtual core on which a thread got started
  int vcore1; // a virtual core on which a thread ended
} record_t;

#ifpy VER <= 2
llint get_clock() {
  return _rdtsc();
}
#elsepy
llint get_clock(perf_event_counter_t pc) {
#if CLOCK_IS_CORE_CLOCK
  /* get core clock */
  return perf_event_counter_get(pc);
#else
  /* read timestamp counter instruction (reference clock) */
  return _rdtsc();
#endif
}
#endifpy


#ifpy VER >= 2
#define V(x) (*((doublev*)&x))
#endifpy
/* this thread repeats x = a x + b (N * M) times.
   it records the clock N times (every M iterations of x = a x + b)
   to array T.
   final result of x = a x + b, as well as SM each thread was executed
   on are recorded to R. */
void thread_fun(double a, double b, record_t * R,
                llint * T, llint n, llint m) {
  int idx = omp_get_thread_num();
  // initial value (not important)
#ifpy VER == 1
  double x = idx;
#elsepy
  double x[K];
  for (long i = 0; i < K; i++) {
    x[i] = idx * K + i;
  }
#endifpy
#ifpy 2 <= VER <= 3
  doublev x0 = V(x[0]);
#elifpy VER == 4
  doublev x0 = V(x[0]);
  doublev x1 = V(x[L]);
#endifpy
  
  // where clocks are recorded
  T = &T[idx * n];
  // record starting SM
  R[idx].vcore0 = sched_getcpu();
  // main thing. repeat a x + b many times,
  // occasionally recording the clock
#ifpy VER >= 3
  perf_event_counter_t pc = mk_perf_event_counter();
#endifpy
  for (long i = 0; i < n; i++) {
#ifpy VER <= 2
    T[i] = get_clock();
#elsepy
    T[i] = get_clock(pc);
#endifpy
    asm volatile("# begin loop");
    for (long j = 0; j < m; j++) {
#ifpy VER == 1
      x = a * x + b;
#elifpy VER <= 3
      x0 = a * x0 + b;
#elifpy VER <= 4
      x0 = a * x0 + b;
      x1 = a * x1 + b;
#elsepy
      for (long k = 0; k < K; k += L) {
        V(x[k]) = a * V(x[k]) + b;
      }
#endifpy
    }
    asm volatile("# end loop");
  }
#ifpy VER >= 3
  perf_event_counter_destroy(pc);
#endifpy
  // record ending SM (must be = sm0)
  R[idx].vcore1 = sched_getcpu();
  // record result, just so that the computation is not
  // eliminated by the compiler
#ifpy 2 <= VER <= 3
  V(x[0]) = x0;
#elifpy VER == 4
  V(x[0]) = x0;
  V(x[L]) = x1;
#endifpy
#ifpy VER == 1
  R[idx].x[0] = x;
#elsepy
  for (int i = 0; i < K; i++) {
    R[idx].x[i] = x[i];
  }
#endifpy
}

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
