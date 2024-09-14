#com 5
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

#include "ilp_rec_main.h"
#ifpy VER >= 3
#include "perf.h"
#endifpy

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

