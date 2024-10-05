/*** com 5 */
// record of execution
typedef long long int llint;
/*** if VER >= 2 */
typedef double doublev __attribute__((vector_size(64), __may_alias__, aligned(sizeof(double))));
enum { L = sizeof(doublev) / sizeof(double) };
/*** endif */

/*** if VER == 1 */
enum { K = 1 };
/*** elif VER <= 3 */
enum { K = L };
/*** elif VER <= 4 */
enum { K = 2 * L };
/*** elif VER <= 5 */
#ifndef C
#define C 2
#endif
enum { K = C * L };
/*** endif */

typedef struct {
/*** if VER == 1 */
  double x[1];
/*** else */
  double x[K];                     // a (meaningless) answer
/*** endif */
  int vcore0; // a virtual core on which a thread got started
  int vcore1; // a virtual core on which a thread ended
} record_t;

#include "ilp_rec_main.h"
/*** if VER >= 3 */
#include "perf.h"
/*** endif */

/*** if VER <= 2 */
llint get_clock() {
  return _rdtsc();
}
/*** else */
llint get_clock(perf_event_counter_t pc) {
#if CLOCK_IS_CORE_CLOCK
  /* get core clock */
  return perf_event_counter_get(pc);
#else
  /* read timestamp counter instruction (reference clock) */
  return _rdtsc();
#endif
}
/*** endif */

/*** if VER >= 2 */
#define V(x) (*((doublev*)&x))
/*** endif */
/* this thread repeats x = a x + b (N * M) times.
   it records the clock N times (every M iterations of x = a x + b)
   to array T.
   final result of x = a x + b, as well as SM each thread was executed
   on are recorded to R. */
void thread_fun(double a, double b, record_t * R,
                llint * T, llint n, llint m) {
  int idx = omp_get_thread_num();
  // initial value (not important)
/*** if VER == 1 */
  double x = idx;
/*** else */
  double x[K];
  for (long i = 0; i < K; i++) {
    x[i] = idx * K + i;
  }
/*** endif */
/*** if 2 <= VER <= 3 */
  doublev x0 = V(x[0]);
/*** elif VER == 4 */
  doublev x0 = V(x[0]);
  doublev x1 = V(x[L]);
/*** endif */
  
  // where clocks are recorded
  T = &T[idx * n];
  // record starting SM
  R[idx].vcore0 = sched_getcpu();
  // main thing. repeat a x + b many times,
  // occasionally recording the clock
/*** if VER >= 3 */
  perf_event_counter_t pc = mk_perf_event_counter();
/*** endif */
  for (long i = 0; i < n; i++) {
/*** if VER <= 2 */
    T[i] = get_clock();
/*** else */
    T[i] = get_clock(pc);
/*** endif */
    asm volatile("# begin loop");
    for (long j = 0; j < m; j++) {
/*** if VER == 1 */
      x = a * x + b;
/*** elif VER <= 3 */
      x0 = a * x0 + b;
/*** elif VER <= 4 */
      x0 = a * x0 + b;
      x1 = a * x1 + b;
/*** else */
      for (long k = 0; k < K; k += L) {
        V(x[k]) = a * V(x[k]) + b;
      }
/*** endif */
    }
    asm volatile("# end loop");
  }
/*** if VER >= 3 */
  perf_event_counter_destroy(pc);
/*** endif */
  // record ending SM (must be = sm0)
  R[idx].vcore1 = sched_getcpu();
  // record result, just so that the computation is not
  // eliminated by the compiler
/*** if 2 <= VER <= 3 */
  V(x[0]) = x0;
/*** elif VER == 4 */
  V(x[0]) = x0;
  V(x[L]) = x1;
/*** endif */
/*** if VER == 1 */
  R[idx].x[0] = x;
/*** else */
  for (int i = 0; i < K; i++) {
    R[idx].x[i] = x[i];
  }
/*** endif */
}

