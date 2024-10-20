#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "omp_gpu_util.h"

// record of execution
typedef unsigned int uint;
typedef struct {
  double x;                     // a (meaningless) answer
  uint sm0;                     // SM on which a thread got started
  uint sm1;                     // SM on which a thread ended (MUST BE = sm0; just to verify that)
} record_t;

/* this thread repeats x = a x + b (N * M) times.
   it records the clock N times (every M iterations of x = a x + b)
   to array T.
   final result of x = a x + b, as well as SM each thread was executed
   on are recorded to R. */
void iter(double a, double b, record_t * R, 
          long * T, long n, long m, long idx) {
  // initial value (not important)
  double x = idx;
  // where clocks are recorded
  T = &T[idx * n];
  // record starting SM
  R[idx].sm0 = get_smid();
  // main thing. repeat a x + b many times,
  // occasionally recording the clock
  for (long i = 0; i < n; i++) {
    T[i] = clock64();
    for (long j = 0; j < m; j++) {
      x = a * x + b;
    }
  }
  // record ending SM (must be = sm0)
  R[idx].sm1 = get_smid();
  // record result, just so that the computation is not
  // eliminated by the compiler
  R[idx].x = x;
}

int getenv_int(const char * v) {
  char * s = getenv(v);
  if (!s) {
    fprintf(stderr, "specify environment variable %s\n", v);
    exit(1);
  }
  return atoi(s);
}

/* usage
   ./omp_gpu_sched_rec N_TEAMS N_THREADS_PER_TEAM N M A B

   creates about N_THREAD_BLOCKS blocks x THREAD_BLOCK_SZ 
   threads.
   each thread repeats x = A x + B (N * M) times.
*/
int main(int argc, char ** argv) {
  int n_teams = getenv_int("OMP_NUM_TEAMS");
  int n_threads_per_team = getenv_int("OMP_NUM_TEAMS");
  int i = 1;
  long n             = (argc > i ? atol(argv[i]) : 100);   i++;
  long m             = (argc > i ? atol(argv[i]) : 1000);  i++;
  double a           = (argc > i ? atof(argv[i])  : 0.99); i++;
  double b           = (argc > i ? atof(argv[i])  : 1.00); i++;

  // allocate record_t array (both on host and device)
  long n_threads = n_teams * n_threads_per_team;
  // iter x uses R[x]
  record_t * R = (record_t *)calloc(sizeof(record_t), n_threads);
  // allocate clock array (both on host and device)
  long * T = (long *)calloc(sizeof(long), n * n_threads);

#pragma omp target teams distribute parallel for num_teams(n_teams) num_threads(n_threads_per_team) map(tofrom: R[:n_threads], T[:n * n_threads])
  for (long idx = 0; idx < n_threads; idx++) {
    iter(a, b, R, T, n, m, idx);
  }
  // dump the for visualization
  long k = 0;
  for (long idx = 0; idx < n_threads; idx++) {
    printf("thread=%ld x=%f sm0=%u sm1=%u", idx, R[idx].x, R[idx].sm0, R[idx].sm1);
    for (long i = 0; i < n; i++) {
      printf(" %ld", T[k]);
      k++;
    }
    printf("\n");
  }
  return 0;
}

