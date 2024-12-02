#include <assert.h>
#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <sched.h>
#include <stdint.h>
#include <x86intrin.h>

long cur_time_ns() {
  struct timespec ts[1];
  if (clock_gettime(CLOCK_REALTIME, ts) == -1) err(1, "clock_gettime");
  return ts->tv_sec * 1000000000L + ts->tv_nsec;
}

#if __NVCOMPILER
#include <nv/target>
/* get core number (SM id for GPU). */
__host__ __device__ static unsigned int get_core(void) {
  if target(nv::target::is_device) {
    unsigned int sm;
    asm("mov.u32 %0, %%smid;" : "=r"(sm));
    return sm;
  } else {
    return sched_getcpu();
  }
}

/* get GPU/CPU clock (for Clang LLVM compiler) */
__attribute__((unused,nothrow))
static long get_clock(void) {
  if target(nv::target::is_device) {
    long clock;
    asm volatile("mov.s64 %0, %%clock64;" : "=l" (clock));
    return clock;
  } else {
    uint32_t low, high;
    asm volatile("rdtsc" : "=a"(low), "=d"(high));
    return ((uint64_t)high << 32) | low;
  }
}

#else  // __clang__ or GCC
/* get SM id (for Clang LLVM compiler).
   return -1 if called on CPU */
__attribute__((unused))
static unsigned int get_core(void) {
#if __CUDA_ARCH__
  unsigned int sm;
  asm("mov.u32 %0, %%smid;" : "=r"(sm));
  return sm;
#else
  return sched_getcpu();
#endif
}

/* get GPU/CPU clock (for Clang LLVM compiler) */
__attribute__((unused,nothrow))
static long get_clock(void) {
#if __CUDA_ARCH__
  long clock;
  asm volatile("mov.s64 %0, %%clock64;" : "=l" (clock));
  return clock;
#else
  return _rdtsc();
#endif
}
#endif

typedef struct {
  double x;
  int core[2];
} record_t;

/* the function for an iteration
   perform
   x = a x + b
   (M * N) times and record current time
   every N iterations to T.
   record thread and cpu to R.
 */

void iter_fun(double a, double b, long i, long M, long N,
              record_t * R, long * T) {
  // initial value (not important)
/*** if VER == 1 */
  double x = i;
/*** elif VER == 2 */
  double x0 = i, x1 = i + 0.5;
/*** elif VER == 3 */
#ifndef C
#error "give -DC=xxx in the command line"  
#endif  
  double x[C];
  for (int c = 0; c < C; c++) {
    x[c] = i + c / (double)C;
  }
/*** endif */
  // record in T[i * M] ... T[(i+1) * M - 1]
  T = &T[i * M];
  // record starting SM
  R[i].core[0] = get_core();
  // repeat a x + b many times.
  // record time every N iterations
  for (long j = 0; j < M; j++) {
    T[j] = get_clock();
    asm volatile("// ========== loop begins ==========");
    for (long k = 0; k < N; k++) {
/*** if VER == 1 */
      x = a * x + b;
/*** elif VER == 2 */
      x0 = a * x0 + b;
      x1 = a * x1 + b;
/*** elif VER == 3 */
      for (int c = 0; c < C; c++) {
        x[c] = a * x[c] + b;
      }
/*** endif */
    }
    asm volatile("// ---------- loop ends ----------");
  }
  // record ending SM (must be = sm[0])
  R[i].core[1] = get_core();
  // record result, just so that the computation is not
  // eliminated by the compiler
/*** if VER == 1 */
  R[i].x = x;
/*** elif VER == 2 */
  R[i].x = (x0 + x1) / 2.0;
/*** elif VER == 3 */
  double t = 0.0;
  for (int c = 0; c < C; c++) {
    t += x[c];
  }
  R[i].x = t / C;
/*** endif */
}

void dump(record_t * R, long * T, long L, long M, long N) {
  long k = 0;
  for (long i = 0; i < L; i++) {
    long dt = T[k + M - 1] - T[k];
    double avg = (double)dt / ((double)M * (double)N);
    printf("i=%ld x=%f core0=%d core1=%d cycles_per_iter=%f",
           i, R[i].x, R[i].core[0], R[i].core[1], avg);
    for (long j = 0; j < M; j++) {
      printf(" %ld", T[k]);
      k++;
    }
    printf("\n");
  }
}

int getenv_int(const char * v) {
  char * s = getenv(v);
  if (!s) {
    fprintf(stderr, "specify environment variable %s\n", v);
    exit(1);
  }
  return atoi(s);
}

int main(int argc, char ** argv) {
  int idx = 1;
  long L   = (idx < argc ? atol(argv[idx]) : 100);  idx++;
  long M   = (idx < argc ? atol(argv[idx]) : 100);  idx++;
  long N   = (idx < argc ? atol(argv[idx]) : 100);  idx++;
  double a = (idx < argc ? atof(argv[idx]) : 0.99); idx++;
  double b = (idx < argc ? atof(argv[idx]) : 1.00); idx++;
  int n_teams = getenv_int("OMP_NUM_TEAMS");
  int n_threads_per_team = getenv_int("OMP_NUM_THREADS");
  record_t * R = (record_t *)calloc(L, sizeof(record_t));
  long * T = (long *)calloc(L * M, sizeof(long));
#pragma omp target teams distribute parallel for num_teams(n_teams) num_threads(n_threads_per_team) map(tofrom: R[0:L], T[0:L*M])
  for (long i = 0; i < L; i++) {
    iter_fun(a, b, i, M, N, R, T);
  }
  dump(R, T, L, M, N);
  return 0;
}

