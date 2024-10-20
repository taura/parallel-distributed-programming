#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <nv/target>

long cur_time_ns() {
  struct timespec ts[1];
  if (clock_gettime(CLOCK_REALTIME, ts) == -1) err(1, "clock_gettime");
  return ts->tv_sec * 1000000000L + ts->tv_nsec;
}

#if __NVCOMPILER
/* get SM id (for NVIDIA compiler).
   return -1 if called on CPU */
__host__ __device__ static unsigned int get_smid(void) {
  if target(nv::target::is_device) {
    unsigned int sm;
    asm("mov.u32 %0, %%smid;" : "=r"(sm));
    return sm;
  } else {
    return (unsigned int)(-1);
  }
}
#endif

#if __clang__
/* get SM id (for Clang LLVM compiler).
   return -1 if called on CPU */
__attribute__((unused))
static unsigned int get_smid(void) {
#if __CUDA_ARCH__
  unsigned int sm;
  asm("mov.u32 %0, %%smid;" : "=r"(sm));
  return sm;
#else
  return (unsigned int)(-1);
#endif
}

/* get GPU clock (for Clang LLVM compiler).
   return -1 if called on CPU */
__attribute__((unused))
static long long int clock64(void) {
#if __CUDA_ARCH__
  long long int clock;
  asm volatile("mov.s64 %0, %%clock64;" : "=r" (clock));
  return clock;
#else
  return (unsigned int)(-1);
#endif
}
#endif

typedef struct {
  double x;
  int team[2];
  int thread[2];
  int sm[2];
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
  double x = i;
  // record in T[i * M] ... T[(i+1) * M - 1]
  T = &T[i * M];
  // record starting thread/cpu
  R[i].team[0] = omp_get_team_num();
  R[i].thread[0] = omp_get_thread_num();
  R[i].sm[0] = get_smid();
  // repeat a x + b many times.
  // record time every N iterations
  for (long j = 0; j < M; j++) {
    T[j] = clock64();
    for (long k = 0; k < N; k++) {
      x = a * x + b;
    }
  }
  // record ending SM (must be = thread0)
  R[i].team[1] = omp_get_team_num();
  R[i].thread[1] = omp_get_thread_num();
  R[i].sm[1] = get_smid();
  // record result, just so that the computation is not
  // eliminated by the compiler
  R[i].x = x;
}

void dump(record_t * R, long * T, long L, long M, long t0) {
  long k = 0;
  for (long i = 0; i < L; i++) {
    printf("i=%ld x=%f team0=%d thread0=%d sm0=%d team1=%d thread1=%d sm1=%d",
           i, R[i].x,
           R[i].team[0], R[i].thread[0], R[i].sm[0],
           R[i].team[1], R[i].thread[1], R[i].sm[1]);
    for (long j = 0; j < M; j++) {
      printf(" %ld", T[k] - t0);
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
  long t0 = cur_time_ns();
#pragma omp target teams distribute parallel for num_teams(n_teams) num_threads(n_threads_per_team)
  for (long i = 0; i < L; i++) {
    iter_fun(a, b, i, M, N, R, T);
  }
  long t1 = cur_time_ns();
  printf("%ld nsec\n", t1 - t0);
  dump(R, T, L, M, t0);
  return 0;
}

