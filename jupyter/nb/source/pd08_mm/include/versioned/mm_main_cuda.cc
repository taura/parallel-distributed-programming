/* 
 * mm_main_cuda.cc
 */

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "mm_cuda.h"
#include "event.h"

void gemm(matrix A, matrix B, matrix C);

/* initialize elements randomly */
static void rand_init(matrix A, unsigned short rg[3]) {
  for (idx_t i = 0; i < A.M; i++) {
    for (idx_t j = 0; j < A.N; j++) {
      A(i,j) = erand48(rg);
    }
  }
}

/* initialize all elements by c */
static void const_init(matrix A, real c) {
  for (idx_t i = 0; i < A.M; i++) {
    for (idx_t j = 0; j < A.N; j++) {
      A(i,j) = c;
    }
  }
}

static void zero_init(matrix A) {
  const_init(A, 0.0);
}

static real comp_ij(matrix A, matrix B,
                    idx_t i, idx_t j, long times) {
  real s1 = 0.0;
  idx_t K = A.N;
  for (idx_t k = 0; k < K; k++) {
    s1 += A(i,k) * B(k,j);
  }
  real s = 0.0;
  for (long t = 0; t < times; t++) {
    s += s1;
  }
  return s;
}

int main(int argc, char ** argv) {
  int i = 1;
  const long M = (argc > i ? atol(argv[i]) :   8); i++;
  const long N = (argc > i ? atol(argv[i]) :  32); i++;
  const long K = (argc > i ? atol(argv[i]) : 192); i++;
  const long approx_fmas = (argc > i ? atol(argv[i]) : 1L * 1000L * 1000L * 1000L); i++;
  const long chk  = (argc > i ? atol(argv[i]) : 5); i++;
  const long seed = (argc > i ? atol(argv[i]) : 76843802738543); i++;

  matrix A(M, K);
  matrix B(K, N);
  matrix C(M, N);

  unsigned short rg[3] = { (unsigned short)((seed >> 16) & 65535),
			   (unsigned short)((seed >> 8)  & 65535),
			   (unsigned short)((seed >> 0)  & 65535) };
  rand_init(A, rg);
  rand_init(B, rg);
  zero_init(C);
  A.to_dev();
  B.to_dev();
  C.to_dev();
  
  const long fmas = (long)M * (long)N * (long)K;
  const long repeat = (approx_fmas + fmas - 1) / fmas;
  const long fmas_total = fmas * repeat;
  printf("M = %ld, N = %ld, K = %ld\n", M, N, K);
  printf("sizeof(real) = %ld\n", sizeof(real));
  printf("A : %ld x %ld (ld=%ld) %ld bytes\n",
         M, K, (long)A.ld, M * A.ld * sizeof(real));
  printf("B : %ld x %ld (ld=%ld) %ld bytes\n",
         K, N, (long)B.ld, K * B.ld * sizeof(real));
  printf("C : %ld x %ld (ld=%ld) %ld bytes\n",
         M, N, (long)C.ld, M * C.ld * sizeof(real));
  printf("total = %ld bytes\n",
	 (M * A.ld + K * B.ld + M * C.ld) * sizeof(real));
  printf("repeat : %ld times\n", repeat);
  printf("perform %ld fmas ... ", fmas_total); fflush(stdout);

  const char * events = getenv("EV");
  if (!events) events = "ref-cycles";
  perf_event_counters_t pc = mk_perf_event_counters(events);
  perf_event_values_t v0 = perf_event_counters_get(pc);
  long long int t0 = get_gpu_clock();
  /* real thing happens here */
  for (long i = 0; i < repeat; i++) {
    gemm(A, B, C);
  }
  long long int t1 = get_gpu_clock();
  long long int dt = t1 - t0;
  /* real thing ends here */
  perf_event_values_t v1 = perf_event_counters_get(pc);
  printf("done\n");
  C.to_host();

  /* show performance counters */
  for (int i = 0; i < pc.n; i++) {
    printf("%s : %lld\n", pc.events[i], v1.values[i] - v0.values[i]);
  }
  printf("gpu-clock : %lld\n", dt);
  for (int i = 0; i < pc.n; i++) {
    if (strcmp(pc.events[i], "cycles") == 0) {
      long dt = v1.values[i] - v0.values[i];
      printf("%f fmas/core-cycle\n", fmas_total / (double)dt);
    }
    if (strcmp(pc.events[i], "ref-cycles") == 0) {
      long dt = v1.values[i] - v0.values[i];
      printf("%f fmas/ref-cycle\n", fmas_total / (double)dt);
    }
    if (strcmp(pc.events[i], "instructions") == 0) {
      long di = v1.values[i] - v0.values[i];
      printf("%f fmas/instruction\n", fmas_total / (double)di);
    }
  }
  printf("%f fmas/gpu-clock\n", fmas_total / (double)dt);
  printf("=== checking results of randomly picked %ld elements ===\n", chk);
  for (long c = 0; c < chk; c++) {
    long i = nrand48(rg) % M;
    long j = nrand48(rg) % N;
    real s = comp_ij(A, B, i, j, repeat);
    printf("C(%ld,%ld) = %f, ans = %f, |C(%ld,%ld) - s| = %.9f\n",
	   i, j, C(i,j), s,
           i, j, fabs(C(i,j) - s));
  }
  perf_event_counters_destroy(pc);
  return 0;
}
