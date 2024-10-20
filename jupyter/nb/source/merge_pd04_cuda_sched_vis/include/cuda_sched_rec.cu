/*** com 3 */
#include <assert.h>
#include <stdio.h>

// error check utility (check_api_error and check_launch_error)
#include "cuda_util.h"

// record of execution
typedef long long int llint;
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
__global__ void cuda_thread_fun(double a, double b, record_t * R,
                                llint * T, llint n, llint m,
/*** if VER == 3 */
                                int D,
/*** endif */
                                int nthreads) {
  // my thread index
  int idx      = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= nthreads) return;
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
/*** if VER == 3 */
    if ((idx / D) % 2 == 0) {
      for (long j = 0; j < m; j++) {
        x = a * x + b;
      }
    }
/*** else */
    for (long j = 0; j < m; j++) {
      x = a * x + b;
    }
/*** endif */
  }
  // record ending SM (must be = sm0)
  R[idx].sm1 = get_smid();
  // record result, just so that the computation is not
  // eliminated by the compiler
  R[idx].x = x;
}

/* usage
   ./cuda_sched NTHREADS THREAD_BLOCK_SZ N M S A B

   creates about NTHREADS threads, with THREAD_BLOCK_SZ
   threads in each thread block. 
   each thread repeats x = A x + B (N * M) times.

   S is the shared memory allocated for each thread block
   (just to control the number of thread blocks simultaneously
   scheduled on an SM). shared memory is not actually used at all.
 */
int main(int argc, char ** argv) {
  int i = 1;
  int nthreads        = (argc > i ? atoi(argv[i])  : 100);  i++;
  int thread_block_sz = (argc > i ? atoi(argv[i])  : 64);   i++;
  llint n             = (argc > i ? atoll(argv[i]) : 100);  i++;
  llint m             = (argc > i ? atoll(argv[i]) : 100);  i++;
  int D               = (argc > i ? atoll(argv[i]) : 1);    i++;
  int shm_sz          = (argc > i ? atoi(argv[i])  : 0);    i++;
  double a            = (argc > i ? atof(argv[i])  : 0.99); i++;
  double b            = (argc > i ? atof(argv[i])  : 1.00); i++;

  // get the required number of thread blocks
  int n_thread_blocks = (nthreads + thread_block_sz - 1) / thread_block_sz;
  printf("%d threads/block * %d blocks\n", thread_block_sz, n_thread_blocks);

  // allocate record_t array (both on host and device)
  long R_sz = sizeof(record_t) * nthreads;
  record_t * R = (record_t *)calloc(R_sz, 1);
  record_t * R_dev;
  check_api_error(cudaMalloc(&R_dev, R_sz));
  check_api_error(cudaMemcpy(R_dev, R, R_sz, cudaMemcpyHostToDevice));

  // allocate clock array (both on host and device)
  long T_sz = sizeof(llint) * n * nthreads;
  llint * T = (llint *)calloc(T_sz, 1);
  llint * T_dev;
  check_api_error(cudaMalloc(&T_dev, T_sz));
  check_api_error(cudaMemcpy(T_dev, T, T_sz, cudaMemcpyHostToDevice));

  // call the kernel
  int shm_elems = shm_sz / sizeof(double);
  int shm_size = shm_elems * sizeof(double);
/*** if VER == 3 */
  check_launch_error((cuda_thread_fun<<<n_thread_blocks,thread_block_sz,shm_size>>>
                      (a, b, R_dev, T_dev, n, m, D, nthreads)));
/*** else */
  check_launch_error((cuda_thread_fun<<<n_thread_blocks,thread_block_sz,shm_size>>>
                      (a, b, R_dev, T_dev, n, m, nthreads)));
/*** endif */
  cudaDeviceSynchronize();

  // get back the results and clocks
  check_api_error(cudaMemcpy(R, R_dev, R_sz, cudaMemcpyDeviceToHost));
  check_api_error(cudaMemcpy(T, T_dev, T_sz, cudaMemcpyDeviceToHost));
  // dump the for visualization
  long k = 0;
  for (long idx = 0; idx < nthreads; idx++) {
    printf("thread=%ld x=%f sm0=%u sm1=%u", idx, R[idx].x, R[idx].sm0, R[idx].sm1);
    for (long i = 0; i < n; i++) {
      printf(" %lld", T[k]);
      k++;
    }
    printf("\n");
  }
  return 0;
}
