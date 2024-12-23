#include <assert.h>
#include <stdio.h>
#include <omp.h>
/*** if "cuda" in VER */
#include "cuda_util.h"
/*** elif "omp" in VER */
#if __NVCOMPILER                // NVIDIA nvc++
#include <nv/target>
__device__ int get_thread_index() {
  if target(nv::target::is_device) {
    unsigned int thread_idx;
    unsigned int block_idx;
    unsigned int block_dim;
    asm volatile ("mov.u32 %0, %%ntid.x;"  : "=r"(block_dim));
    asm volatile ("mov.u32 %0, %%ctaid.x;" : "=r"(block_idx));
    asm volatile ("mov.u32 %0, %%tid.x;"   : "=r"(thread_idx));
    int global_idx = thread_idx + block_idx * block_dim;
    return global_idx;
  } else {
    return omp_get_thread_num();
  }
}
__device__ int get_n_threads() {
  if target(nv::target::is_device) {
    unsigned int grid_dim;
    unsigned int block_dim;
    asm volatile ("mov.u32 %0, %%ntid.x;"  : "=r"(block_dim));
    asm volatile ("mov.u32 %0, %%nctaid.x;" : "=r"(grid_dim));
    return grid_dim * block_dim;
  } else {
    return omp_get_num_threads();
  }
}
#else  // Clang
#define __host__
#define __device__
#define __global__
__device__ int get_thread_index() {
#if __CUDA_ARCH__
    unsigned int thread_idx;
    unsigned int block_idx;
    unsigned int block_dim;
    asm volatile ("mov.u32 %0, %%ntid.x;"  : "=r"(block_dim));
    asm volatile ("mov.u32 %0, %%ctaid.x;" : "=r"(block_idx));
    asm volatile ("mov.u32 %0, %%tid.x;"   : "=r"(thread_idx));
    int global_idx = thread_idx + block_idx * block_dim;
    return global_idx;
#else
    return omp_get_thread_num();
#endif
}
__device__ int get_n_threads() {
#if __CUDA_ARCH__
    unsigned int grid_dim;
    unsigned int block_dim;
    asm volatile ("mov.u32 %0, %%ntid.x;"  : "=r"(block_dim));
    asm volatile ("mov.u32 %0, %%nctaid.x;" : "=r"(grid_dim));
    return grid_dim * block_dim;
#else
    return omp_get_num_threads();
#endif
}
#endif
/*** endif */

/*** if "ilp_c" in VER */
template<long C>
void cycle_conc_t(long * a, long idx, long n, long * end, int * thread_idx) {
/*** if DBG >= 1 */
#if DBG >= 1
  printf("chase_ptrs : cells = %p\n", a);
#endif
/*** endif */
  long k[C];
  for (long c = 0; c < C; c++) {
    k[c] = idx + c;
  }
  asm volatile("// ========== loop begins C = %0 ========== " : : "i" (C));
#pragma unroll(8)
  for (long i = 0; i < n; i++) {
    for (long c = 0; c < C; c++) {
/*** if DBG >= 2 */
#if DBG >= 2
      printf("cycle [%ld] : p = %ld\n", idx + c, k[c]);
#endif
/*** endif */
      k[c] = a[k[c]];
    }
  }
  asm volatile("// ---------- loop ends C = %0 ---------- " : : "i" (C));
  for (long c = 0; c < C; c++) {
/*** if DBG >= 2 */
#if DBG >= 2
    printf("chase_ptrs [%ld] : return %ld\n", idx + c, k[c]);
#endif
/*** endif */
    end[idx + c] = k[c];
    thread_idx[idx + c] = get_thread_index();
  }
}

void cycle_conc(long * a, long idx, long C, long n, long * end, int * thread_idx) {
  const long max_const_c = 12;
  long c;
  for (c = 0; c + max_const_c <= C; c += max_const_c) {
    cycle_conc_t<max_const_c>(a, idx + c, n, end, thread_idx + c);
  }
  switch (C - c) {
  case 0:
    break;
  case 1:
    cycle_conc_t<1>(a, idx + c, n, end, thread_idx + c);
    break;
  case 2:
    cycle_conc_t<2>(a, idx + c, n, end, thread_idx + c);
    break;
  case 3:
    cycle_conc_t<3>(a, idx + c, n, end, thread_idx + c);
    break;
  case 4:
    cycle_conc_t<4>(a, idx + c, n, end, thread_idx + c);
    break;
  case 5:
    cycle_conc_t<5>(a, idx + c, n, end, thread_idx + c);
    break;
  case 6:
    cycle_conc_t<6>(a, idx + c, n, end, thread_idx + c);
    break;
  case 7:
    cycle_conc_t<7>(a, idx + c, n, end, thread_idx + c);
    break;
  case 8:
    cycle_conc_t<8>(a, idx + c, n, end, thread_idx + c);
    break;
  case 9:
    cycle_conc_t<9>(a, idx + c, n, end, thread_idx + c);
    break;
  case 10:
    cycle_conc_t<10>(a, idx + c, n, end, thread_idx + c);
    break;
  case 11:
    cycle_conc_t<11>(a, idx + c, n, end, thread_idx + c);
    break;
  default:
    assert(C < max_const_c);
    break;
  }
}
  
/*** elif "ilp" in VER */
void cycle_conc(long * a, long idx, long C, long n, long * end, int * thread_idx) {
/*** if DBG >= 1 */
#if DBG >= 1
  printf("chase_ptrs : cells = %p\n", a);
#endif
/*** endif */
  long k[C];
  /* track only every L elements */
  for (long c = 0; c < C; c++) {
    k[c] = idx + c;
  }
  asm volatile("// ========== loop begins ========== ");
#pragma unroll(8)
  for (long i = 0; i < n; i++) {
    for (long c = 0; c < C; c++) {
/*** if DBG >= 2 */
#if DBG >= 2
      printf("cycle [%ld] : p = %ld\n", idx + c, k[c]);
#endif
/*** endif */
      k[c] = a[k[c]];
    }
  }
  asm volatile("// ---------- loop ends ---------- ");
  for (long c = 0; c < C; c++) {
/*** if DBG >= 2 */
#if DBG >= 2
    printf("chase_ptrs [%ld] : return %ld\n", idx + c, k[c]);
#endif
/*** endif */
    end[idx + c] = k[c];
    thread_idx[idx + c] = get_thread_index();
  }
}
/*** else */
/* starting from cell &a[idx], chase ->next ptr n times
   and put where it ends in end[idx] */
__host__ __device__
void cycle(long * a, long idx, long n, long * end, int * thread_idx) {
/*** if DBG >= 1 */
#if DBG >= 1
  printf("cycle : a = %p\n", a);
#endif
/*** endif */
  long k = idx;
  asm volatile("// ========== loop begins ========== ");
#pragma unroll(8)
  for (long i = 0; i < n; i++) {
/*** if DBG >= 2 */
#if DBG >= 2
    printf("cycle [%ld,%ld] : %ld\n", idx, i, k);
#endif
/*** endif */
    k = a[k];
  }
  asm volatile("// ---------- loop ends ---------- ");
/*** if DBG >= 2 */
#if DBG >= 2
  printf("cycle : return %ld\n", k);
#endif
/*** endif */
  end[idx] = k;
  thread_idx[idx] = k;
}
/*** endif */

/*** if "cuda" in VER */
__global__ void cycles_g(long * a, long n_cycles, long n, long * end, int * thread_idx) {
  long nthreads = get_n_threads();
  for (long idx = get_thread_index(); idx < n_cycles; idx += nthreads) {
    cycle(a, idx, n, end, thread_idx);
  }
}
/*** endif */

/* a is an array of m cells;
   starting from &a[idx] for each idx in [0:n_cycles],
   chase ->next ptr n times and put where it ends in end[idx] */
void cycles(long * a, long m, long n, long * end, long n_cycles,
            long n_conc_cycles,
            long n_teams, long n_threads_per_team, int * thread_idx) {
/*** if "cuda" in VER */
  check_cuda_launch((cycles_g<<<n_teams,n_threads_per_team>>>(a, n_cycles, n, end, thread_idx)));
/*** elif "ilp" in VER */
#pragma omp target teams distribute parallel for num_teams(n_teams) num_threads(n_threads_per_team) map(tofrom: a[0:m], end[0:n_cycles], thread_idx[0:n_cycles])
  for (long idx = 0; idx < n_cycles; idx += n_conc_cycles) {
    cycle_conc(a, idx, n_conc_cycles, n, end, thread_idx);
  }
/*** else */
#pragma omp target teams distribute parallel for num_teams(n_teams) num_threads(n_threads_per_team) map(tofrom: a[0:m], end[0:n_cycles], thread_idx[0:n_cycles])
  for (long idx = 0; idx < n_cycles; idx++) {
    cycle(a, idx, n, end, thread_idx);
  }
/*** endif */
}

