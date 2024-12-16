#include <assert.h>
/*** if "omp" in VER */
#if __NVCOMPILER                // NVIDIA nvc++
#else  // Clang
#define __host__
#define __device__
#define __global__
#endif
/*** endif */

/*** if "ilp_c" in VER */
template<long C>
void cycle_conc_t(long * a, long idx, long n, long * end) {
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
/*** if DBG >= 2 */
#if DBG >= 2
    printf("cycle [%ld] : p = %ld\n", idx, k);
#endif
/*** endif */
    for (long c = 0; c < C; c++) {
      k[c] = a[k[c]];
    }
  }
  asm volatile("// ---------- loop ends C = %0 ---------- " : : "i" (C));
/*** if DBG >= 2 */
#if DBG >= 2
  printf("chase_ptrs : return %ld\n", k);
#endif
/*** endif */
  for (long c = 0; c < C; c++) {
    end[idx + c] = k[c];
  }
}

void cycle_conc(long * a, long idx, long C, long n, long * end) {
  const long max_const_c = 12;
  long c;
  for (c = 0; c + max_const_c <= C; c += max_const_c) {
    cycle_conc_t<max_const_c>(a, idx + c, n, end);
  }
  switch (C - c) {
  case 0:
    break;
  case 1:
    cycle_conc_t<1>(a, idx + c, n, end);
    break;
  case 2:
    cycle_conc_t<2>(a, idx + c, n, end);
    break;
  case 3:
    cycle_conc_t<3>(a, idx + c, n, end);
    break;
  case 4:
    cycle_conc_t<4>(a, idx + c, n, end);
    break;
  case 5:
    cycle_conc_t<5>(a, idx + c, n, end);
    break;
  case 6:
    cycle_conc_t<6>(a, idx + c, n, end);
    break;
  case 7:
    cycle_conc_t<7>(a, idx + c, n, end);
    break;
  case 8:
    cycle_conc_t<8>(a, idx + c, n, end);
    break;
  case 9:
    cycle_conc_t<9>(a, idx + c, n, end);
    break;
  case 10:
    cycle_conc_t<10>(a, idx + c, n, end);
    break;
  case 11:
    cycle_conc_t<11>(a, idx + c, n, end);
    break;
  default:
    assert(C < max_const_c);
    break;
  }
}
  
/*** elif "ilp" in VER */
void cycle_conc(long * a, long idx, long C, long n, long * end) {
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
/*** if DBG >= 2 */
#if DBG >= 2
    printf("cycle [%ld] : p = %ld\n", idx, k);
#endif
/*** endif */
    for (long c = 0; c < C; c++) {
      k[c] = a[k[c]];
    }
  }
  asm volatile("// ---------- loop ends ---------- ");
/*** if DBG >= 2 */
#if DBG >= 2
  printf("chase_ptrs : return %ld\n", k);
#endif
/*** endif */
  for (long c = 0; c < C; c++) {
    end[idx + c] = k[c];
  }
}
/*** elif "simd" in VER */
#define V(p) (*(volatile Tv*)&(p))

template<typename T, typename Tv, long C>
void chase_ptrs_simd_ilp_t(T * a, long n, T * end, long idx) {
/*** if DBG >= 1 */
#if DBG >= 1
  printf("chase_ptrs : cells = %p\n", a);
#endif
/*** endif */
  const long L = sizeof(Tv) / sizeof(T);
  assert(C % L == 0);
  T k[C];
  /* track only every L elements */
  for (long i = 0; i < C; i += L) {
    k[i] = idx + i;
  }
  asm volatile("// ========== loop begins ========== ");
#pragma unroll(8)
  for (long i = 0; i < n; i++) {
/*** if DBG >= 2 */
#if DBG >= 2
    printf("cycle [%ld] : p = %ld\n", idx, k);
#endif
/*** endif */
    for (long i = 0; i < C; i += L) {
      k[i] = V(a[k[i]])[0];
    }
  }
  asm volatile("// ---------- loop ends ---------- ");
/*** if DBG >= 2 */
#if DBG >= 2
  printf("chase_ptrs : return %ld\n", k);
#endif
/*** endif */
  for (long i = 0; i < C; i += L) {
    for (long j = 0; j < L; j++) {
      end[idx + i + j] = k[i] + j;
    }
  }
}

template<typename T, typename Tv>
void chase_ptrs_simd_ilp(T * a, long n, T * end, long idx, long C) {
  const long L = sizeof(Tv) / sizeof(T);
  assert(C % L == 0);
  switch (C / L) {
  case 1:
    chase_ptrs_simd_ilp_t<T,Tv,L>(a, n, end, idx);
    break;
  case 2:
    chase_ptrs_simd_ilp_t<T,Tv,2 * L>(a, n, end, idx);
    break;
  case 3:
    chase_ptrs_simd_ilp_t<T,Tv,3 * L>(a, n, end, idx);
    break;
  case 4:
    chase_ptrs_simd_ilp_t<T,Tv,4 * L>(a, n, end, idx);
    break;
  case 5:
    chase_ptrs_simd_ilp_t<T,Tv,5 * L>(a, n, end, idx);
    break;
  case 6:
    chase_ptrs_simd_ilp_t<T,Tv,6 * L>(a, n, end, idx);
    break;
  case 7:
    chase_ptrs_simd_ilp_t<T,Tv,7 * L>(a, n, end, idx);
    break;
  case 8:
    chase_ptrs_simd_ilp_t<T,Tv,8 * L>(a, n, end, idx);
    break;
  case 9:
    chase_ptrs_simd_ilp_t<T,Tv,9 * L>(a, n, end, idx);
    break;
  case 10:
    chase_ptrs_simd_ilp_t<T,Tv,10 * L>(a, n, end, idx);
    break;
  case 11:
    chase_ptrs_simd_ilp_t<T,Tv,11 * L>(a, n, end, idx);
    break;
  case 12:
    chase_ptrs_simd_ilp_t<T,Tv,12 * L>(a, n, end, idx);
    break;
  default:
    assert(C <= 12);
    break;
  }
}
/*** else */
/* starting from cell &a[idx], chase ->next ptr n times
   and put where it ends in end[idx] */
__host__ __device__
void cycle(long * a, long idx, long n, long * end) {
/*** if DBG >= 1 */
#if DBG >= 1
  printf("cycle : a = %p\n", a);
#endif
/*** endif */
  long k = idx;
  asm volatile("// ========== loop begins ========== ");
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
}
/*** endif */

/*** if "cuda" in VER */
__global__ void cycles_g(long * a, long n_cycles, long n, long * end) {
  long nthreads = n_threads();
  for (long idx = thread_index(); idx < n_cycles; idx += nthreads) {
    cycle(a, idx, n, end);
  }
}
/*** endif */

/* a is an array of m cells;
   starting from &a[idx] for each idx in [0:n_cycles],
   chase ->next ptr n times and put where it ends in end[idx] */
void cycles(long * a, long m, long n, long * end, long n_cycles,
            long n_conc_cycles,
            long n_teams, long n_threads_per_team) {
/*** if "cuda" in VER */
  check_cuda_launch((cycles_g<<<n_teams,n_threads_per_team>>>(a, n_cycles, n, end)));
/*** elif "ilp" in VER */
#pragma omp target teams distribute parallel for num_teams(n_teams) num_threads(n_threads_per_team) map(tofrom: a[0:m], end[0:n_cycles])
  for (long idx = 0; idx < n_cycles; idx += n_conc_cycles) {
    cycle_conc(a, idx, n_conc_cycles, n, end);
  }
/*** elif "simd" in VER */
  assert(n_conc_cycles % L == 0);
#pragma omp parallel for num_threads(n_threads_per_team)
  for (long idx = 0; idx < n_cycles; idx += n_conc_cycles) {
    chase_ptrs_simd_ilp<T,Tv>(cells, n, end, idx, n_conc_cycles);
  }
/*** else */
#pragma omp target teams distribute parallel for num_teams(n_teams) num_threads(n_threads_per_team) map(tofrom: a[0:m], end[0:n_cycles])
  for (long idx = 0; idx < n_cycles; idx++) {
    cycle(a, idx, n, end);
  }
/*** endif */
}

