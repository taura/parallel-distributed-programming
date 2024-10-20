#include <nv/target>
#if __NVCOMPILER
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

__attribute__((unused))
static long long int get_gpu_clock(void) {
  long long int t = 0;
#pragma omp target map(from: t)
  t = clock64();
  return t;
}
