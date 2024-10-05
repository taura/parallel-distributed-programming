/*** if VER == 1 */
/* 
 * mm_cpu.h
 */
/*** elif VER == 2 */
/* 
 * mm_cuda.h
 */
/*** endif */

/* type definition */
typedef float real;
/*** if VER == 1 */
typedef long idx_t;

#if ! defined(__AVX512F__)
#error "__AVX512F__ must be defined (forgot to give -mavx512f -mfma?)"
#endif

#include <x86intrin.h>
enum { vwidth = 64 };
typedef real realv __attribute__((vector_size(vwidth),__may_alias__,aligned(vwidth)));
enum { L = sizeof(realv) / sizeof(real) };

__attribute__((unused))
static realv U(real c) {
  return _mm512_set1_ps(c);
  // return _mm512_set1_pd(c);
}

__attribute__((unused))
static realv& V(real& p) {
  return *((realv*)&p);
}
/*** elif VER == 2 */
typedef int idx_t;

#include <stdio.h>

/**
   @brief do not use this function directly. use check_api_error macro
   @sa check_api_error
 */
__attribute__((unused))
static void check_api_error_(cudaError_t e,
                              const char * msg, const char * file, int line) {
  if (e) {
    fprintf(stderr, "%s:%d:error: %s %s\n",
            file, line, msg, cudaGetErrorString(e));
    exit(1);
  }
}

/**
   @brief check if a CUDA API invocation succeeded and show the error msg if any
   @details usage:  check_api_error(cuda_api_call()). for example,
   check_api_error(cudaMalloc(&p, size));
 */

#define check_api_error(e) check_api_error_(e, #e, __FILE__, __LINE__)

/**
   @brief do not use this function directly. use check_launch_error macro
   @sa check_launch_error
 */

__attribute__((unused))
static void check_launch_error_(const char * msg, const char * file, int line) {
  cudaError_t e = cudaGetLastError();
  if (e) {
    fprintf(stderr, "%s:%d:error: %s %s\n",
            file, line, msg, cudaGetErrorString(e));
    exit(1);
  }
}

/**
   @brief check kernel launch error
   @details usage: check_launch_error((kernel-launch-expression)). for example,
   check_launch_error((your_gpu_kernel<<<n_blocks,block_sz>>>(a,b,c))). 
   note that you need to put parens around the expression.
 */

#define check_launch_error(exp) do { exp; check_launch_error_(#exp, __FILE__, __LINE__); } while (0)

/**
   @brief get SM executing the caller
 */
__attribute__((unused))
__device__ static uint get_smid(void) {
  uint ret;
  asm("mov.u32 %0, %%smid;" : "=r"(ret) );
  return ret;
}

/**
   @brief get device frequency
 */
__attribute__((unused))
static int get_freq() {
  struct cudaDeviceProp prop[1];
  check_api_error(cudaGetDeviceProperties(prop, 0));
  return prop->clockRate;
}

/**
   @brief wrap cudaMalloc.  cudaMalloc + error check + more ordinary malloc-like interface (return pointer)
 */
__attribute__((unused))
static void * dev_malloc(size_t sz) {
  void * a = 0;
  cudaError_t e = cudaMalloc(&a, sz);
  if (!a) {
    fprintf(stderr, "error: %s\n", cudaGetErrorString(e));
    exit(1);
  }
  return a;
}

/**
   @brief wrap cudaFree
 */
__attribute__((unused))
static void dev_free(void * a) {
  cudaFree(a);
}

/**
   @brief wrap cudaMemcpy to copy from device to host (and check an error if any)
 */
__attribute__((unused))
static void to_host(void * dst, void * src, size_t sz) {
  check_api_error(cudaMemcpy(dst, src, sz, cudaMemcpyDeviceToHost));
}

/**
   @brief wrap cudaMemcpy to copy from host to device (and check an error if any)
 */
__attribute__((unused))
static void to_dev(void * dst, void * src, size_t sz) {
  check_api_error(cudaMemcpy(dst, src, sz, cudaMemcpyHostToDevice));
}

/**
   @brief get clock from host
 */
__attribute__((unused))
__global__ static void get_clock_cuda(long long int * t_dev) {
  t_dev[0] = clock64();
}

__attribute__((unused))
static long long int get_gpu_clock(void) {
  long long int t[1];
  long long int * t_dev = (long long int *)dev_malloc(sizeof(long long int));
  get_clock_cuda<<<1,1>>>(t_dev);
  to_host(t, t_dev, sizeof(long long int));
  dev_free(t_dev);
  return t[0];
}
/*** endif */

#define CHECK_IDX 0

struct matrix {
  idx_t M;                      // number of rows
  idx_t N;                      // number of columns
  idx_t ld;                     // leading dimension (usually = N)
  real * a;                     // array of values (M x ld elements)
/*** if VER == 2 */
  real * a_dev;                 // shadow of a on GPU
/*** endif */
  matrix(idx_t _M, idx_t _N) {
    M = _M;
    N = _N;
    ld = _N;
/*** if VER == 1 */
    a = (real *)aligned_alloc(vwidth, sizeof(real) * M * ld);
/*** elif VER == 2 */
    a = (real *)malloc(sizeof(real) * M * ld);
    a_dev = (real *)dev_malloc(sizeof(real) * M * ld);
/*** endif */
  }
  /* return a scalar A(i,j) */
/*** if VER == 2 */
  __host__ __device__
/*** endif */
  real& operator() (idx_t i, idx_t j) {
#if CHECK_IDX
    assert(i < M);
    assert(j < N);
    assert(i >= 0);
    assert(j >= 0);
#endif
/*** if VER == 1 */
    return a[i * ld + j];
/*** elif VER == 2 */
#ifdef __CUDA_ARCH__
    return a_dev[i * ld + j];
#else
    return a[i * ld + j];
#endif
/*** endif */
  }
/*** if VER == 1 */
  /* A.V(i,j) returns a vector at A(i,j) (i.e., A(i,j:j+L)).
     you can put it on lefthand side, e.g., A.V(i,j) = ... */
  realv& V(idx_t i, idx_t j) {
#if CHECK_IDX
    assert(i < M);
    assert(j + L <= N);
    assert(i >= 0);
    assert(j >= 0);
#endif
    return ::V(a[i * ld + j]);
  }
/*** endif */
/*** if VER == 2 */
  void to_dev() {
    ::to_dev(a_dev, a, sizeof(real) * M * ld);
  }
  void to_host() {
    ::to_host(a, a_dev, sizeof(real) * M * ld);
  }
/*** endif */
};

