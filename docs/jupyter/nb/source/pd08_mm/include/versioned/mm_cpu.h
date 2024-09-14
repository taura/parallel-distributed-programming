/* 
 * mm_cpu.h
 */

/* type definition */
typedef float real;
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

#define CHECK_IDX 0

struct matrix {
  idx_t M;                      // number of rows
  idx_t N;                      // number of columns
  idx_t ld;                     // leading dimension (usually = N)
  real * a;                     // array of values (M x ld elements)
  matrix(idx_t _M, idx_t _N) {
    M = _M;
    N = _N;
    ld = _N;
    a = (real *)aligned_alloc(vwidth, sizeof(real) * M * ld);
  }
  /* return a scalar A(i,j) */
  real& operator() (idx_t i, idx_t j) {
#if CHECK_IDX
    assert(i < M);
    assert(j < N);
    assert(i >= 0);
    assert(j >= 0);
#endif
    return a[i * ld + j];
  }
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
};

