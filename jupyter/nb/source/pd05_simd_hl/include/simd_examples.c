/*** com 20 */
/*** if VER % 100 == 1 */
void axpb(float a, float * x, float b, long n) {
  for (long i = 0; i < n; i++) {
    x[i] = a * x[i] + b;
  }
}
/*** elif VER % 100 == 2 */
void axpb_scalar(float a, float * x, float b, long n) {
  asm volatile("# ========== loop begins ==========");
  for (long i = 0; i < n; i++) {
    x[i] = a * x[i] + b;
  }
  asm volatile("# ---------- loop ends ----------");
}
/*** elif VER % 100 < 7 */
/*** if VER % 100 == 3 */
void axpb_simd(float a, float * x, float b, long n) {
  asm volatile("# ========== loop begins ==========");
  for (long i = 0; i < n; i++) {
    x[i] = a * x[i] + b;
  }
  asm volatile("# ---------- loop ends ----------");
}
/*** elif VER % 100 == 4 */
void axpb_simd_no_remainder(float a, float * x, float b, long n) {
  n = (n / 16) * 16;    /* just so that there are no scalar iterations */
  asm volatile("# ========== loop begins ==========");
  for (long i = 0; i < n; i++) {
    x[i] = a * x[i] + b;
  }
  asm volatile("# ---------- loop ends ----------");
}
/*** elif VER % 100 == 5 */
void dependency(float a, float * x, float b, long n) {
  asm volatile("# ========== loop begins ==========");
  for (long i = 0; i < n - 1; i++) {
    x[i+1] = a * x[i] + b;
  }
  asm volatile("# ---------- loop ends ----------");
}
/*** elif VER % 100 == 6 */
void uncertain_dependency(float a, float * x, float b, float * y, long n) {
  n = (n / 16) * 16;
  asm volatile("# ========== loop begins ==========");
  for (long i = 0; i < n; i++) {
    y[i] = a * x[i] + b;
  }
  asm volatile("# ---------- loop ends ----------");
}
/*** endif */
/*** else */
/*** if VER % 100 == 7 */
void axpb_omp_simd(float a, float * x, float b, float * y, long n) {
  n = (n / 16) * 16;
  asm volatile("# ========== loop begins ==========");
#pragma omp simd
  for (long i = 0; i < n; i++) {
    y[i] = a * x[i] + b;
  }
  asm volatile("# ---------- loop ends ----------");
}
/*** elif VER % 100 == 8 */
float sum(float * x, long n) {
  n = (n / 16) * 16;
  float s = 0.0;
  asm volatile("# ========== loop begins ==========");
#pragma omp simd reduction(+:s)
  for (long i = 0; i < n; i++) {
    s += x[i];
  }
  asm volatile("# ---------- loop ends ----------");
  return s;
}
/*** elif VER % 100 == 9 */
void branch(float a, float * x, float b, long n) {
  n = (n / 16) * 16;
  asm volatile("# ========== loop begins ==========");
#pragma omp simd
  for (long i = 0; i < n; i++) {
    if (i % 2 == 0) {
      x[i] = a * x[i] + b;
    }
  }
  asm volatile("# ---------- loop ends ----------");
}
/*** elif VER % 100 == 10 */
/* inner loop, with a compile-time constant trip count */
void loop_c(float a, float * x, float b, long n) {
  n = (n / 16) * 16;
  asm volatile("# ========== loop begins ==========");
#pragma omp simd
  for (long i = 0; i < n; i++) {
    for (long k = 0; k < 10; k++) {
      x[i] = a * x[i] + b;
    }
  }
  asm volatile("# ---------- loop ends ----------");
}
/*** elif VER % 100 == 11 */
/* inner loop, with a loop-invariant trip count */
void loop_i(float a, float * x, float b, long n, long m) {
  n = (n / 16) * 16;
  asm volatile("# ========== loop begins ==========");
#pragma omp simd
  for (long i = 0; i < n; i++) {
    for (int k = 0; k < m; k++) {
      x[i] = a * x[i] + b;
    }
  }
  asm volatile("# ---------- loop ends ----------");
}
/*** elif VER % 100 == 12 */
/* inner loop, with a variable trip count */
void loop_v(float a, float * x, float b, long n) {
  n = (n / 16) * 16;
  asm volatile("# ========== loop begins ==========");
#pragma omp simd
  for (long i = 0; i < n; i++) {
    for (int k = 0; k < i; k++) {
      x[i] = a * x[i] + b;
    }
  }
  asm volatile("# ---------- loop ends ----------");
}
/*** elif VER % 100 == 13 */
float f(float a, float x, float b);
void funcall(float a, float * x, float b, long n) {
  n = (n / 16) * 16;
  asm volatile("# ========== loop begins ==========");
#pragma omp simd
  for (long i = 0; i < n; i++) {
    x[i] = f(a, x[i], b);
  }
  asm volatile("# ---------- loop ends ----------");
}
/*** elif VER % 100 == 14 */
#pragma omp declare simd uniform(a, b) notinbranch
float f(float a, float x, float b);

void funcall_decl_simd(float a, float * x, float b, long n) {
  n = (n / 16) * 16;
  asm volatile("# ========== loop begins ==========");
#pragma omp simd
  for (long i = 0; i < n; i++) {
    x[i] = f(a, x[i], b);
  }
  asm volatile("# ---------- loop ends ----------");
}
/*** elif VER % 100 == 15 */
#pragma omp declare simd uniform(a, b) notinbranch
float fundef_decl_simd(float a, float x, float b) {
  return a * x + b;
}
/*** elif VER % 100 == 16 */
void stride_load(float a, float * x, float b, float * y, long n) {
  n = (n / 16) * 16;
  asm volatile("# ========== loop begins ==========");
#pragma omp simd
  for (long i = 0; i < n; i++) {
    y[i] = a * x[i * 2] + b;
  }
  asm volatile("# ---------- loop ends ----------");
}
/*** elif VER % 100 == 17 */
void stride_store(float a, float * x, float b, float * y, long n) {
  n = (n / 16) * 16;
  asm volatile("# ========== loop begins ==========");
#pragma omp simd
  for (long i = 0; i < n; i++) {
    y[i * 2] = a * x[i] + b;
  }
  asm volatile("# ---------- loop ends ----------");
}
/*** elif VER % 100 == 18 */
typedef struct {
  float x;
  float y;
} point_t;
void struct_load(float a, point_t * p, float b, point_t * q, long n) {
  n = (n / 16) * 16;
  asm volatile("# ========== loop begins ==========");
#pragma omp simd
  for (long i = 0; i < n; i++) {
    q[i].x = a * p[i].x + b;
  }
  asm volatile("# ---------- loop ends ----------");
}
/*** elif VER % 100 == 19 */
void non_affine_idx(float a, float * x, float b, float * y, long n) {
  n = (n / 16) * 16;
  asm volatile("# ========== loop begins ==========");
  for (long i = 0; i < n; i++) {
    y[i] = a * x[i * i % n] + b;
  }
  asm volatile("# ---------- loop ends ----------");
}
/*** elif VER % 100 == 20 */
void indirect_idx(float a, float * x, long * idx, float b, float * y, long n) {
  n = (n / 16) * 16;
  asm volatile("# ========== loop begins ==========");
  for (long i = 0; i < n; i++) {
    y[i] = a * x[idx[i]] + b;
  }
  asm volatile("# ---------- loop ends ----------");
}
/*** else */
#if defined(__AVX512F__)
enum { simd_width = 64 };       /* 512 bit = 64 bytes */
#elif defined(__AVX2__) || defined(__AVX__)
enum { simd_width = 32 };       /* 256 bit = 32 bytes */
#else
#error "sorry, you must have either __AVX__, __AVX2__, or __AVX512F__"
#endif
typedef float floatv __attribute__((vector_size(simd_width),aligned(sizeof(float))));
const int n_lanes = sizeof(floatv) / sizeof(float);

#define V(p) (*((floatv *)&p))
void axpy(float a, float * x, float b, long m) {
  for (long j = 0; j + n_lanes <= m; j += n_lanes) {
    V(x[j]) = a * V(x[j]) + b;
  }
}
/*** endif */
/*** endif */
