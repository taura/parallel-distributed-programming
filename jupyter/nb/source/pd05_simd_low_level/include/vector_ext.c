/*** com 7 */
/*** if VER == 1 */
typedef float float_4 __attribute__((vector_size(16),__may_alias__,aligned(sizeof(float))));
typedef float float_8 __attribute__((vector_size(32),__may_alias__,aligned(sizeof(float))));
typedef float float_16 __attribute__((vector_size(64),__may_alias__,aligned(sizeof(float))));

float_4 dist_4(float_4 x, float_4 y) {
  return x * x + y * y;
}
float_8 dist_8(float_8 x, float_8 y) {
  return x * x + y * y;
}
float_16 dist_16(float_16 x, float_16 y) {
  return x * x + y * y;
}
/*** elif VER == 2 */
typedef float float_16 __attribute__((vector_size(64),__may_alias__,aligned(sizeof(float))));

float_16 axpb_16(float a, float_16 x) {
  return a * x + 3.0;
}
/*** elif VER == 3 */
#if defined(__AVX512F__)
#warning "__AVX512F__ defined. SIMD width = 64 bytes"
enum { simd_width = 64 };
#elif defined(__AVX2__) || defined(__AVX__)
#warning "__AVX__ defined. SIMD width = 32 bytes"
enum { simd_width = 32 };
#elif defined(__SSE2__) || defined(__SSE__)
#warning "__SSE__ defined. SIMD width = 16 bytes"
enum { simd_width = 16 };
#else
#error "sorry, you must have one of __SSE__, __SSE2__, __AVX__, __AVX2__, or __AVX512F__"
#endif

typedef float floatv __attribute__((vector_size(simd_width),__may_alias__,aligned(sizeof(float))));
const int n_float_lanes = sizeof(floatv) / sizeof(float);

floatv distv(floatv x, floatv y) {
  return x * x + y * y;
}
/*** else */
#if defined(__AVX512F__)
enum { simd_width = 64 };
#else
#error "you must have __AVX512F__ (forgot to give -mavx512f -mfma??)"
#endif

typedef float floatv __attribute__((vector_size(simd_width),__may_alias__,aligned(sizeof(float))));
const int n_float_lanes = sizeof(floatv) / sizeof(float);
/*** endif */

/*** if VER == 4 */
floatv loadv(float * a) {
  /* a vector {a[0],a[1],...,a[L-1]} */
  return *((floatv *)a);
}
/*** elif VER == 5 */
floatv make_vector(float a0, float a1, float a2, float a3,
                   float a4, float a5, float a6, float a7,
                   float a8, float a9, float a10, float a11,
                   float a12, float a13, float a14, float a15
                   ) {
  float a[16] = { a0, a1, a2, a3, a4, a5, a6, a7,
    a8, a9, a10, a11, a12, a13, a14, a15
  };
  return *((floatv *)a);
}
/*** elif VER == 6 */
void storev(float * a, floatv v) {
  *((floatv *)a) = v;
}
/*** elif VER == 7 */
float get_i(floatv v, int i) {
  return v[i];
}
/*** endif */
