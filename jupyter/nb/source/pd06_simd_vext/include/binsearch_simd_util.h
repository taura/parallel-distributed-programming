/* a convenient type definition and overloaded operators */

/* an aux type representing 16 ints, defined in the way suggested in slides */
typedef int _intv __attribute__((vector_size(64), __may_alias__, aligned(sizeof(int))));
enum { L = sizeof(_intv) / sizeof(int) };

/* a data structure that looks vector operations look nice 
   using C++ operator overloading */

/* this is the type you are going to use to represent 16 ints */
struct intv {
  /* takes 16 values */
  intv(_intv _v) { v = _v; }
  /* takes a single value (e.g., intv(3)) */
  intv(int _v) { v = _mm512_set1_epi32(_v); }
  _intv v;
};
/* 16 bit mask representing the result of a comparison */
typedef __mmask16 mask;

// C++ trick so that (a == b) returns the mask
mask operator==(intv a, intv b) {
  mask k = _mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_EQ);
  return k;
}

// ditto, but accept a scalar 
mask operator==(intv a, int b) {
  mask k = a == intv(b);
  return k;
}

// ditto, but accept a scalar 
mask operator==(int a, intv b) {
  mask k = intv(a) == b;
  return k;
}

// C++ trick so that (a < b) returns the mask
mask operator<(intv a, intv b) {
  mask k = _mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_LT);
  return k;
}

mask operator<(intv a, int b) {
  mask k = a < intv(b);
  return k;
}

mask operator<(int a, intv b) {
  mask k = intv(a) < b;
  return k;
}

// C++ trick so that (a > b) returns the mask
mask operator>(intv a, intv b) {
  mask k = _mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NLE);
  return k;
}

mask operator>(intv a, int b) {
  mask k = a > intv(b);
  return k;
}

mask operator>(int a, intv b) {
  mask k = intv(a) > b;
  return k;
}

// C++ trick so that (a <= b) returns the mask
mask operator<=(intv a, intv b) {
  mask k = _mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_LE);
  return k;
}

mask operator<=(intv a, int b) {
  mask k = a <= intv(b);
  return k;
}

mask operator<=(int a, intv b) {
  mask k = intv(a) <= b;
  return k;
}

// blend a and b by mask value
// k[i] is 1 -> take a[i]
// k[i] is 0 -> take b[i]
// read it as if it is C's (k ? a : b) expression
intv blend(mask k, intv a, intv b) {
  _intv v = _mm512_mask_blend_epi32(k, b.v, a.v);
  return intv(v);
}

// a + b (add other versions if necessary)
intv operator+(intv a, intv b) {
  _intv v = a.v + b.v;
  return intv(v);
}

// a - b (add other versions if necessary)
intv operator-(intv a, intv b) {
  intv v = a.v - b.v;
  return intv(v);
}

// a * b (add other versions if necessary)
intv operator*(intv a, intv b) {
  intv v = a.v * b.v;
  return intv(v);
}

// a / b (add other versions if necessary)
intv operator/(intv a, intv b) {
  _intv v = a.v / b.v;
  return intv(v);
}

intv operator/(intv a, int b) {
  intv v = a / intv(b);
  return v;
}

// gather(a, I, k) returns
// a[I0], a[I1], ..., a[IL-1] for lanes whose bit is set in k.
// unset lanes become zero
intv gather(int * a, intv i, mask k) {
  _intv y = _mm512_mask_i32gather_epi32(intv(0).v, k, i.v, a, sizeof(int));
  return intv(y);
}

// return true if all bits of k are set
int all(mask k) {
  return (0xffff & k) == 0xffff;
}

// return true if any bit of k is set
int any(mask k) {
  return k != 0;
}

// count the number of bits set in k
int count_one(mask k) {
  int x = _mm_popcnt_u32((unsigned int)k);
  return x;
}

// access consecutive L locations starting from an int location 
intv& V(int& p) {
  return *((intv*)&p);
}

// [a, a+1, a+2, ..., a+L-1]
intv linear(int a) {
  int v[L];
  for (int i = 0; i < L; i++) {
    v[i] = a + i;
  }
  return intv(*((_intv*)v));
}

