#com 2
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <omp.h>

/* 3-element vector */
typedef struct {
  double a[3];
} vec_t;

/* x += y */
void vec_add(vec_t * x, vec_t * y) {
  for (int i = 0; i < 3; i++) {
    x->a[i] += y->a[i];
  }
}

/* x = {0,0,0} */
void vec_init(vec_t * x) {
  for (int i = 0; i < 3; i++) {
    x->a[i] = 0;
  }
}

#ifpy VER == 2
#pragma omp declare reduction                   \
  (vp : vec_t : vec_add(&omp_out,&omp_in))      \
  initializer(vec_init(&omp_priv))
#endifpy

/* add an appropriate #pragma omp declare reduction ... here */
  
int main() {
  vec_t v;
  vec_init(&v);
  double t0 = omp_get_wtime();
  /* add an appropriate reduction clause, so that
     the result is always {10000,10000,10000} */
#ifpy VER == 2
#pragma omp parallel for reduction(vp:v)
#elsepy
#pragma omp parallel for
#endifpy
  for (int i = 0; i < 30000; i++) {
    v.a[i % 3]++;
  }
  double t1 = omp_get_wtime();
  printf("ans = {%.1f, %.1f, %.1f} in %f sec\n", v.a[0], v.a[1], v.a[2], t1 - t0);
  return 0;
}

