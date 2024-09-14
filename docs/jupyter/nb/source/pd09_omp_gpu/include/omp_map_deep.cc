#include <stdio.h>
#include <omp.h>
struct vec {
  int n;
  float * a;
  vec(int n_) {
    n = n_;
    a = new float[n];
  }
  float& operator[] (int i) {
    return a[i];
  }
};

int main(int argc, char ** argv) {
  int i = 1;
  int m = (argc > i ? atoi(argv[i]) : 5); i++;
  vec x(m);
  vec y(m);
  for (int i = 0; i < m; i++) {
    x[i] = i + 1;
    y[i] = 0;
  }
#pragma omp target map(tofrom: x, x.a[0:m], y, y.a[0:m])
#pragma omp teams distribute parallel for
  for (int i = 0; i < x.n; i++) {
    y[i] = x[i] * x[i];
  }
  for (int i = 0; i < m; i++) {
    printf("y[%d] = %.1f\n", i, y.a[i]);
  }
  return 0;
}
