#com 2
#include <stdio.h>
#include <omp.h>
int main(int argc, char ** argv) {
  int i = 1;
  float t = (argc > i ? atof(argv[i]) : 10.0); i++;
  float * a = new float[3];     // heap-allocated data
  for (int i = 0; i < 3; i++) { a[i] = t + i; }
#ifpy VER == 1
#pragma omp target
#elsepy
#pragma omp target map(to: a[0:3])
#endifpy
  {
    printf("t = %f\n", t);
    printf("a = { %f, %f, %f }\n", a[0], a[1], a[2]);
  }
  return 0;
}
