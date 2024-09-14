#com 3
#include <stdio.h>
#include <omp.h>
int main(int argc, char ** argv) {
  int i = 1;
  float t = (argc > i ? atof(argv[i]) : 10.0); i++;
  float a[3] = { t, t + 1, t + 2 };
  float * pa = a;
#ifpy VER == 2
  printf("[host]  a @ %p = { %f, %f, %f }\n", a, a[0], a[1], a[2]);
  printf("[host] pa @ %p = { %f, %f, %f }\n", pa, pa[0], pa[1], pa[2]);
#endifpy
#pragma omp target
  {
#ifpy VER == 1
    printf(" a = { %f, %f, %f }\n", a[0], a[1], a[2]);
    printf("pa = { %f, %f, %f }\n", pa[0], pa[1], pa[2]);
#elifpy VER == 2
    printf("[dev ]  a @ %p = { %f, %f, %f }\n", a, a[0], a[1], a[2]);
    printf("[dev ] pa @ %p = { %f, %f, %f }\n", pa, pa[0], pa[1], pa[2]);
#elsepy
    printf("pa = { %f, %f, %f }\n", pa[0], pa[1], pa[2]);
#endifpy
  }
  return 0;
}
