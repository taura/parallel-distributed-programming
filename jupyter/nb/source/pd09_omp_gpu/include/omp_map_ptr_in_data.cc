#include <stdio.h>
#include <omp.h>
struct cell { float x; float * a; };
int main(int argc, char ** argv) {
  int i = 1;
  float t = (argc > i ? atof(argv[i]) : 10.0); i++;
  float a[3] = { t, t + 1, t + 2 };
  cell c = { t + 3, a };
#ifpy VER >= 2
  printf("[host]   t @ %p = %f\n", &t, t);
  printf("[host]   a @ %p = { %f, %f, %f }\n", a, a[0], a[1], a[2]);
  printf("[host] c.x @ %p = %f\n", &c.x, c.x);
  printf("[host] c.a @ %p = { %f, %f, %f }\n", c.a, c.a[0], c.a[1], c.a[2]);
#endifpy
#ifpy VER == 1
#pragma omp target
#elsepy
#pragma omp target map(to: c, c.a[0:3])
#endifpy
  {
#ifpy VER == 1
    printf("  t = %f\n", t);
    printf("  a = { %f, %f, %f }\n", a[0], a[1], a[2]);
    printf("c.x = %f\n", c.x);
    printf("c.a = %p\n", c.a);
    printf("c.a = { %f, %f, %f }\n", c.a[0], c.a[1], c.a[2]);
#elsepy
    printf("[dev ]   t @ %p = %f\n", &t, t);
    printf("[dev ]   a @ %p = { %f, %f, %f }\n", a, a[0], a[1], a[2]);
    printf("[dev ] c.x @ %p = %f\n", &c.x, c.x);
    printf("[dev ] c.a @ %p\n", c.a);
    printf("[dev ] c.a @ %p = { %f, %f, %f }\n", c.a, c.a[0], c.a[1], c.a[2]);
#endifpy
  }
  return 0;
}
