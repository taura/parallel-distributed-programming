#com 2
#include <stdio.h>
#include <omp.h>
struct point { float x; float y; };
int main(int argc, char ** argv) {
  int i = 1;
  float t = (argc > i ? atof(argv[i]) : 10.0); i++;
  float a[3] = { t, t + 1, t + 2 };
  point p = { t + 3, t + 4 };
#ifpy VER >= 2
  printf("[host] t @ %p = %f\n", &t, t);
  printf("[host] a @ %p = { %f, %f, %f }\n", a, a[0], a[1], a[2]);
  printf("[host] p @ %p = { %f, %f }\n", &p, p.x, p.y);
#endifpy
  // you do not have to explicitly say anything about t, a, or p.
  // they are automatically available on GPU
#pragma omp target
  {
#ifpy VER == 1
    printf("t = %f\n", t);
    printf("a = { %f, %f, %f }\n", a[0], a[1], a[2]);
    printf("p = { %f, %f }\n", p.x, p.y);
#elsepy
    printf("[dev ] t @ %p = %f\n", &t, t);
    printf("[dev ] a @ %p = { %f, %f, %f }\n", a, a[0], a[1], a[2]);
    printf("[dev ] p @ %p = { %f, %f }\n", &p, p.x, p.y);
#endifpy
  }
  return 0;
}
