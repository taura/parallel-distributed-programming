#include <stdio.h>
#include <omp.h>
struct point { float x; float y; };
int main(int argc, char ** argv) {
  int i = 1;
  float t = (argc > i ? atof(argv[i]) : 10.0); i++;
  float a[3] = { t, t + 1, t + 2 };
  point p = { t + 3, t + 4 };
#pragma omp target 
  {
    t *= 2.0;
    for (int i = 0; i < 3; i++) a[i] *= 2.0;
    p.x *= 2.0; p.y *= 2.0;
  }
  printf("t = %f\n", t);
  printf("a = { %f, %f, %f }\n", a[0], a[1], a[2]);
  printf("p = { %f, %f }\n", p.x, p.y);
  return 0;
}
