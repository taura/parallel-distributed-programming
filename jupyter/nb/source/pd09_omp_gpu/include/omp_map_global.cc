#include <stdio.h>
#include <omp.h>
struct point { float x; float y; };
float t;
float a[3];
point p;
int main(int argc, char ** argv) {
  int i = 1;
  t = (argc > i ? atof(argv[i]) : 10.0); i++;
  for (int i = 0; i < 3; i++) { a[i] = t + i; }
  p.x = t + 3; p.y = t + 4;
#ifpy VER >= 2
  printf("[host] t @ %p = %f\n", &t, t);
  printf("[host] a @ %p = { %f, %f, %f }\n", a, a[0], a[1], a[2]);
  printf("[host] p @ %p = { %f, %f }\n", &p, p.x, p.y);
#endifpy
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
