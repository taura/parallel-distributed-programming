#com 2
#include <stdio.h>
#include <unistd.h>
#include <omp.h>

int main() {
  int x = 123;
  printf("before : x = %d\n", x);
  /* add private(x)/firstprivate(x) clause and see the difference */
#ifpy VER == 1
#pragma omp parallel
#elifpy VER == 2
#pragma omp parallel private(x)
#elifpy VER == 3
#pragma omp parallel firstprivate(x)
#endifpy  
  {
    int id = omp_get_thread_num();
    x++;
    printf("thread %d : x = %d\n", id, x);
  }
  printf("after : x = %d\n", x);
  return 0;
}
