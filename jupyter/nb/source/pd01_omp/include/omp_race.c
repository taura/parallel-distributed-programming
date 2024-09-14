#com 4
#include <stdio.h>
#include <unistd.h>
#include <omp.h>

int main() {
  int x = 123;
  printf("before : x = %d\n", x);
#ifpy VER < 4  
#pragma omp parallel
#elsepy
#pragma omp parallel reduction(+:x)
#endifpy
  {
    int id = omp_get_thread_num();
#ifpy VER == 2
#pragma omp critical
#elifpy VER == 3
#pragma omp atomic
#endifpy
    x++;
    printf("thread %d : x = %d\n", id, x);
  }
  printf("after : x = %d\n", x);
  return 0;
}

