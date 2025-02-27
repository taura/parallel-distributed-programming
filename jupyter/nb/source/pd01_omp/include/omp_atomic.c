#include <stdio.h>
#include <unistd.h>
#include <omp.h>

int main() {
  int x = 123;
  printf("before : x = %d\n", x);
#pragma omp parallel
  {
    int id = omp_get_thread_num();
/*** if VER == 2 */
#pragma omp atomic
/*** endif */
    x++;
    printf("thread %d : x = %d\n", id, x);
  }
  printf("after : x = %d\n", x);
  return 0;
}

