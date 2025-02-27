/*** com 2 */
#include <stdio.h>
#include <unistd.h>
#include <omp.h>

int main() {
  int x = 123;
  printf("before : x = %d\n", x);
  /* add private(x) clause below and see the difference */
/*** if VER == 1 */
#pragma omp parallel
/*** elif VER == 2 */
#pragma omp parallel private(x)
/*** endif */
  {
    int id = omp_get_thread_num();
    printf("thread %d : x = %d\n", id, x);
  }
  printf("after : x = %d\n", x);
  return 0;
}
