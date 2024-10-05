/*** com 4 */
#include <stdio.h>
#include <unistd.h>
#include <omp.h>

int main() {
  int x = 123;
  printf("before : x = %d\n", x);
/*** if VER < 4 */
#pragma omp parallel
/*** else */
#pragma omp parallel reduction(+:x)
/*** endif */
  {
    int id = omp_get_thread_num();
/*** if VER == 2 */
#pragma omp critical
/*** elif VER == 3 */
#pragma omp atomic
/*** endif */
    x++;
  }
  printf("after : x = %d\n", x);
  return 0;
}
