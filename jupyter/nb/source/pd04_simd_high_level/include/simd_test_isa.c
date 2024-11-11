/*** com 2 */
/*** if VER == 1 */
#include <stdio.h>
#include <x86intrin.h>
int main(int argc, char ** argv) {
  int i = 1;
  float a = (argc > i ? atof(argv[i]) : 1.23); i++;
  float b = (argc > i ? atof(argv[i]) : 1.23); i++;
  __m256 v = _mm256_set1_ps(b);
  __m256 c = a * v;
  printf("OK: c[0] = %f\n", c[0]);
  return 0;
}
/*** elif VER == 2 */
#include <stdio.h>
#include <x86intrin.h>
int main(int argc, char ** argv) {
  int i = 1;
  float a = (argc > i ? atof(argv[i]) : 1.23); i++;
  float b = (argc > i ? atof(argv[i]) : 1.23); i++;
  __m512 v = _mm512_set1_ps(b);
  __m512 c = a * v;
  printf("OK: c[0] = %f\n", c[0]);
  return 0;
}
/*** endif */
