#include <stdio.h>
int main() {
  printf("hello on host\n");
#pragma omp target
  printf("hello from target (hopefully GPU)\n");
  printf("back on host\n");
  return 0;
}
