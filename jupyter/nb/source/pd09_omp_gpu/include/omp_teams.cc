#include <stdio.h>
int main() {
  printf("hello on host\n");
#pragma omp target
#pragma omp teams
  printf("hello, I am the master of a team\n");
  printf("back on host\n");
  return 0;
}
