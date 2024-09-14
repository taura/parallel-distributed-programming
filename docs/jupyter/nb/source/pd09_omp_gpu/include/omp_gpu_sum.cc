#com 2
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>

long cur_time() {
  struct timespec ts[1];
  clock_gettime(CLOCK_REALTIME, ts);
  return ts->tv_sec * 1000L * 1000L * 1000L + ts->tv_nsec;
}

// a small vector-like data structure
// vec v(n);
// and you can access elements by usual array index
// notatin v[i], thanks to operator overloading below
struct vec {
  long n;
  float * a;
  vec(long n_) {
    n = n_;
    a = new float[n];
  }
  // operator overloading to make v[i] access the element
  float& operator[](long i) {
    return a[i];
  }
};

// a function that calculates the sum of all elements of v
float sum(vec v) {
  float s = 0.0;
#ifpy VER >= 2
#pragma omp target teams distribute parallel for reduction(+:s) map(to: v, v.a[0:v.n]) map(tofrom: s)
#endifpy
  for (long i = 0; i < v.n; i++) {
    s += v[i];
  }
  return s;
}

int main(int argc, char ** argv) {
  int i = 1;
  float m = (argc > i ? atof(argv[i]) : 1000000); i++;
  vec v(m);
  // init array (on CPU)
  for (long i = 0; i < v.n; i++) {
    v[i] = 1.0;
  }
  long t0 = cur_time();
  // get sum of the array (you make it happen on GPU)
  float s = sum(v);
  long t1 = cur_time();
  printf("pid = %d, answer = %f, took %ld ns\n",
         getpid(), s, t1 - t0);
  return 0;
}
