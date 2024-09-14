void axpb(float a, float * x, float b, long n) {
  for (long i = 0; i < n; i++) {
    x[i] = a * x[i] + b;
  }
}
