long scan(long * a, long n) {
  long k = 0;
  for (long i = 0; i < n; i++) {
    k = a[k];
  }
  return k;
}
