idx_t scan(T * a, idx_t n, idx_t s, T x) {
  idx_t k = 0;
  for (idx_t i = 0; i < n; i++) {
    if (a[k] == x) return k;
    k += s;
    k = (k < n ? k : k - n);
  }
}

scan(T * a, idx_t * start, idx_t c, idx_t n, T x) {
  for (idx_t i = 0; i < c; i++) {
    idx_t k = start[i];
    for (idx_t j = 0; j < n; j++) {
      if (a[k] == x) return k;
      k = a[k];
    }
  }
}
