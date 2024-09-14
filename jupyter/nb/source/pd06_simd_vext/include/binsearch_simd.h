intv binsearch_simd(int * a, int n, intv x) {
  intv p = intv(0);
  intv q = intv(n - 1);
  mask k0 = x < a[0];
  mask k1 = a[n-1] <= x;
  p = blend(k0,    intv(-1), p);
  p = blend(k1, intv(n - 1), p);
  mask k = ~k0 & ~k1;
  k &= (q - p > 1);
  while (any(k)) {
    /* INV: a[p] <= x < a[q] */
    intv r = (p + q) / 2;
    mask c = gather(a, r, k) <= x;
    p = blend( c & k, r, p);
    q = blend(~c & k, r, q);
    k &= (q - p > 1);
  }
  /* a[p] <= x < a[p+1] */
  return p;
}

int binsearch_many_simd(int * a, int n, int * x, int m) {
  int c = 0;
  assert(L == 16);
  for (int i = 0; i < m; i += L) {
    intv xv = intv(-1);
    intv li = linear(i);
    xv = blend(li < m, V(x[i]), xv);
    intv p = binsearch_simd(a, n, xv);
    assert(all(-1 <= p));
    assert(all((p == -1)    | (gather(a, p, 0 <= p) <= xv)));
    assert(all(p < n));
    assert(all((p == n - 1) | (xv < gather(a, p + 1, p < n - 1))));
    c += count_one((0 <= p) & (gather(a, p, 0 <= p) == xv));
  }
  return c;
}

