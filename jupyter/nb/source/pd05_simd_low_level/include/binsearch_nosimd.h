/* search a[0:n] looking for value x,
   assuming a is sorted.
   return an integer p s.t.
   a[p] <= x < a[p+1]
   (a[-1] is considered smaller than any value and 
    a[n] larger than any value)
 */
int binsearch(int * a, int n, int x) {
  if (x < a[0]) return -1;
  if (a[n-1] <= x) return n - 1;
  /* a[0] <= x < a[n-1]  */
  int p = 0, q = n - 1;
  while (q - p > 1) {
    /* INV: a[p] <= x < a[q] */
    int r = (p + q) / 2;
    if (a[r] <= x) p = r;
    else q = r;
  }
  /* a[p] <= x < a[p+1] */
  return p;
}

/* search a[0:n] for all values in x[0:m]
   and return the number of elements found */
int binsearch_many(int * a, int n, int * x, int m) {
  int c = 0;
  for (int i = 0; i < m; i++) {
    int p = binsearch(a, n, x[i]);
    assert(-1 <= p);
    assert(p == -1 || a[p] <= x[i]);
    assert(p < n);
    assert(p == n - 1 || x[i] < a[p + 1]);
    if (0 <= p && a[p] == x[i]) c++; // found? -> c++
  }
  return c;
}

