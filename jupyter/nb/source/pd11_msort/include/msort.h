#pragma once
/* merge a[p:q] and a[s:t] into b[d:..] */
void merge(float * a, float * b, long p, long q, long s, long t, long d, long th);

/* merge, called from main */
void merge_from_main(float * a, float * b, long p, long q, long s, long t, long d, long th);
  
/* merge sort, called from the main */
void msort_from_main(float * a, float * b, float * g, long p, long q, long th0, long th1);

