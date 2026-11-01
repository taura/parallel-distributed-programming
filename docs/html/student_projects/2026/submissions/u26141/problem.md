Pseudo code of the main loop:

```
for each pixel (x, y):
  c = (cr, ci) = position of the pixel center in the complex plane
  z = (0, 0); k = 0
  while (k < max_iter and |z|^2 <= 4):
    z = z*z + c      // written with real and imaginary parts
    k = k + 1
  iters[y][x] = k
```

Everything else (area estimation, image output) is done after this loop, from the array `iters`.

How the result is verified:
- The area of the Mandelbrot set is known to be about 1.5066. The program estimates it as
  (area of the region) x (number of pixels with k == max_iter) / (number of all pixels).
  With finite `max_iter` and resolution, the estimate is slightly larger than the true value
  (pixels that diverge late are counted as inside), and it approaches 1.5066 as
  `max_iter` and the resolution increase.
- The image should show the well-known shape (a main cardioid with circular bulbs).
- `total_iters` (the sum of the iteration counts of all pixels) is printed.
  A parallel version must give exactly the same value as the serial version.

Result of the serial baseline on Miyabi (1 thread, `./mandelbrot 1024 1024 1000`):

| item | value |
|---|---|
| inside (pixels in the set) | 176006 |
| total_iters | 181039268 |
| estimated_area | 1.5107 (known: about 1.5066) |
| time | 0.549 sec |

This is about 3.3 x 10^8 iterations per second. About 97% of `total_iters` comes from the
pixels inside the set, each of which costs `max_iter` iterations, so the cost is
dominated by the inside of the set and its boundary.

# Where is parallelism

- Every pixel is computed independently from the others (no data dependency),
  so all W x H pixels can be computed in parallel. Each iteration of the outer loops can be
  given to a different thread (e.g. `#pragma omp parallel for`, possibly with `collapse(2)`),
  and the same structure fits a GPU well.
- The cost per pixel varies a lot: pixels far outside the set finish after a few iterations,
  while pixels inside the set take `max_iter` iterations. Therefore a plain static
  partition causes load imbalance, and the scheduling policy
  (static / dynamic / chunked / tasks) will matter.
- Computing the statistics (`inside`, `total_iters`) is a reduction.