# Making Mandelbrot Fractal Images from Scratch


# Problem

This project renders the Mandelbrot fractal. For each point in the complex plane,
$c=c_r+i c_i$, start with $z_0=0$ and repeatedly compute

$$z_{n+1}=z_n^2+c.$$

If this sequence remains bounded, then $c$ belongs to the Mandelbrot set. Each image pixel represents a complex point, and its color indicates how quickly the sequence escapes. This visualization of a complex dynamical system is also useful for studying parallelization with uneven computational workloads.

The program can perform only a finite number of iterations. If $|z_n|^2>4$ at any step, the sequence will diverge. Points that have not escaped by the iteration limit are drawn in black. **Black means only that no escape was observed within the specified limit; it does not prove that the point belongs to the set.**

The inputs are the image width, height, maximum iteration count, and output filename. The outputs are a P6 PPM image, the main computation time, and correctness metrics. No external data or random numbers are required.

# How the algorithm/baseline code works

`mandelbrot.cc` is a serial implementation using only the C++17 standard library.

1. Map the real-axis interval $[-2.5,1.0]$ to the image width, sampling pixel centers. The imaginary-axis interval is centered at zero and spans $3.5\times height/width$, giving equal horizontal and vertical pixel scales. Changing the aspect ratio changes the visible imaginary-axis range.
2. Visit every pixel row by row and column by column using two nested loops.
3. For each pixel, evaluate the complex recurrence using real arithmetic:
   $$x_{n+1}=x_n^2-y_n^2+c_r,\qquad y_{n+1}=2x_ny_n+c_i.$$
   Compute both new values from the old values, then test for escape.
4. Store the first escape iteration in a flat array; zero means that no escape was observed. Escape on the final allowed iteration is distinguished from no escape. For escaped points, perform two additional iterations and store the resulting squared magnitude in another array to improve the smooth-coloring approximation.
5. After the main computation, collect statistics, compute smooth colors, and write the image. `compute_seconds` includes per-pixel computation and the two post-escape iterations, but excludes array allocation, statistics, logarithm/palette calculations, and file writing. `postprocess_seconds` includes statistics, symmetry checks, coloring, and file writing.

The worst-case time complexity is $O(width\times height\times max\_iter)$, and storing the results requires $O(width\times height)$ space. The actual workload depends on how quickly each point escapes. The baseline uses no multithreading, GPU computation, or analytical shortcuts for identifying points inside the set.

## Smooth coloring

Coloring by integer escape counts assigns the same color to pixels escaping on the same iteration, producing visible bands. The current version uses a continuous value for escaped points:

$$\mu=(n+2)+1-\log_2(\ln|z_{n+2}|).$$

Here, $n$ is the first escape iteration. The two extra iterations improve the approximation of post-escape growth by repeated squaring, reducing seams between bands. The value $t=\ln(1+\max(\mu,0))$ is fed into a continuous cosine palette. Points with no observed escape remain black. Smooth coloring improves color transitions but is not edge antialiasing.

`total_iterations` retains its original definition, counting only up to the first escape or the iteration limit, so that versions can be compared with identical parameters. `coloring_extra_iterations` separately counts the additional iterations used for coloring; their sum is the actual number of orbit iterations performed. At the default size, the two result arrays occupy approximately 104 MB, assuming the usual 4-byte int and 8-byte double.

## Correctness checks

- `./mandelbrot --self-test` checks known points and iteration-limit cases. For example, the orbits for $c=0,-1,-2$ are bounded; for $c=1$, the sequence is $0,1,2,5,\ldots$, so it should escape on the third iteration.
- Conjugate points $c_r+i c_i$ and $c_r-i c_i$ should have identical escape counts. The program checks vertically mirrored pixels, and `symmetry_mismatches` should be zero.
- `smooth_symmetry_mismatches` checks vertical symmetry of the stored squared magnitudes and should also be zero. The self-test additionally verifies a known post-escape orbit, finiteness of smooth values, and the iteration-offset correction.
- The image should show a central cardioid and a circular region to its left, symmetric about the real axis, with intricate boundary details.
- `total_iterations` and `nonescaped_pixels` can be compared across repeated runs with identical parameters. These metrics and symmetry checks provide supporting evidence, not a complete mathematical proof of correctness.

## Building and running on Miyabi

The assignment notebook uses the following compilation cell:

```bash
%%bash
cd code
module purge && module load nvidia/25.9 cuda/12.9 && PATH=/work/gt81/share/env/llvm/bin:$PATH
clang++ -O3 -Wall -std=c++17 mandelbrot.cc -o mandelbrot
```

After each kernel restart, run `%load_ext miyabi` in a Python cell, then submit the computation with the following cell:

```bash
%%sub
cd code
./mandelbrot --self-test && ./mandelbrot 3600 2400 77000 mandelbrot.ppm
```

The command syntax is `./mandelbrot [width] [height] [max_iter] [output]`, with defaults of `3600 2400 77000 mandelbrot.ppm`, targeting approximately five minutes on Miyabi. Explicit command-line arguments override the defaults; the current notebook command uses these parameters. Dimensions must be between 1 and 16384, and the iteration limit between 1 and 1000000. These are input bounds, not a guarantee that the largest settings finish in a reasonable time.

The following sizes range from a small correctness check to the selected serial benchmark:

| Purpose | Width × height | Iteration limit |
|---|---|---|
| Small image check | 300 × 200 | 100 |
| Initial benchmark | 1200 × 800 | 500 |
| Larger benchmark for the original version | 3600 × 2400 | 3000 |
| Five-minute smooth-coloring benchmark (measured: 296.90 s) | 3600 × 2400 | 77000 |

### Miyabi records for the original integer-coloring version

The following measurements are from the original integer-coloring version, as recorded in the student's run outputs in `pd02_app.chat`:

| Width × height | Iteration limit | Main computation (s) | Total iterations | Nonescaped pixels |
|---|---:|---:|---:|---:|
| 1200 × 800 | 500 | 0.224865 | 93439722 | 177950 |
| 3600 × 2400 | 3000 | 11.804647 | 4832613712 | 1595104 |

Both runs passed the self-test and reported zero `symmetry_mismatches`. In the second run, nonescaped pixels accounted for approximately 18.46% of all pixels but approximately 99.02% of the iterations.

At the same resolution, increasing the limit to 77000 gave an estimated main computation time of roughly 300 seconds based on these pixels' iteration workload. The smooth-coloring version subsequently measured 296.899866 seconds, meeting the approximately five-minute target. Increasing the iteration limit mainly extends the search for slowly escaping points and increases the workload; it does not guarantee a large visual change across the image. Improved color transitions come from smooth coloring itself.

### Latest results for the smooth-coloring version

The run command was `./mandelbrot 3600 2400 77000 mandelbrot.ppm`. The values below come from the saved execution output in `pd02_app.ipynb`. The run took place on Miyabi using the compilation command above, and the generated `mandelbrot.png` was visually inspected.

| Metric | Measured result |
|---|---:|
| Resolution | 3600 × 2400 (8640000 pixels) |
| Iteration limit | 77000 |
| `compute_seconds` | 296.899866 s |
| `postprocess_seconds` | 0.581852 s |
| `total_iterations` | 122790349356 |
| `coloring_extra_iterations` | 14092140 |
| `nonescaped_pixels` | 1593930 / 8640000 |
| `symmetry_mismatches` | 0 |
| `smooth_symmetry_mismatches` | 0 |
| Coloring mode | smooth |
| Self-test | PASS |

The two measured times sum to **297.481718 seconds**, or approximately 4 minutes 57.5 seconds. This covers the main computation and postprocessing, excluding job queueing, process startup, array allocation before timing, the self-test, and PNG conversion in the notebook. The main computation takes approximately 4 minutes 56.9 seconds, about 1.03% below the 300-second target, so the current parameters are retained for subsequent serial benchmarking.

- **Image structure**: The image shows the Mandelbrot cardioid, the circular region to its left, and fine branches near the boundary, with visual symmetry about the real axis.
- **Color transitions**: Large exterior regions show continuous gradients, while rich colored details remain near the boundary. This matches the expected behavior of smooth escape-time coloring.
- **Check results**: All 8 known-point/iteration-limit tests, 1200 conjugate-symmetry tests, and smooth-coloring tests passed. The full image has zero symmetry mismatches for both escape counts and smooth-coloring data.
- **Iteration depth**: At the same resolution, the nonescaped count decreased from 1595104 with a limit of 3000 to 1593930, meaning that an additional 1174 pixels were identified as escaping within the higher limit. This also illustrates why failure to escape within a finite limit does not prove set membership.
- **Workload distribution**: Nonescaped pixels account for approximately 18.45% of pixels, but their $1593930\times77000$ iterations contribute approximately 99.953% of `total_iterations`. Equal pixel counts therefore do not imply equal workloads, motivating subsequent load-balancing experiments.
- **Timing analysis**: Postprocessing accounts for approximately 0.196% of the two measured times combined, so orbit iteration dominates at this size. Including the additional smooth-coloring steps, the actual total is 122804441496 orbit iterations. The relative contribution of postprocessing may increase after parallelizing the main computation.

Both the coloring method and the iteration limit changed between the old and new runs, so these runs cannot isolate the performance overhead of smooth coloring. Measuring that overhead requires comparing versions at the same resolution and iteration limit.

These are single-run records. Subsequent speedup measurements should use repeated runs with identical parameters and clearly defined timing boundaries to obtain a more stable serial reference.

Increasing the resolution or iteration limit lengthens the run; decreasing them shortens it. The target serial runtime ranges from seconds to a few minutes, and each job is limited to 15 minutes.

Image conversion cell:

```bash
%%bash
convert code/mandelbrot.ppm code/mandelbrot.png
```

Image display cell:

```python
from IPython.display import Image, display
display(Image("code/mandelbrot.png"))
```

The notebook contains the saved numerical output and displayed image from this run.

# Where is parallelism

- **Independent pixels**: Each pixel reads fixed parameters and writes to its own array entries, allowing multiple pixels to be computed concurrently.
- **Multicore CPU**: Rows, pixel blocks, or individual pixels can be assigned to multiple threads. Interior points and points near the boundary usually require more work, so static assignment of equal numbers of rows may cause load imbalance. Different scheduling strategies can be compared in later work.
- **GPU**: Each thread can handle one pixel. Pixels within the same execution group may escape at different times, causing branch divergence and waiting that limit speedup.
- **Dependencies within a pixel**: $z_{n+1}$ depends on $z_n$, so all iterations for one pixel cannot simply run simultaneously. Techniques such as SIMD are mainly suitable for advancing multiple pixels together.
- **Performance analysis**: Compare computation time and speedup across resolutions, iteration limits, and scheduling strategies, and study how load imbalance, serial sections, and output overhead affect overall performance.
