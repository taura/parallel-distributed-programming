
# Making Ray Tracing from Scratch

# Problem

This project builds a baseline renderer for **ray tracing** in C++17.
The program generates a synthetic 3D scene (many spheres with different materials),
traces rays from a virtual camera, and outputs a rendered image as a PPM file.

Why this problem is interesting:

- It is realistic: ray tracing is a core algorithm in graphics and visualization.
- It has rich computation: each pixel needs many stochastic samples and recursive light transport.
- It is easy to scale runtime using parameters like image size, samples per pixel, and maximum bounce depth.
- It provides meaningful/verifiable outputs:
  - the generated image should look visually plausible,
  - average luminance and a checksum are printed for reproducibility checks.

The baseline is intentionally serial and simple so it can be parallelized later in the course.

# How the algorithm/baseline code works

`raytrace.cpp` implements a small serial path tracer:

1. **Scene generation**
   - Create a ground sphere and a few large spheres.
   - Add many small random spheres (`--small-spheres`) with Lambertian, metal, or dielectric materials.
   - Use a fixed seed (`--seed`) for deterministic scene and sampling.

2. **Camera and rays**
   - Use a pinhole camera with configurable image resolution (`--width`, `--height`).
   - For each pixel, cast multiple jittered primary rays (`--spp`) for anti-aliasing.

3. **Ray-object intersection**
   - Intersect rays with all spheres.
   - Keep the nearest valid hit in `(t_min, t_max)`.

4. **Material scattering (path tracing loop)**
   - **Lambertian**: random diffuse bounce.
   - **Metal**: reflected direction with fuzz.
   - **Dielectric**: reflection/refraction using Schlick approximation.
   - Iterate up to `--max-depth` bounces.

5. **Accumulation and output**
   - Average samples for each pixel.
   - Apply gamma correction (`sqrt`) and store RGB bytes.
   - Save image in PPM P6 format (`--output`).
   - Print timing (`render_ms`) for the main render loop.
   - Print `avg_luminance` and `checksum` for informal correctness/reproducibility checks.

Example compile and run:

```bash
clang++ -O3 -Wall -std=c++17 raytrace.cpp -o raytrace
./raytrace --width 960 --height 540 --spp 64 --max-depth 12 --small-spheres 120 --seed 12345 --output raytrace.ppm
```

# Where is parallelism

This application has ample parallelism.

- **Across pixels (embarrassingly parallel):**
  each pixel is independent in the baseline implementation.
- **Across samples in each pixel:**
  `spp` rays per pixel are independent and can be reduced into one color.
- **Across rays/bounces on GPU:**
  many rays can be traced concurrently with SIMT execution.

Expected multicore/GPU approach (later assignments):

- Multicore CPU: parallelize the outer loops over image rows/pixels.
- GPU: map one thread (or one warp strategy) per pixel/sample and optimize memory access.

Possible speedup limits:

- Load imbalance from variable path lengths (some rays terminate quickly, others bounce more).
- Branch divergence from different materials and random control flow.
- Memory bandwidth/cache behavior when reading scene objects repeatedly.
- RNG cost and synchronization/overhead if work granularity is too fine.
