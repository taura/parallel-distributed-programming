# Making a Pure Path Tracer from Scratch

# Problem

In this assignment, I implement **path tracing**, a core algorithm in computer graphics, from scratch in C++ without external libraries.
The target scene is a Cornell box (a box-shaped interior) with:

- sphere objects (one mirror, one diffuse)
- a small triangle mesh object
- an area light (an emissive patch on the ceiling)

The renderer outputs a PPM image, and image noise decreases as the number of samples per pixel (spp) increases.

This is a realistic application and is a good baseline for later parallelization (multicore CPU and GPU), where performance improvements can be measured clearly.

# How the algorithm/baseline code works

The implementation is in `code/pathtracer.cc`. The main steps are:

1. **Scene construction**
   - The Cornell box is built from triangles (2 triangles x 5 faces; the front side is open for the camera).
   - A ceiling emissive patch is added as an area light.
   - Two spheres and several triangles are added as scene objects.

2. **Intersection tests**
   - Spheres: analytic ray-sphere intersection
   - Triangles: Moller-Trumbore ray-triangle intersection
   - The nearest valid hit is selected

3. **Path tracing (pure path tracing)**
   - For each pixel, the code traces `spp` camera rays with jittered sampling.
   - At each hit, the next direction is chosen based on the material:
      - diffuse: cosine-weighted hemisphere sampling
      - mirror: perfect specular reflection
      - light: add emitted radiance and terminate the path
   - The path is traced across multiple bounces while updating throughput.
   - Russian roulette is applied at deeper bounces.

4. **Output and verification metrics**
   - Linear RGB is gamma-corrected (2.2) and written as PPM (P6).
   - `render_ms` (computation time) is printed.
   - `checksum` (sum over all pixel channels) is printed for reproducibility checks.

Execution parameters are configurable from the command line:

- `--width`, `--height`: image resolution
- `--spp`: samples per pixel
- `--max-depth`: maximum path depth (bounces)
- `--seed`: random seed (fixed seed gives deterministic behavior)
- `--output`: output PPM filename

By changing these parameters, runtime can be scaled from seconds to a few minutes.

# Where is parallelism

This baseline has abundant data parallelism:

- **Pixel parallelism**
  - Final pixel colors are independent.
  - On CPU, work can be split by rows or tiles.

- **Sample parallelism**
  - Samples within the same pixel are also independent.
  - They can be combined by reduction (averaging).

- **Intersection-level parallelism (future extension)**
  - Intersection checks against many objects can be parallelized.
  - Adding a BVH can reduce computation and improve memory behavior.
