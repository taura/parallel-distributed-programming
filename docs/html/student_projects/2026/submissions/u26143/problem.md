
* _this is an example file you should finish_
* write problem description, how the algorithm/baseline code works, and where you think you can apply parallel computing
* if you are looking at this file from Jupyter, you can see the preview by right-clicking on the cell and "Show Markdown Preview"

# Making XXXXXX from scratch

# Problem


The core objective of ray tracing is to synthesize a 2D image of a 3D virtual scene by simulating the physics of light transport. Rather than solving the full recursive Kajiya Rendering Equation via stochastic Monte Carlo integration, the **Whitted-Style Ray Tracing model** evaluates an analytical approximation focusing on direct illumination, sharp specular reflection, and specular refraction:

$$
L_o(p, \omega_o) = L_e(p, \omega_o) + L_{\text{local}}(p, \omega_o) + k_r \, L_i(p, \omega_r) + k_t \, L_i(p, \omega_t)
$$

The problem is computationally bounded by recursive ray-primitive intersection queries and geometric visibility evaluations across millions of primary and secondary rays.

# How the algorithm/baseline code works

The baseline algorithm executes serially over the frame buffer. Primary rays are generated through a pinhole camera, tested for intersection against scene primitives, and recursively shaded up to a fixed tree depth.
```cpp
void Renderer::render_tile(
    Framebuffer& fb,
    const Camera& cam,
    int start_x,
    int start_y,
    int end_x,
    int end_y
) const {
    const int w = fb.width();
    const int h = fb.height();

    for (int y = start_y; y < end_y; ++y) {
        for (int x = start_x; x < end_x; ++x) {
            Vec3 accum(0.0, 0.0, 0.0);

            // Per-pixel deterministic RNG state (parallel-safe by design).
            uint64_t rng = 0x9e3779b97f4a7c15ULL
                         ^ (static_cast<uint64_t>(x) * 0xbf58476d1ce4e5b9ULL)
                         ^ (static_cast<uint64_t>(y) * 0x94d049bb133111ebULL);

            for (int s = 0; s < render_params_.samples_per_pixel; ++s) {
                const double jx = math::rand01(rng);
                const double jy = math::rand01(rng);

                const double u = (static_cast<double>(x) + jx) / static_cast<double>(w);
                const double v = (static_cast<double>(h - 1 - y) + jy) / static_cast<double>(h);

                const Ray primary = cam.generate_ray(u, v);
                accum += trace_ray(primary, render_params_.max_depth);
            }

            accum /= static_cast<double>(render_params_.samples_per_pixel);
            fb.set_pixel(x, y, accum);
        }
    }
}
```

# Where is parallelism

* **Location in Code**: `src/renderer/renderer.cpp` $\rightarrow$ `render_frame()` / `render_tile()`
* **Mechanism**:
  - The outer two loops over image coordinates `(x, y)` have zero cross-iteration data dependencies.
  - The destination buffer `fb.set_pixel(x, y, ...)` writes to disjoint memory addresses in `src/renderer/framebuffer.h`.
* **Parallel Strategy**:
  - **OpenMP**: Add `#pragma omp parallel for collapse(2) schedule(dynamic, 16)` over the `y` and `x` loops.
  - **Tile / Job System**: Divide the image into $16 \times 16$ or $32 \times 32$ tiles. Push tile jobs to a multi-threaded worker pool executing `render_tile(tile_x0, tile_y0, tile_x1, tile_y1)`. Dynamic scheduling ensures high load balancing despite varying ray recursion depths per pixel.
