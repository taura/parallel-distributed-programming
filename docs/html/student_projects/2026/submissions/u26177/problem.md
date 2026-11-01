# Building a Ray Tracer from Scratch

## Problem

Ray tracing generates images of three-dimensional scenes by following rays of light. For each image pixel, the program traces a ray from the camera, finds the first object it hits, and computes the pixel color from the material, light sources, and visibility.

The scenes are built from procedurally generated geometry, so no external model or texture files are needed. Four scenes are available:

- `character`: an original stylized magical-girl figure built from ellipsoids, triangles, and other primitives.
- `gallery`: an exhibition hall with pillars, platforms, and reflective spheres.
- `transport`: a small room for observing light transport, with red and green walls, diffuse, mirror, and glass spheres, and an overhead light.
- `spheres`: spheres with different materials above a ground surface.

The goal is to explore rendering from direct lighting to multiple reflections, refraction, and indirect illumination, and to compare image quality, computational work, and runtime. The program writes a PPM image and reports timing, brightness, and other summary statistics.

## How the baseline code works

The baseline is a serial C++17 program that uses no external libraries. `raytrace.cc` contains vector operations, ray-object intersections, camera logic, and image output. Scene construction and path tracing are implemented in header files in the same directory.

### From pixels to colors

1. For each pixel, generate one or more camera rays according to the image resolution and camera view. Multiple samples use random offsets within the pixel to reduce aliasing.
2. Find intersections with spheres, boxes, ellipsoids, triangles, and the ground, and select the nearest valid hit.
3. Compute the color using the selected rendering algorithm. Rays that miss the scene sample the sky background.
4. Average the samples for the pixel, apply tone mapping and gamma correction, and write the result to a PPM image.

### Lighting modes

The `--integrator` option selects the rendering mode:

- `whitted`: computes ambient light, Phong diffuse and specular lighting, shadows, and ideal mirror reflections up to a specified depth.
- `path`: uses Monte Carlo path tracing. Rays can bounce from diffuse surfaces, estimating indirect light and color bleeding. Next-event estimation (NEE) samples area lights directly, and multiple importance sampling (MIS) combines this with material-direction sampling. Russian roulette can terminate longer paths.

Path tracing also supports:

- Glass reflection and refraction using the index of refraction, Snell’s law, and Fresnel reflection probability, with radiance weighting for refraction.
- `--bsdf ggx` for GGX microfacet reflection, Smith masking, Schlick Fresnel, and visible normal distribution function (VNDF) sampling. These model rough metallic highlights and blurred reflections.
- `--adaptive` for adaptive sampling based on online variance estimates in linear RGB. A pixel may stop before its sample limit when the stopping criterion is met. `--samples-out` can write a heat map of sample counts.

Path tracing is a random estimator, so images with few samples can be noisy. Increasing `--spp` generally makes the result more stable but requires more computation. The adaptive stopping criterion is empirical and does not guarantee a strict error bound.

### Running and parameters

After compiling on Miyabi, an example run is:

```bash
./raytrace --integrator path --bsdf ggx --scene character --detail 1 --w 240 --h 300 --spp 16 --light-samples 1 --max-depth 6 --seed 42 --out character.ppm
```

| Parameter | Purpose |
|---|---|
| `--scene` | Select `character`, `gallery`, `transport`, or `spheres`. |
| `--integrator` | Select `whitted` or `path`. |
| `--bsdf` | Select the ideal diffuse/mirror model (`ideal`) or GGX/VNDF (`ggx`) for path tracing. |
| `--roughness` | Override the roughness of opaque GGX materials, from 0.02 to 1; otherwise each material uses its own setting. |
| `--w`, `--h` | Set image width and height. |
| `--spp` | Set the maximum samples per pixel; with adaptive sampling disabled, this is the fixed sample count. |
| `--max-depth` | Set the maximum number of scattering events per path; the path-tracing default is 8. |
| `--light-samples` | Set the number of direct-light samples per area light at each hit. |
| `--adaptive`, `--min-spp` | Set the adaptive stopping threshold and minimum samples per pixel. |
| `--seed` | Fix the random seed for reproducible scene generation and sampling. |
| `--out` | Set the output PPM filename. |

## Where is parallelism

- Pixels are largely independent: their camera rays, subsequent paths, and accumulated colors affect only their own pixel. Image rows, tiles, or individual pixels could be assigned to multicore CPU threads or GPU threads.
- Samples within one pixel could also be calculated in parallel and combined with a reduction. With adaptive sampling, pixels may require different numbers of samples, causing load imbalance.
- Scattering events along one path depend on previous events and cannot all be computed at once, but many different paths can be traced in parallel.

Potential limits include branch divergence when rays hit different objects and materials, and load imbalance from varying path lengths. The current program tests scene objects one by one for each ray, so intersection cost increases with scene size. Random number generation, triangle and ellipsoid intersections, path traversal, and output memory access can also affect performance. A bounding volume hierarchy (BVH) and CPU or GPU parallelization are possible directions for later implementation and analysis.

## Correctness checks

The program provides `--check-geometry`, `--check-transport`, and `--check-advanced` self-checks covering geometry intersections, shadow distances, refraction and total internal reflection, sampling probability densities, GGX reflection-energy estimates, and adaptive-sampling statistics. Rendered images can also be inspected for plausible colors, shadows, and reflections.

These checks can reveal implementation errors, but they do not prove correctness for every scene and parameter combination. Performance experiments should record the parameters and runtime, use a fixed seed, and examine how the image and runtime change as sample count or maximum depth increases.
