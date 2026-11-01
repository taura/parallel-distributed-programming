# Making a path tracer from scratch

## Problem

Render a small room (a "Cornell box") containing a matte, a mirror and a glass
sphere, lit by a ceiling lamp. The color of a pixel is the light that reaches the
camera through it, after bouncing any number of times between surfaces. Summing
over all possible light paths cannot be done exactly, so we estimate it by random
sampling (Monte Carlo): trace many random paths per pixel and average them. More
samples per pixel (spp) give less noise (the error shrinks as 1/sqrt(spp)) but
cost proportionally more computation. This is a simplified version of the
technique used for film and photorealistic rendering.

## How the baseline code works

The program is serial C++17 and uses only the standard library.
For every pixel and sample it shoots a ray from the camera and follows it: find the closest surface,
add any light the surface emits, then choose a new direction from the material
and dim the path by the surface color. A path stops when it hits the lamp or the background, reaches the
bounce limit, or is randomly terminated. The pixel color is the average of
its samples.

Objects are spheres or triangle meshes. Finding which triangle a ray hits first
would be slow if every triangle were tested, so a bounding volume hierarchy (BVH),
a tree of boxes that each contain a group of triangles, lets a ray skip any group
whose box it misses. Each pixel/sample pair has its own deterministic random seed,
so the image is identical no matter in which order the samples are computed.

Command-line options control the image size (`-w`, `-h`), samples per pixel (`-s`),
maximum path depth (`-d`), scene (`-S`: 0 analytic spheres, 1 triangle meshes,
2 white-environment test) and mesh detail (`-D`: 20 * 4^D triangles per object).
Run time is proportional to w * h * spp. The program writes a PPM image and reports
scene-building time, rendering time, ray count, mean color and a checksum.

Build and run on Miyabi (one core):

```
clang++ -O3 -std=c++17 -Wall -o pt *.cc
./pt -w 256 -h 256 -s 256 -S 1 -D 5 -o image.ppm
```
```
scene=1 detail=5 triangles=61452 spheres=0 bvh_nodes=32767 bvh_depth=15
size=256x256 spp=256 max_depth=8
build 0.021 s, render 63.641 s
rays=60470550 (3.60 per sample), 0.950 Mrays/s
mean color = (0.477733, 0.447023, 0.413393)
checksum = b806698ad56fcdd1
```

Results can be checked in several ways:

- the image shows the expected effects: red/green light bleeding onto the white
  walls, soft shadows, reflections, and light focused under the glass sphere;
- the white-environment test (`-S 2`) must give exactly 1 in every pixel, and it does;
- the mesh scene and the analytic sphere scene give the same mean color;
- the checksum is the same on Miyabi and on an x86 PC, and must stay the same in
  parallel versions.

## Where is parallelism

Pixels, and samples within a pixel, are independent: a multicore CPU can distribute
pixels or image tiles among threads, and a GPU can assign pixels or samples to
threads, with no communication except combining sample results and ray counts.
The main obstacles are uneven work (paths have different lengths, e.g. rays inside
the glass sphere bounce many times while rays leaving the room stop at once), which
unbalances CPU threads and makes GPU threads in a warp diverge; irregular memory
access during BVH traversal, which stresses the caches; and the serial scene/BVH
construction, which becomes noticeable only when the rendering itself is fast.
SIMD instructions can be used to test one ray against several boxes, or several
rays against one box, at the same time.
