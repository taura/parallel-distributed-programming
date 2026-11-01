
* _this is an example file you should finish_
* write problem description, how the algorithm/baseline code works, and where you think you can apply parallel computing
* if you are looking at this file from Jupyter, you can see the preview by right-clicking on the cell and "Show Markdown Preview"

# Making ray tracing algorithm from scratch

## Problem

Render a two-dimensional image of a three-dimensional scene from a virtual camera. For each pixel, the renderer must find the first surface hit by a ray and estimate its color from reflected or scattered light. This example scene contains a red metal sphere, flat gray terrain, and a sky gradient. It produces a 2000 × 1000 pixel, 24-bit BMP image.

The work is computationally expensive: 16 samples per pixel produce 32 million primary rays, and a ray may generate further rays when it hits a surface. The current implementation is a single-threaded baseline for exploring parallel computing.

## How the algorithm/baseline code works

1. [main.cpp](src/main.cpp) creates the scene and loops over every pixel. For each of its 16 samples, it adds a random offset within the pixel and asks the camera in [ray.h](src/ray.h) for a ray. This sampling helps smooth object edges and diffuse lighting.

2. [Scene::render](src/scene.h) tests the ray against every registered [Shape](src/shape.h), keeps the closest valid hit, and calls the surface's [Material](src/material.h). The terrain is a horizontal plane with a gray Lambertian material that scatters rays in random directions. The sphere has a red Metal material with zero fuzz, so it reflects rays in the mirror direction.

3. The renderer recursively traces each scattered ray and multiplies the returned color by the material's color. 

4. [main.cpp](src/main.cpp) averages the 16 colors for each pixel. [bitmap.h](src/bitmap.h) clamps the result to 8-bit RGB values and writes `render.bmp` as an uncompressed 24-bit BMP with BGR pixel order and padded rows.

## Where is parallelism

The baseline currently renders pixels one after another. Different pixels do not depend on one another, so rows or rectangular tiles are the main units of work to distribute across CPU threads or GPU workers. The 16 samples within a pixel are also independent.

