# Making a 2D waterdrop simulator from scratch

# Problem

I chose a 2D incompressible fluid simulation where the liquid is restricted to an
initial waterdrop region. The simulation tracks velocity fields (`u`, `v`) and a
liquid fraction field (`liq` in `[0,1]`) on a regular grid.

The drop falls by gravity and deforms while staying as a compact liquid blob. The
program renders a series of PPM images (`drop_0000.ppm`, `drop_0001.ppm`,
...) so I can see time evolution, plus one final image.

This is interesting because it is a realistic PDE-based application, it gives
visually meaningful output, and it has enough computation to study performance.

# How the algorithm/baseline code works

The code is in `fluid2d.cc` and is fully serial C++17 with no external libraries.

Main loop per time step is:

1. Add body force from gravity mainly inside liquid cells.
2. Advect velocity with a semi-Lagrangian method (bilinear sampling).
3. Apply viscosity diffusion by Jacobi iterations (implicit diffusion).
4. Projection step for incompressibility:
   - compute velocity divergence,
   - solve Poisson equation for pressure using Jacobi iterations,
   - subtract pressure gradient from velocity.
5. Advect liquid fraction (`liq`) using the projected velocity.
6. Sharpen the liquid-air interface and damp velocity outside the drop.
7. Run a short second projection to keep divergence small after damping.
8. Periodically render a frame as a PPM image.

The code prints:

- simulation parameters,
- compute time of the main loop,
- maximum absolute divergence (correctness indicator),
- liquid volume and max speed,
- number of rendered frames and final image name.

All key parameters are configurable from command line (`nx`, `ny`, `steps`, `dt`,
`visc`, `jacobi`, `gravity`, `frame-every`, output names), so runtime can be scaled
from seconds to minutes.

# Where is parallelism

Most heavy loops are per-cell stencil updates and are data-parallel:

- velocity advection,
- liquid fraction advection,
- divergence calculation,
- pressure Jacobi iteration,
- gradient subtraction,
- frame rendering loop.

These loops can later be parallelized with multicore CPU threads and GPU kernels.
Likely speedup limits are memory bandwidth and the iterative pressure solve
(many sweeps over the whole grid each step).
