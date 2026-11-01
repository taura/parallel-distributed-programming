# Making a particle hydrodynamics solver from scratch

# Problem

When a small particle moves through a viscous liquid, it disturbs the surrounding
fluid and can move nearby particles, even if those particles are not directly
pushed. These hydrodynamic interactions matter in suspensions such as paints,
battery slurries, and catalyst inks.

The problem is to calculate the instantaneous velocities of N spherical particles,
given their positions and applied forces. I use the Rotne-Prager-Yamakawa (RPY)
approximation, which estimates how forces are transmitted through the liquid
without solving a fluid velocity field on a grid:

$$
\mathbf{U}_i = \sum_{j=1}^{N} \mathbf{M}_{ij}\mathbf{F}_j.
$$

Here, $\mathbf{F}_j$ is the non-hydrodynamic force on particle $j$, $\mathbf{U}_i$
is the velocity of particle $i$, and the $3\times3$ matrix $\mathbf{M}_{ij}$ describes
the response of particle $i$ to a force on particle $j$.

The baseline uses equal-sized, non-overlapping spheres in an unbounded liquid at
low Reynolds number, where viscous effects dominate fluid inertia. Particle inertia
is also neglected. It calculates one instantaneous response per load case; it does
not yet simulate trajectories, Brownian motion, walls, or drying. RPY also omits the
strong near-contact lubrication resistance included in more detailed models.

# How the algorithm/baseline code works

The program generates a slightly perturbed cubic lattice of particles. The first
load case applies the same force to every particle. Additional cases apply different
random forces to the same positions. These cases are not time steps.

For a sphere of radius $a$ in a liquid of viscosity $\eta$, the self-response is

$$
\mathbf{M}_{ii} = \frac{1}{6\pi\eta a}\mathbf{I}.
$$

For distinct particles, let $r$ be their center-to-center distance and
$\mathbf{e}=(\mathbf{x}_i-\mathbf{x}_j)/r$ the unit vector along their separation.
For $r\geq2a$, the RPY formula can be applied directly to a force:

$$
\mathbf{M}_{ij}\mathbf{F}_j
= A\mathbf{F}_j+B\mathbf{e}(\mathbf{e}\cdot\mathbf{F}_j),
$$

$$
A=\frac{1}{8\pi\eta r}\left(1+\frac{2a^2}{3r^2}\right),
\qquad
B=\frac{1}{8\pi\eta r}\left(1-\frac{2a^2}{r^2}\right).
$$

The code uses $a=1$ and $\eta=1$ in consistent dimensionless units. It checks that
particles do not overlap before evaluating the velocities. The main computation is
the following direct summation (pseudocode):

```cpp
for (i = 0; i < N; ++i) {
    U[i] = F[i] / (6 * pi * eta * a);
    for (j = 0; j < N; ++j) {
        if (i == j) continue;
        r_vector = x[i] - x[j];
        r = length(r_vector);
        e = r_vector / r;
        A = (1 + 2*a*a/(3*r*r)) / (8*pi*eta*r);
        B = (1 - 2*a*a/(r*r)) / (8*pi*eta*r);
        U[i] += A*F[j] + B*e*dot(e, F[j]);
    }
}
```

All particle pairs are included. The algorithm takes $O(N^2)$ time per load case
and $O(N)$ working memory because it does not store the full mobility matrix.
`main.cpp` handles inputs, case generation, timing, and output; `rpy.cpp` implements
the computation and analytical checks; `ppm.cpp` draws optional images. Their headers
provide the shared interfaces. Only C++17 and the standard library are required.

Compile and run from the source directory:

```sh
c++ -O3 -std=c++17 main.cpp rpy.cpp ppm.cpp -o /tmp/RPYSolver
/tmp/RPYSolver --check
/tmp/RPYSolver --n 1000 --cases 1 --seed 42 --output /tmp/result.csv
/tmp/RPYSolver --n 4000 --cases 10
```

`--n` controls the particle count and `--cases` controls the number of load cases.
These allow the workload to be adjusted to the available machine. The commands
above are example sizes, not a claim about runtime on Miyabi.

`--check` compares results with single-sphere and two-sphere analytical predictions,
checks a rotated configuration and viscosity scaling, and verifies overlap rejection.
CSV output contains positions, forces, and velocities for the last case.

A small visualization example places 49 particles in a plane and applies a force
only to the central particle. The calculation still uses the three-dimensional RPY
formula. Orange dots mark directly forced particles; blue dots have no applied
force. Arrows show the XY components of velocity on a common linear scale:

```sh
/tmp/RPYSolver --demo --image /tmp/velocity.ppm --output /tmp/demo.csv
convert /tmp/velocity.ppm /tmp/velocity.png
```

PPM output needs no graphics library; the optional conversion command uses
ImageMagick. The arrows represent instantaneous velocities, not trajectories.

The program reports kernel time separately from geometry validation and image
output, as well as total execution time. Geometry validation is currently also
$O(N^2)$, so it may become significant after the main kernel is accelerated.

Formula reference: Wajnryb et al. (2013), [Generalization of the
Rotne-Prager-Yamakawa mobility and shear disturbance tensors](https://doi.org/10.1017/jfm.2013.402),
Eq. 3.12.

# Where is parallelism

- **Across target particles:** each outer-loop iteration computes a different
  `U[i]`. Positions and forces are read-only, so different threads can calculate
  different target particles without conflicting writes.
- **Inside each sum:** source-particle contributions can be evaluated with SIMD
  instructions or GPU threads, followed by a reduction. Changing the summation
  order may introduce small floating-point differences.
- **Data reuse:** all target particles read the same source arrays. CPU cache
  blocking and GPU shared-memory tiling may reduce repeated memory access.
  Separate coordinate arrays may also improve vectorization and GPU access.

I plan to compare serial, multicore, and GPU implementations at the same particle
count and accuracy. The performance study will examine scaling with N and thread
count, data layout, and the cost of reductions and data transfers. Both kernel
speedup and total runtime matter. Parallel results will be compared against the
serial velocities using numerical error measures, not just a checksum.
