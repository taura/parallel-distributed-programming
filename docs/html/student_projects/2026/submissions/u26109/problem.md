# Simulation of Two-Dimensional Flow Around an Airfoil

# Problem

This project studies two-dimensional flow around a fixed airfoil using the Lattice Boltzmann Method (LBM), with an emphasis on parallel computing performance.

The project will include:

- Generating a NACA four-digit airfoil and representing it as solid cells on a regular lattice, with configurable orientation.
- Simulating flow around the fixed airfoil and its downstream wake.
- Computing the lift and drag force histories.
- Visualizing the velocity field and plotting lift and drag.
- First completing a serial C++ CPU solver, then implementing and optimizing a CUDA version and evaluating its correctness and performance.

The initial experiments will use one fixed airfoil geometry and orientation. The baseline uses a channel-flow configuration with a parabolic inlet profile; this setup will be documented explicitly when interpreting the flow and forces. The reported forces will not be presented as nondimensional lift and drag coefficients without an explicit normalization.

Following the course requirements, I will complete the project in two stages. First, I will develop a working serial C++ version of the airfoil simulation based on the selected numerical scheme and reference code. Second, I will port the computationally intensive operations to CUDA and optimize their GPU execution. The C++ version will serve as the correctness and performance baseline for the CUDA version.

# How the algorithm/baseline code works

The airfoil generator produces obstacle-cell coordinates, which are combined with the grid dimensions, Reynolds number, simulation duration, and velocity parameters to form the solver input.

The solver uses the D2Q9 lattice model, storing nine particle distribution values per cell. A Two-Relaxation-Time (TRT) collision model updates these distributions. Macroscopic density and velocity are computed from their moments. Streaming transfers distributions to neighboring cells, while boundary operations handle the domain edges and the solid airfoil surface. The baseline uses Zou-He boundary treatment and bounce-back at obstacles.

The following pseudocode summarizes the CPU baseline at a high level:

```text
load simulation parameters and airfoil obstacle coordinates
initialize lattice cells and distribution values

for each time step:
    for each lattice cell:
        apply inlet and domain-boundary updates
        compute density and velocity
        compute equilibrium and TRT collision
        apply obstacle bounce-back

    for each lattice cell:
        stream distributions to neighboring cells

    if this is an output step:
        accumulate force contributions into drag and lift
        save the velocity field and forces

generate a velocity-field animation and force-history plots
```

For a grid of Nx by Ny cells and T time steps, the core computation costs O(T * Nx * Ny), with O(Nx * Ny) memory. Geometry generation and plotting are preprocessing and postprocessing tasks.

In the first stage, all solver loops and force accumulation will execute sequentially in C++. I will check the resulting velocity fields and force histories before beginning the CUDA implementation. Geometry generation and visualization may reuse the repository's Python scripts; the numerical solver itself will be implemented in C++ and then accelerated with CUDA.

# Where is parallelism

- **Cell calculations:** Macroscopic-variable and collision calculations operate on individual cells. A CUDA kernel can assign one GPU thread to each lattice cell, using a two-dimensional grid of thread blocks. Boundary operations must be checked separately for neighboring-data dependencies.
- **Streaming:** Each cell exchanges distributions with its neighbors. Separate source and destination buffers can support parallel streaming, with each destination distribution assigned a unique writer to avoid races.
- **Force accumulation:** Contributions from airfoil-adjacent cells can be computed concurrently on the GPU. These contributions must then be summed using a parallel reduction or atomic operations. I will investigate reducing atomic contention through block-level partial sums.
- **Synchronization:** Dependent stages must finish before the next stage begins. Separate kernel launches in the same CUDA stream can enforce ordering across the full grid; block-local synchronization alone is insufficient for dependencies between blocks. Consecutive time steps remain dependent.

After obtaining a correct CUDA implementation, I will investigate a Structure of Arrays (SoA) layout for coalesced memory access, thread-block dimensions, and force reduction. Simulation data will remain in GPU memory across time steps, with results transferred to the CPU only when needed for output or validation. Optimizations will be selected based on measured bottlenecks.

CUDA correctness will be checked against the serial C++ reference using velocity-field differences and lift/drag histories, with matching inputs, numerical parameters, and floating-point precision. Comparisons will allow for floating-point differences caused by arithmetic and parallel summation order. This checks implementation consistency; it does not by itself establish physical accuracy.

Performance experiments will vary lattice resolution and CUDA thread-block dimensions. CPU and GPU runs will use identical inputs and step counts for each comparison. Grid-refinement experiments will preserve the intended geometry and physical setup. Measurements will include serial CPU solver time, GPU computation time, and GPU solver time including device allocation and required CPU-GPU transfers. Speedup will be reported with the timing scope stated explicitly:

$$
S = \frac{T_{\mathrm{CPU,serial}}}{T_{\mathrm{GPU}}}.
$$

Solver timing will include numerical updates and force calculations at a fixed sampling interval, but exclude file writing, geometry generation, and visualization. GPU measurements will account for asynchronous execution by using CUDA events for device computation and synchronizing before stopping host-side timers. Each configuration will be warmed up and measured multiple times.
