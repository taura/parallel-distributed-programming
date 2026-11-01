# Making 2D Acoustic Wave Simulation from Scratch

# Problem
We simulate acoustic wave propagation on a 2D square grid using the finite difference method.
A Ricker wavelet source at the center of the domain generates wavefields with both positive and negative amplitudes, forming concentric wavefronts spreading outward.

The program solves the 2D acoustic wave equation:
$$ \frac{\partial^2 u}{\partial t^2} = c^2 \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) $$

where:
- u(x,y,t) is the wave displacement at position (x,y) and time t
- c is the constant wave speed

The grid size N (N × N points) and the number of time steps Nt are passed from the command line, so we can easily adjust the total computation time.

# How the algorithm/baseline code works
This is a single-threaded serial baseline implementation written in C++.

We use three arrays to store the wave field at three consecutive time steps:
- u_prev: wave field at time t-1
- u_curr: wave field at time t
- u_next: wave field at time t+1 (to be computed)

For each time step, we inject a Ricker wavelet source at the center, then use the second-order finite difference scheme:
1. Compute the Laplacian (second spatial derivative) of u_curr for every inner grid point.
2. Update u_next using the time-stepping formula:
$$ u_{next}[j][i] = 2 \cdot u_{curr}[j][i] - u_{prev}[j][i] + c^2 \frac{\Delta t^2}{\Delta x^2} \nabla^2 u_{curr}[j][i] $$
3. Swap the three arrays to prepare for the next iteration.
4. After all time steps, write a snapshot of the final wave field to a PPM image file. The visualization uses a gray background, red for positive amplitudes and blue for negative amplitudes, which is a common style for seismic wavefield plots.

The CFL condition (Δt ≤ Δx / (c·√2)) is respected to ensure numerical stability.

This serial implementation supports adjustable grid size N and time steps Nt via command-line arguments.
Small settings such as N=256, Nt=300 are used for fast testing and visualization.
Larger settings (e.g., N=1024, Nt=2000) increase computation cost and will run for several minutes, satisfying the requirement of this assignment.

# Where is parallelism
The computation has abundant parallelism:
- At each time step, the update of every inner grid point (j,i) is **independent** of all other grid points in the same time step.
- Each point only reads values from u_curr and u_prev (read-only), and writes to its own position in u_next.
- There is no data dependency between different grid points within the same time step.

