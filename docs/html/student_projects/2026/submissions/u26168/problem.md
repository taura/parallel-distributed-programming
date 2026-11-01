# Making a 2D Navier–Stokes Solver from scratch

# Problem
The goal of this project is to develop a program that numerically solves the two-dimensional incompressible Navier–Stokes equations using the finite difference method.

The Navier–Stokes equations describe how the velocity and pressure of a fluid change over time. In this project, I will simulate the Taylor–Green vortex, which is a flow consisting of regularly arranged vortices. Since an analytical solution is available, I can compare the numerical results with the exact solution to check their accuracy.

# How the algorithm/baseline code works
First, the advection and diffusion terms of the Navier–Stokes equations are calculated using the finite difference method to obtain a temporary velocity field.

However, this temporary velocity field does not necessarily satisfy the incompressibility condition. Therefore, the pressure Poisson equation is solved, and the velocity is corrected using the pressure gradient.

The pressure Poisson equation is converted into a system of linear equations using the finite difference method and solved using the Conjugate Gradient (CG) method.

# Where is parallelism

Many calculations in the program can be performed independently at each grid point, so they can be parallelized using a GPU.

- Temporary velocity calculation: Each grid point uses the velocities of neighboring points, so the calculations can be performed in parallel.

- Divergence and Laplacian calculations: These calculations can be performed independently at each grid point.

- Velocity correction: The velocity at each grid point can be updated in parallel using the calculated pressure.

