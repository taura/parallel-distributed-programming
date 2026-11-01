# Monte Carlo Trajectory Optimization for a Mobile Robot

## 1. Problem and motivation

A wheeled robot must travel from a start pose to a goal in a bounded 2D environment with circular obstacles. The problem is to find a collision-free trajectory that reaches the goal efficiently while keeping steering inputs reasonably smooth. This simplified task is relevant to local motion planning in mobile robotics and autonomous driving.

## 2. Mathematical model

The robot has planar state `(x, y, theta)` and follows a unicycle model with forward velocity `v` and angular velocity `omega`:

- `dx/dt = v cos(theta)`
- `dy/dt = v sin(theta)`
- `dtheta/dt = omega`

We approximate the motion using a fixed time step `dt = 0.14 s` and midpoint integration. A nominal heading controller points toward the goal, and the optimizer adds a smooth, sampled angular correction. An input trajectory is represented by 12 steering-correction values interpolated across the time horizon. Velocity and steering rate have upper bounds.

## 3. Serial baseline algorithm

This implementation uses the **Cross-Entropy Method (CEM)**, a form of population-based Monte Carlo optimization:

1. Initialize a Gaussian sampling distribution over the 12 steering-correction parameters.
2. Sample `N` candidate control trajectories (independently).
3. Simulate each candidate over `H` steps using the robot model.
4. Evaluate a cost combining goal progress, final goal distance, steering effort, proximity to obstacles, and large penalties for actual collisions or crossing the map boundary.
5. Keep the best 10% of candidates and update the Gaussian mean and standard deviation.
6. Repeat for a specified number of optimization iterations and return the lowest-cost trajectory observed.

This is a simple **serial C++17 baseline**, not a GPU implementation and not the full Model Predictive Path Integral (MPPI) controller. It uses no external C++ libraries.

## 4. Inputs and outputs

Build with the standard GCC compiler:

```bash
g++ -std=c++17 -O3 -Wall -Wextra main.cpp -o trajectory
```

Example run (in the `code/` directory):

```bash
./trajectory --samples 1400 --steps 90 --iterations 7 --seed 42 --output trajectory.ppm
```

Runtime options:

- `--samples N`: sampled trajectories per optimization iteration (default `1400`)
- `--steps H`: motion-integration steps per trajectory (default `90`)
- `--iterations K`: CEM optimization iterations (default `7`)
- `--seed S`: deterministic random seed (default `42`)
- `--output FILE.ppm`: PPM image path (default `trajectory.ppm`)

Outputs are:

- `trajectory.ppm`: static 2D image showing ALL candidate trajectories from every CEM iteration with a light-to-dark blue color gradient and a numeric iteration legend, plus the green selected path and obstacles
- `best_trajectory.csv`: selected path with time, position and heading
- `summary.txt`: best cost, terminal error, obstacle clearance, collision step count, execution time, and visualized iteration/trajectory counts

Convert the PPM to PNG using the provided notebook command, `convert code/trajectory.ppm code/trajectory.png`.

## 5. Validation

For the default seed and parameters, the generated outputs can be checked by:

- ensuring the minimum obstacle clearance is positive and the number of collision steps is zero;
- checking that the selected trajectory ends near the target location;
- comparing the cost of the selected trajectory with the nominal non-optimized trajectory;
- viewing the resulting static 2D image and checking that the green best trajectory avoids the red circular obstacles and their safety margins.

Obstacle clearance is measured from the robot boundary to the physical obstacle boundary; positive clearance indicates separation at the sampled trajectory states. This discrete check does not guarantee continuous-time safety between integration steps.

The visualization shows **all** Monte Carlo candidate trajectories, using lighter blue for earlier iterations and darker blue for later iterations. A numeric color legend maps colors to iteration numbers. It also shows a **green** selected best trajectory, **red** physical obstacles, **orange** inflated exclusion boundaries, a **blue** start marker and a **yellow** goal marker.

## 6. Parallelization opportunity

The dominant computation is evaluating independent candidate trajectories. During each CEM iteration, different candidates can be simulated and scored in parallel using **OpenMP** on CPU cores or **CUDA** on GPU threads. The small distribution-update step is a reduction over the elite samples. Larger values of `--samples`, `--steps`, and `--iterations` scale the workload and will allow performance and speedup experiments in later course assignments.
