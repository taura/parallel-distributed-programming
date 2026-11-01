
* _this is an example file you should finish_
* write problem description, how the algorithm/baseline code works, and where you think you can apply parallel computing
* if you are looking at this file from Jupyter, you can see the preview by right-clicking on the cell and "Show Markdown Preview"

# Making fluid simulation using SPH method from scratch

# Problem
Simulate a 2D fluid dynamics, where a block of water collapses under gravity inside a $1\,\mathrm{m}\times1\,\mathrm{m}$ box, using the SPH (Smoothed Particle Hydrodynamics) method. 

For each particle, the density, pressure, and acceleration are computed as follows:

$$
\rho_i = \sum_j m\,W(r_{ij}),\qquad
P_i = c^2 \max(0,\ \rho_i - \rho_0)
$$

$$
\frac{d\mathbf v_i}{dt} = -\sum_{j\neq i} m\left(\frac{P_i}{\rho_i^2} + \frac{P_j}{\rho_j^2} + \Pi_{ij}\right)\nabla W(r_{ij}) + \mathbf g
$$

The density uses the 2D poly6 kernel, and the force uses the gradient of the 2D spiky kernel:

$$
W(r) = \frac{4}{\pi h^8}\left(h^2 - r^2\right)^3,\qquad
\nabla W(r_{ij}) = -\frac{30}{\pi h^5}\,(h - r)^2\,\frac{\mathbf r_{ij}}{r}\qquad (r < h)
$$

where $\rho_i$, $P_i$, and $\mathbf v_i$ are the density, pressure, and velocity of particle $i$; $\mathbf r_{ij} = \mathbf r_i - \mathbf r_j$ is the relative position and $r = |\mathbf r_{ij}|$; $m$ is the particle mass; $\rho_0 = 1000\,\mathrm{kg/m^3}$ is the reference density of water; $c = 35\,\mathrm{m/s}$ is the numerical sound speed; $\Pi_{ij}$ is the artificial viscosity; $\mathbf g$ is gravity; and $h$ is the kernel support radius ($W = 0$ for $r \ge h$). Here $h = 2\Delta = 0.0175\,\mathrm{m}$, where $\Delta$ is the initial particle spacing, so each particle interacts with about 12 neighbors. $\Delta t$ is chosen to satisfy the CFL condition $\Delta t \le 0.25\,h/c$.

# How the algorithm/baseline code works
Each time step calls three functions:
```
for (int step = 0; step < steps; ++step) {
    compute_density(p, h, mass);
    compute_acceleration(p, h, mass);
    integrate(p, dt);
    ...
}
```
## 1. Density and pressure:
```
void compute_density(std::vector<Particle>& p, double h, double mass) {
    const double h2 = h * h;
    const double coefficient = 4.0 / (pi * h2 * h2 * h2 * h2);
    for (std::size_t i = 0; i < p.size(); ++i) {
        double rho = 0.0;
        for (std::size_t j = 0; j < p.size(); ++j) {
            const double dx = p[i].x - p[j].x;
            const double dy = p[i].y - p[j].y;
            rho += mass * density_kernel(dx * dx + dy * dy, h2, coefficient);
        }
        p[i].rho = rho;
        // No tensile (negative) pressure near the free surface.
        p[i].pressure = sound_speed * sound_speed * std::max(0.0, rho - rho0);
    }
}
```

## 2. Acceleration:
```
void compute_acceleration(std::vector<Particle>& p, double h, double mass) {
    const double h2 = h * h;
    const double grad_coefficient = -30.0 / (pi * h2 * h2 * h);
    for (std::size_t i = 0; i < p.size(); ++i) {
        double ax = 0.0, ay = -gravity;
        for (std::size_t j = 0; j < p.size(); ++j) {
            if (i == j) continue;
            double dx = p[i].x - p[j].x;
            double dy = p[i].y - p[j].y;
            double r2 = dx * dx + dy * dy;
            if (r2 >= h2) continue;
            if (r2 == 0.0) {
                // Give distinct coincident particles opposite finite directions.
                dx = i < j ? 0.125 * h : -0.125 * h;
                dy = 0.0;
                r2 = dx * dx;
            }

            const double r = std::sqrt(r2);
            // Gradient of the 2D spiky kernel points towards particle j.
            const double grad_factor = grad_coefficient * (h - r) * (h - r) / r;
            const double dv_dot_dr = (p[i].vx - p[j].vx) * dx
                                   + (p[i].vy - p[j].vy) * dy;
            double viscosity = 0.0;
            if (dv_dot_dr < 0.0) {
                const double rho_avg = 0.5 * (p[i].rho + p[j].rho);
                viscosity = -alpha * sound_speed * h * dv_dot_dr
                          / (rho_avg * (r2 + 0.01 * h2));
            }
            const double pair = -mass * (p[i].pressure / (p[i].rho * p[i].rho)
                                       + p[j].pressure / (p[j].rho * p[j].rho)
                                       + viscosity) * grad_factor;
            ax += pair * dx;
            ay += pair * dy;
        }
        p[i].ax = ax;
        p[i].ay = ay;
    }
}
```

## 3. Update and wall reflection:
```
void integrate(std::vector<Particle>& p, double dt) {
    for (Particle& q : p) {
        // Symplectic Euler; all accelerations were computed from the old state.
        q.vx += dt * q.ax;
        q.vy += dt * q.ay;
        q.x += dt * q.vx;
        q.y += dt * q.vy;
        if (q.x < wall) { q.x = 2.0 * wall - q.x; if (q.vx < 0) q.vx *= -wall_damping; }
        if (q.x > 1.0 - wall) { q.x = 2.0 * (1.0 - wall) - q.x; if (q.vx > 0) q.vx *= -wall_damping; }
        if (q.y < wall) { q.y = 2.0 * wall - q.y; if (q.vy < 0) q.vy *= -wall_damping; }
        if (q.y > 1.0 - wall) { q.y = 2.0 * (1.0 - wall) - q.y; if (q.vy > 0) q.vy *= -wall_damping; }
        q.x = std::clamp(q.x, wall, 1.0 - wall);
        q.y = std::clamp(q.y, wall, 1.0 - wall);
    }
}
```

# Where is parallelism
- The baseline is serial. compute_density and compute_acceleration check every pair of particles, so each time step costs $O(N^2)$.
- The outer loops over i in compute_density and compute_acceleration, integrate are independent, and each i writes only to p[i], so they can be parallelized.
- Particles farther than $h$ do not interact, so a cell list that searches only nearby particles reduces the cost.