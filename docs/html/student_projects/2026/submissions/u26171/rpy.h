#ifndef RPY_H
#define RPY_H

#include <cstddef>
#include <vector>

namespace rpy {

struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;
};

struct Parameters {
    double radius = 1.0;
    double viscosity = 1.0;
};

// Equal-radius spheres in an unbounded, quiescent Newtonian fluid.
// Forces are non-hydrodynamic forces. Only translational mobility is used.
// No periodic images, walls, lubrication, Brownian noise or time integration.
void validate_geometry(const std::vector<Vec3>& positions,
                       const Parameters& parameters);

// Preconditions: validated non-overlapping geometry; finite forces;
// positions, forces and velocities have equal sizes and do not alias.
// Overwrites velocities. Does not allocate, print or modify the input.
void apply_mobility_serial(const std::vector<Vec3>& positions,
                           const std::vector<Vec3>& forces,
                           const Parameters& parameters,
                           std::vector<Vec3>& velocities);

// Throws on failure. Analytical checks of the implementation, not validation
// of RPY against exact many-sphere hydrodynamics or experiments.
void run_self_checks();

} // namespace rpy
#endif
