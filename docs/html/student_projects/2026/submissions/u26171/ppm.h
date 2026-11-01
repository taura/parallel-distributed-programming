#ifndef RPY_PPM_H
#define RPY_PPM_H

#include "rpy.h"
#include <string>
#include <optional>

namespace rpy {
struct ImageOptions {
    std::optional<double> z; // Automatic: occupied Z value nearest cloud midpoint.
    double thickness = 2.0; // Full slab thickness in simulation length units.
    std::size_t max_arrows = 225;
};

// XY view of a Z slab with spatially separated representative particles.
// Arrows are individual velocities, never averages; scale is linear per image.
// Orange dots have nonzero applied force; markers are not to scale.
// Writes a binary P6 PPM with no external graphics dependencies.
void write_velocity_ppm(const std::string& path,
                        const std::vector<Vec3>& positions,
                        const std::vector<Vec3>& forces,
                        const std::vector<Vec3>& velocities,
                        bool planar_demo,
                        const ImageOptions& options = {});
}
#endif
