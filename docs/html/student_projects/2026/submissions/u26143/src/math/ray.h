#pragma once

#include "vec3.h"

// Parametric ray: P(t) = origin + t * direction
struct Ray {
    Vec3 origin;
    Vec3 direction;  // Typically normalized, but not strictly required.

    constexpr Ray() : origin(), direction(0.0, 0.0, -1.0) {}
    constexpr Ray(const Vec3& o, const Vec3& d) : origin(o), direction(d) {}

    // Evaluates the point at distance parameter t along the ray.
    constexpr Vec3 at(double t) const {
        return origin + direction * t;
    }
};