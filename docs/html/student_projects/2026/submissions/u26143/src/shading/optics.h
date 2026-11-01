#pragma once

#include <cmath>

#include "math/math_utils.h"
#include "math/vec3.h"

// Perfect reflection direction.
inline Vec3 reflect(const Vec3& incident, const Vec3& normal) {
    return incident - 2.0 * dot(incident, normal) * normal;
}

// Snell refraction. Returns false for total internal reflection.
inline bool refract(
    const Vec3& incident,
    const Vec3& normal,
    double eta_i,
    double eta_t,
    Vec3& refracted_out
) {
    const double eta = eta_i / eta_t;
    const double cos_i = -dot(incident, normal);
    const double sin2_t = eta * eta * (1.0 - cos_i * cos_i);

    if (sin2_t > 1.0) {
        return false; // Total internal reflection.
    }

    const double cos_t = std::sqrt(1.0 - sin2_t);
    refracted_out = eta * incident + (eta * cos_i - cos_t) * normal;
    return true;
}

// Schlick approximation for Fresnel reflectance.
inline double fresnel_schlick(double cos_theta, double eta_i, double eta_t) {
    const double r0 = (eta_i - eta_t) / (eta_i + eta_t);
    const double r0_sq = r0 * r0;
    const double m = 1.0 - math::clamp(cos_theta, 0.0, 1.0);
    return r0_sq + (1.0 - r0_sq) * m * m * m * m * m;
}