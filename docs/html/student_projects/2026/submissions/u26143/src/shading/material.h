#pragma once

#include "math/vec3.h"

enum class MaterialType {
    Diffuse,
    Mirror,
    Dielectric
};

struct Material {
    MaterialType type = MaterialType::Diffuse;

    // Base color used by diffuse and checkerboard.
    Vec3 albedo = Vec3(0.8, 0.8, 0.8);

    // Blinn-Phong specular highlight parameters.
    Vec3 specular_color = Vec3(1.0, 1.0, 1.0);
    double shininess = 64.0;

    // Optical parameters.
    double reflectivity = 0.0;   // For mirror-like surfaces [0, 1].
    double transparency = 0.0;   // For transmissive surfaces [0, 1].
    double ior = 1.5;            // Index of refraction for dielectrics.

    // Procedural checker support (intended for plane material).
    bool use_checker = false;
    Vec3 checker_a = Vec3(0.95, 0.95, 0.95);
    Vec3 checker_b = Vec3(0.10, 0.10, 0.10);
    double checker_scale = 1.0;  // Frequency in plane UV space.
};

inline Vec3 material_base_color(const Material& m, double u, double v) {
    if (!m.use_checker) {
        return m.albedo;
    }

    const int iu = static_cast<int>(std::floor(u * m.checker_scale));
    const int iv = static_cast<int>(std::floor(v * m.checker_scale));
    const bool even = ((iu + iv) & 1) == 0;
    return even ? m.checker_a : m.checker_b;
}