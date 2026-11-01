#pragma once

#include <vector>

#include "geometry/hit_record.h"
#include "geometry/hittable.h"
#include "math/vec3.h"
#include "shading/light.h"
#include "shading/material.h"

struct LightingConfig {
    Vec3 ambient_color = Vec3(0.03, 0.03, 0.03);
    double shadow_epsilon = 1e-4;
};

// Returns true if light path from hit point to light is blocked.
bool is_in_shadow(
    const Vec3& point,
    const Vec3& normal,
    const Light& light,
    const Hittable& world,
    double epsilon
);

// Local illumination only (ambient + Blinn-Phong + hard shadows).
Vec3 evaluate_local_lighting(
    const HitRecord& hit,
    const Vec3& view_dir, // from hit point toward camera
    const std::vector<Light>& lights,
    const Hittable& world,
    const LightingConfig& cfg
);