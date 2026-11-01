#pragma once

#include "math/vec3.h"

// Forward declaration (defined in shading/material.h in Phase 4).
struct Material;

// Stores the result of a ray-object intersection.
struct HitRecord {
    double t = 0.0;                 // Ray parameter at hit.
    Vec3 point;                     // World-space hit point.
    Vec3 normal;                    // Geometric normal (unit length).
    const Material* material = nullptr;

    // Optional surface coordinates for procedural textures/checkerboards.
    double u = 0.0;
    double v = 0.0;

    // True when hit is from outside; false when ray is inside object.
    bool front_face = true;

    // Ensures normal always opposes incoming ray direction.
    void set_face_normal(const Vec3& ray_dir, const Vec3& outward_normal) {
        front_face = dot(ray_dir, outward_normal) < 0.0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};