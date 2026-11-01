#pragma once

#include "math/vec3.h"

enum class LightType {
    Point,
    Directional
};

struct Light {
    LightType type = LightType::Point;
    Vec3 color = Vec3(1.0, 1.0, 1.0);
    double intensity = 1.0;

    // For point light:
    Vec3 position = Vec3(0.0, 5.0, 0.0);

    // For directional light: direction of light travel (from light toward scene).
    Vec3 direction = Vec3(0.0, -1.0, 0.0);
};