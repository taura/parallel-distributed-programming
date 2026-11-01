#pragma once

#include <cmath>

#include "math/math_utils.h"
#include "math/ray.h"
#include "math/vec3.h"

// Pinhole camera with configurable eye/look-at/up/FOV.
class Camera {
public:
    Camera(
        const Vec3& eye,
        const Vec3& target,
        const Vec3& up,
        double vertical_fov_degrees,
        double aspect_ratio
    ) : origin_(eye) {
        const double theta = math::degrees_to_radians(vertical_fov_degrees);
        const double half_h = std::tan(theta * 0.5);
        const double viewport_h = 2.0 * half_h;
        const double viewport_w = aspect_ratio * viewport_h;

        // Camera basis: w points backward, u right, v up.
        w_ = normalize(eye - target);
        u_ = normalize(cross(up, w_));
        v_ = cross(w_, u_);

        horizontal_ = viewport_w * u_;
        vertical_ = viewport_h * v_;
        lower_left_corner_ = origin_ - 0.5 * horizontal_ - 0.5 * vertical_ - w_;
    }

    Ray generate_ray(double s, double t) const {
        const Vec3 dir = normalize(lower_left_corner_ + s * horizontal_ + t * vertical_ - origin_);
        return Ray(origin_, dir);
    }

private:
    Vec3 origin_;
    Vec3 u_;
    Vec3 v_;
    Vec3 w_;
    Vec3 horizontal_;
    Vec3 vertical_;
    Vec3 lower_left_corner_;
};