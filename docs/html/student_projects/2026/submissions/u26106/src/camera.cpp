#include "core.h"

#include <cmath>

namespace pt {

Camera::Camera(const Vec3& position, const Vec3& look_at, const Vec3& up_direction,
               double vertical_fov_degrees, int image_width, int image_height)
    : camera_origin_(position),
      viewport_origin_(),
      pixel_delta_x_(),
      pixel_delta_y_() {
    const double pi = 3.14159265358979323846;
    const double vertical_fov_radians = vertical_fov_degrees * pi / 180.0;
    const double half_viewport_height = std::tan(vertical_fov_radians * 0.5);
    const double aspect_ratio = static_cast<double>(image_width) /
                                static_cast<double>(image_height);

    const Vec3 camera_forward = normalize(look_at - position);
    const Vec3 camera_right = normalize(cross_product(camera_forward, up_direction));
    const Vec3 camera_up = cross_product(camera_right, camera_forward);

    const Vec3 viewport_horizontal = camera_right * (2.0 * half_viewport_height * aspect_ratio);
    const Vec3 viewport_vertical = camera_up * (2.0 * half_viewport_height);

    pixel_delta_x_ = viewport_horizontal / static_cast<double>(image_width);
    pixel_delta_y_ = (-1.0 * viewport_vertical) / static_cast<double>(image_height);

    viewport_origin_ = camera_origin_ + camera_forward - 0.5 * viewport_horizontal +
                       0.5 * viewport_vertical;
}

Ray Camera::GenerateRay(int pixel_x, int pixel_y, double sample_u,
                        double sample_v) const {
    const Vec3 sample_position =
        viewport_origin_ + (static_cast<double>(pixel_x) + sample_u) * pixel_delta_x_ +
        (static_cast<double>(pixel_y) + sample_v) * pixel_delta_y_;

    const Vec3 ray_direction = normalize(sample_position - camera_origin_);
    return Ray(camera_origin_, ray_direction);
}

}  // namespace pt
