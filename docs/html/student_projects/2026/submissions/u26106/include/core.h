#pragma once

#include <cmath>
#include <cstddef>

namespace pt {

struct Vec3 {
    double x;
    double y;
    double z;

    constexpr Vec3() : x(0.0), y(0.0), z(0.0) {}
    constexpr Vec3(double x_value, double y_value, double z_value)
        : x(x_value), y(y_value), z(z_value) {}
};

using Color = Vec3;

inline constexpr Vec3 operator+(const Vec3& lhs, const Vec3& rhs) {
    return Vec3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

inline constexpr Vec3 operator-(const Vec3& lhs, const Vec3& rhs) {
    return Vec3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

inline constexpr Vec3 operator*(const Vec3& lhs, double scalar) {
    return Vec3(lhs.x * scalar, lhs.y * scalar, lhs.z * scalar);
}

inline constexpr Vec3 operator*(double scalar, const Vec3& rhs) {
    return rhs * scalar;
}

inline constexpr Vec3 operator/(const Vec3& lhs, double scalar) {
    return Vec3(lhs.x / scalar, lhs.y / scalar, lhs.z / scalar);
}

inline constexpr Vec3 hadamard_product(const Vec3& lhs, const Vec3& rhs) {
    return Vec3(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z);
}

inline constexpr double dot_product(const Vec3& lhs, const Vec3& rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

inline constexpr Vec3 cross_product(const Vec3& lhs, const Vec3& rhs) {
    return Vec3(lhs.y * rhs.z - lhs.z * rhs.y, lhs.z * rhs.x - lhs.x * rhs.z,
                lhs.x * rhs.y - lhs.y * rhs.x);
}

inline constexpr double length_squared(const Vec3& vector) {
    return dot_product(vector, vector);
}

inline double length(const Vec3& vector) {
    return std::sqrt(length_squared(vector));
}

inline Vec3 normalize(const Vec3& vector) {
    return vector / length(vector);
}

inline constexpr double clamp_value(double value, double min_value,
                                    double max_value) {
    return value < min_value ? min_value
                             : (value > max_value ? max_value : value);
}

class Ray {
public:
    Ray() = default;
    Ray(const Vec3& ray_origin, const Vec3& ray_direction)
        : ray_origin_(ray_origin), ray_direction_(ray_direction) {}

    const Vec3& Origin() const { return ray_origin_; }
    const Vec3& Direction() const { return ray_direction_; }
    Vec3 At(double distance) const {
        return ray_origin_ + distance * ray_direction_;
    }

private:
    Vec3 ray_origin_;
    Vec3 ray_direction_;
};

class Camera {
public:
    Camera(const Vec3& position, const Vec3& look_at, const Vec3& up_direction,
           double vertical_fov_degrees, int image_width, int image_height);
    Ray GenerateRay(int pixel_x, int pixel_y, double sample_u,
                    double sample_v) const;

private:
    Vec3 camera_origin_;
    Vec3 viewport_origin_;
    Vec3 pixel_delta_x_;
    Vec3 pixel_delta_y_;
};

struct RenderConfig {
    int image_width = 1080;
    int image_height = 1080;
    int sample_count = 64;
    int russian_roulette_start_depth = 4;
    double russian_roulette_continue_probability = 0.8;
    const char* output_file_name = "cornell_box.ppm";
};

}  // namespace pt
