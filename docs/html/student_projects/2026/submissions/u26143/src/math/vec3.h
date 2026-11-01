#pragma once

#include <cmath>
#include <ostream>

// Minimal 3D vector for geometry, color, and directions.
struct Vec3 {
    double x;
    double y;
    double z;

    constexpr Vec3() : x(0.0), y(0.0), z(0.0) {}
    constexpr Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    // Unary minus: flips direction.
    constexpr Vec3 operator-() const { return Vec3(-x, -y, -z); }

    // Component-wise arithmetic.
    constexpr Vec3 operator+(const Vec3& rhs) const {
        return Vec3(x + rhs.x, y + rhs.y, z + rhs.z);
    }

    constexpr Vec3 operator-(const Vec3& rhs) const {
        return Vec3(x - rhs.x, y - rhs.y, z - rhs.z);
    }

    // Scalar scale/divide.
    constexpr Vec3 operator*(double s) const { return Vec3(x * s, y * s, z * s); }

    constexpr Vec3 operator/(double s) const { return Vec3(x / s, y / s, z / s); }

    Vec3& operator+=(const Vec3& rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }

    Vec3& operator-=(const Vec3& rhs) {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        return *this;
    }

    Vec3& operator*=(double s) {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    Vec3& operator/=(double s) {
        x /= s;
        y /= s;
        z /= s;
        return *this;
    }

    // Squared length avoids sqrt when only relative comparison is needed.
    constexpr double length_squared() const { return x * x + y * y + z * z; }

    double length() const { return std::sqrt(length_squared()); }
};

// Scalar * vector
inline constexpr Vec3 operator*(double s, const Vec3& v) { return v * s; }

// Dot product: similarity/projection measure.
inline constexpr double dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Cross product: perpendicular vector (right-hand rule).
inline constexpr Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

// Unit vector in same direction. Caller should avoid zero-length input.
inline Vec3 normalize(const Vec3& v) {
    return v / v.length();
}

// Component-wise multiplication (useful for color modulation).
inline constexpr Vec3 hadamard(const Vec3& a, const Vec3& b) {
    return Vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline std::ostream& operator<<(std::ostream& os, const Vec3& v) {
    os << v.x << " " << v.y << " " << v.z;
    return os;
}