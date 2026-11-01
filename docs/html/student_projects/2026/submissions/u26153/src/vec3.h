#ifndef RAYTRACER_VEC3_H
#define RAYTRACER_VEC3_H

#include <cmath>

namespace raytracer {

struct Vec3 {
    float x;
    float y;
    float z;

    Vec3() : x(0.0f), y(0.0f), z(0.0f) {}
    Vec3(float value) : x(value), y(value), z(value) {}
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    Vec3& operator+=(const Vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    Vec3& operator-=(const Vec3& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    Vec3& operator*=(float scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    Vec3& operator/=(float scalar) {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }
};

using vec3 = Vec3;

inline Vec3 operator+(Vec3 a, const Vec3& b) { return a += b; }
inline Vec3 operator-(Vec3 a, const Vec3& b) { return a -= b; }
inline Vec3 operator-(const Vec3& v) { return Vec3(-v.x, -v.y, -v.z); }
inline Vec3 operator*(Vec3 v, float scalar) { return v *= scalar; }
inline Vec3 operator*(float scalar, Vec3 v) { return v *= scalar; }
inline Vec3 operator/(Vec3 v, float scalar) { return v /= scalar; }

inline float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x);
}

inline float lengthSqr(const Vec3& v) { return dot(v, v); }
inline float length(const Vec3& v) { return std::sqrt(lengthSqr(v)); }

inline Vec3 normalize(const Vec3& v) {
    const float len = length(v);
    return len > 0.0f ? v / len : Vec3();
}

inline Vec3 mulPerElem(const Vec3& a, const Vec3& b) {
    return Vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline Vec3 maxPerElem(const Vec3& a, const Vec3& b) {
    return Vec3(a.x > b.x ? a.x : b.x,
                a.y > b.y ? a.y : b.y,
                a.z > b.z ? a.z : b.z);
}

inline Vec3 minPerElem(const Vec3& a, const Vec3& b) {
    return Vec3(a.x < b.x ? a.x : b.x,
                a.y < b.y ? a.y : b.y,
                a.z < b.z ? a.z : b.z);
}

inline Vec3 lerp(float t, const Vec3& a, const Vec3& b) {
    return a + t * (b - a);
}

}  // namespace raytracer

#endif  // RAYTRACER_VEC3_H
