// vec3.h : 3D vector used for points, directions and RGB colors
#pragma once
#include <algorithm>
#include <cmath>

struct Vec3 {
  double x, y, z;
  Vec3() : x(0), y(0), z(0) {}
  Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
  double operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
  Vec3& operator+=(const Vec3& b) { x += b.x; y += b.y; z += b.z; return *this; }
  Vec3& operator*=(const Vec3& b) { x *= b.x; y *= b.y; z *= b.z; return *this; }
  Vec3& operator*=(double s) { x *= s; y *= s; z *= s; return *this; }
};

inline Vec3 operator+(const Vec3& a, const Vec3& b) { return Vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline Vec3 operator-(const Vec3& a, const Vec3& b) { return Vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline Vec3 operator*(const Vec3& a, const Vec3& b) { return Vec3(a.x * b.x, a.y * b.y, a.z * b.z); }  // component-wise (color)
inline Vec3 operator*(const Vec3& a, double s) { return Vec3(a.x * s, a.y * s, a.z * s); }
inline Vec3 operator*(double s, const Vec3& a) { return a * s; }
inline Vec3 operator/(const Vec3& a, double s) { return a * (1.0 / s); }
inline Vec3 operator-(const Vec3& a) { return Vec3(-a.x, -a.y, -a.z); }

inline double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
  return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
inline double length(const Vec3& a) { return std::sqrt(dot(a, a)); }
inline Vec3 normalize(const Vec3& a) { return a / length(a); }
inline Vec3 vmin(const Vec3& a, const Vec3& b) { return Vec3(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)); }
inline Vec3 vmax(const Vec3& a, const Vec3& b) { return Vec3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)); }
inline double max_component(const Vec3& a) { return std::max(a.x, std::max(a.y, a.z)); }
