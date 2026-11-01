#ifndef RAYTRACER_SHAPE_H
#define RAYTRACER_SHAPE_H

#include <cmath>
#include <memory>
#include "ray.h"

namespace raytracer {

class Material;
using MaterialPtr = std::shared_ptr<Material>;

class HitRec {
    public:
        float t;
        vec3 p;
        vec3 n;
        MaterialPtr mat;
};

class Shape {
public:
    virtual ~Shape() = default;
    virtual bool hit(const Ray& r, float t0, float t1, HitRec& hrec) const = 0;
};

class Sphere : public Shape {
public:
    Sphere() {}
    Sphere(const vec3& c, float r, const MaterialPtr& mat)
        : m_center(c)
        , m_radius(r)
        , m_material(mat) {
    }

    virtual bool hit(const Ray& r, float t0, float t1, HitRec& hrec) const override {
        vec3 oc = r.origin() - m_center;
        float a = dot(r.direction(), r.direction());
        float b = 2.0f*dot(oc, r.direction());
        float c = dot(oc, oc) - pow2(m_radius);
        float D = b*b - 4*a*c;
        if (D > 0) {
            float root = sqrtf(D);
            float temp = (-b - root) / (2.0f*a);
            if (temp < t1 && temp > t0) {
                hrec.t = temp;
                hrec.p = r.at(hrec.t);
                hrec.n = (hrec.p - m_center) / m_radius;
                hrec.mat = m_material;
                return true;
            }
            temp = (-b + root) / (2.0f*a);
            if (temp < t1 && temp > t0) {
                hrec.t = temp;
                hrec.p = r.at(hrec.t);
                hrec.n = (hrec.p - m_center) / m_radius;
                hrec.mat = m_material;
                return true;
            }
        }

        return false;
    }

private:
    vec3 m_center;
    float m_radius;
    MaterialPtr m_material;
};

class Terrain : public Shape {
public:
    Terrain(float height, const MaterialPtr& mat)
        : m_height(height), m_material(mat) {}

    bool hit(const Ray& r, float t0, float t1, HitRec& hrec) const override {
        const float dy = r.direction().y;
        if (std::fabs(dy) < EPSILON) return false;

        const float t = (m_height - r.origin().y) / dy;
        if (t <= t0 || t >= t1) return false;

        hrec.t = t;
        hrec.p = r.at(t);
        hrec.n = vec3(0.f, dy < 0.f ? 1.f : -1.f, 0.f);
        hrec.mat = m_material;
        return true;
    }

private:
    float m_height;
    MaterialPtr m_material;
};

}  // namespace raytracer

#endif  // RAYTRACER_SHAPE_H
