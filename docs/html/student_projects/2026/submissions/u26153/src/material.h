#ifndef RAYTRACER_MATERIAL_H
#define RAYTRACER_MATERIAL_H

#include "shape.h"

namespace raytracer {

class ScatterRec {
    public:
        Ray	ray;
        vec3 albedo;
};

class Material {
    public:
        virtual ~Material() = default;
        // 光線の反射方向と色の拡散率を計算する
        virtual bool scatter(const Ray& r, const HitRec& hrec, ScatterRec& srec) const = 0;
};
inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2.f * dot(v, n)*n;
}

inline vec3 randomInUnitSphere() {
    while (true) {
        const vec3 p(2.f * randomFloat() - 1.f,
                     2.f * randomFloat() - 1.f,
                     2.f * randomFloat() - 1.f);
        if (lengthSqr(p) < 1.f) return p;
    }
}

class Lambertian : public Material {
public:
    explicit Lambertian(const vec3& albedo) : m_albedo(albedo) {}

    bool scatter(const Ray&, const HitRec& hrec, ScatterRec& srec) const override {
        vec3 direction = hrec.n + randomInUnitSphere();
        if (lengthSqr(direction) < EPSILON * EPSILON) direction = hrec.n;
        srec.ray = Ray(hrec.p, direction);
        srec.albedo = m_albedo;
        return true;
    }

private:
    vec3 m_albedo;
};

class Metal : public Material {
public:
    Metal(const vec3& c, float fuzz)
        : m_albedo(c), m_fuzz(clamp(fuzz, 0.f, 1.f)) {
    }

    virtual bool scatter(const Ray& r, const HitRec& hrec, ScatterRec& srec) const override {
        vec3 reflected = reflect(normalize(r.direction()), hrec.n);
        if (m_fuzz > 0.f) reflected += m_fuzz * randomInUnitSphere();
        srec.ray = Ray(hrec.p, reflected);
        srec.albedo = m_albedo;
        return dot(srec.ray.direction(), hrec.n) > 0;
    }

private:
    vec3 m_albedo;
    float m_fuzz;
};

}  // namespace raytracer

#endif  // RAYTRACER_MATERIAL_H
