#pragma once

#include "geometry/hittable.h"

class Sphere : public Hittable {
public:
    Sphere(const Vec3& center, double radius, const Material* material);

    bool intersect(
        const Ray& ray,
        double t_min,
        double t_max,
        HitRecord& rec
    ) const override;

private:
    Vec3 center_;
    double radius_;
    const Material* material_;
};