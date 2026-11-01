#pragma once

#include <memory>
#include <vector>

#include "geometry/hit_record.h"
#include "math/ray.h"

// Common interface for all intersectable scene primitives.
class Hittable {
public:
    virtual ~Hittable() = default;

    // Returns true if ray intersects object in [t_min, t_max], filling rec.
    virtual bool intersect(
        const Ray& ray,
        double t_min,
        double t_max,
        HitRecord& rec
    ) const = 0;
};

// Simple linear container of hittables (serial baseline-friendly).
class HittableList : public Hittable {
public:
    std::vector<std::shared_ptr<Hittable>> objects;

    void add(const std::shared_ptr<Hittable>& obj) {
        objects.push_back(obj);
    }

    bool intersect(
        const Ray& ray,
        double t_min,
        double t_max,
        HitRecord& rec
    ) const override {
        HitRecord temp;
        bool hit_anything = false;
        double closest = t_max;

        for (const auto& obj : objects) {
            if (obj->intersect(ray, t_min, closest, temp)) {
                hit_anything = true;
                closest = temp.t;
                rec = temp;
            }
        }
        return hit_anything;
    }
};