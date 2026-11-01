#pragma once

#include <vector>

#include "core.h"

namespace pt {

struct Material {
    Color albedo;
    Color emission;

    Material() = default;
    Material(const Color& surface_albedo, const Color& emitted_radiance)
        : albedo(surface_albedo), emission(emitted_radiance) {}

    bool IsEmitter() const {
        return emission.x > 0.0 || emission.y > 0.0 || emission.z > 0.0;
    }
};

struct Quad {
    Vec3 origin;
    Vec3 edge_u;
    Vec3 edge_v;
    int material_id;

    Quad() = default;
    Quad(const Vec3& quad_origin, const Vec3& quad_edge_u,
         const Vec3& quad_edge_v, int quad_material_id)
        : origin(quad_origin),
          edge_u(quad_edge_u),
          edge_v(quad_edge_v),
          material_id(quad_material_id) {}
};

struct HitRecord {
    double hit_distance;
    Vec3 hit_position;
    Vec3 surface_normal;
    int material_id;
    bool hit_front_face;
};

class Scene {
public:
    Scene() = default;

    static Scene CreateCornellBox();

    int AddMaterial(const Material& material);
    void AddQuad(const Quad& quad);
    bool Intersect(const Ray& ray, double t_min, double t_max,
                   HitRecord& hit_record) const;

    const std::vector<Material>& Materials() const { return materials_; }
    const std::vector<Quad>& Quads() const { return quads_; }

private:
    int add_box(double min_x, double min_y, double min_z, double max_x,
                double max_y, double max_z, double rotate_y_degrees,
                const Vec3& translation, int material_id);
    std::vector<Material> materials_;
    std::vector<Quad> quads_;
};

}  // namespace pt
