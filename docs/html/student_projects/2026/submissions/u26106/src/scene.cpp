#include "scene.h"

#include <cmath>
#include <limits>

namespace pt {

namespace {

Vec3 rotate_y(const Vec3& point, double angle_radians) {
    const double cos_theta = std::cos(angle_radians);
    const double sin_theta = std::sin(angle_radians);
    return Vec3(cos_theta * point.x + sin_theta * point.z, point.y,
                -sin_theta * point.x + cos_theta * point.z);
}

}  // namespace

Scene Scene::CreateCornellBox() {
    Scene scene;

    const int white_material_id = scene.AddMaterial(
        Material(Color(0.73, 0.71, 0.67), Color(0.0, 0.0, 0.0)));
    const int red_material_id = scene.AddMaterial(
        Material(Color(0.65, 0.05, 0.05), Color(0.0, 0.0, 0.0)));
    const int green_material_id = scene.AddMaterial(
        Material(Color(0.12, 0.45, 0.15), Color(0.0, 0.0, 0.0)));
    const int light_material_id = scene.AddMaterial(
        Material(Color(0.0, 0.0, 0.0), Color(18.0, 16.8, 15.0)));

    scene.AddQuad(Quad(Vec3(0.0, 0.0, 0.0), Vec3(555.0, 0.0, 0.0),                                                                        
                       Vec3(0.0, 0.0, 555.0), white_material_id));      // floor
    scene.AddQuad(Quad(Vec3(0.0, 555.0, 0.0), Vec3(555.0, 0.0, 0.0),
                       Vec3(0.0, 0.0, 555.0), white_material_id));      // ceiling
    scene.AddQuad(Quad(Vec3(0.0, 0.0, 555.0), Vec3(555.0, 0.0, 0.0),
                       Vec3(0.0, 555.0, 0.0), white_material_id));      // back
    scene.AddQuad(Quad(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 555.0),
                       Vec3(0.0, 555.0, 0.0), red_material_id));
    scene.AddQuad(Quad(Vec3(555.0, 0.0, 0.0), Vec3(0.0, 555.0, 0.0),
                       Vec3(0.0, 0.0, 555.0), green_material_id));

    scene.AddQuad(Quad(Vec3(213.0, 554.0, 227.0), Vec3(130.0, 0.0, 0.0),
                       Vec3(0.0, 0.0, 105.0), light_material_id));

    scene.add_box(0.0, 0.0, 0.0, 165.0, 165.0, 165.0, -18.0,                                                                
                  Vec3(130.0, 0.0, 65.0), white_material_id);   // short box
    scene.add_box(0.0, 0.0, 0.0, 165.0, 330.0, 165.0, 15.0,
                  Vec3(265.0, 0.0, 295.0), white_material_id);  // tall box

    return scene;
}

int Scene::AddMaterial(const Material& material) {
    materials_.push_back(material);
    return static_cast<int>(materials_.size()) - 1;
}

void Scene::AddQuad(const Quad& quad) {
    quads_.push_back(quad);
}

bool Scene::Intersect(const Ray& ray, double t_min, double t_max,
                      HitRecord& hit_record) const {
    bool did_hit = false;
    double closest_distance = t_max;
    const double epsilon = 1e-9;

    for (const Quad& quad : quads_) {
        const Vec3 normal_unnormalized = cross_product(quad.edge_u, quad.edge_v);
        const double denominator = dot_product(normal_unnormalized, ray.Direction());

        if (std::abs(denominator) < epsilon) {
            continue;
        }

        const double hit_distance =
            dot_product(quad.origin - ray.Origin(), normal_unnormalized) / denominator;

        if (hit_distance < t_min || hit_distance > closest_distance) {
            continue;
        }

        const Vec3 hit_position = ray.At(hit_distance);
        const Vec3 relative_hit = hit_position - quad.origin;

        const double normal_length_squared = length_squared(normal_unnormalized);
        const double alpha =
            dot_product(cross_product(relative_hit, quad.edge_v), normal_unnormalized) /
            normal_length_squared;
        const double beta =
            dot_product(cross_product(quad.edge_u, relative_hit), normal_unnormalized) /
            normal_length_squared;

        if (alpha < 0.0 || alpha > 1.0 || beta < 0.0 || beta > 1.0) {
            continue;
        }

        did_hit = true;
        closest_distance = hit_distance;

        const Vec3 outward_normal = normalize(normal_unnormalized);
        const bool hit_front_face = denominator < 0.0;

        hit_record.hit_distance = hit_distance;
        hit_record.hit_position = hit_position;
        hit_record.surface_normal =
            hit_front_face ? outward_normal : (-1.0 * outward_normal);
        hit_record.material_id = quad.material_id;
        hit_record.hit_front_face = hit_front_face;
    }

    return did_hit;
}

int Scene::add_box(double min_x, double min_y, double min_z, double max_x,
                   double max_y, double max_z, double rotate_y_degrees,
                   const Vec3& translation, int material_id) {
    const double angle_radians = rotate_y_degrees * 3.14159265358979323846 / 180.0;

    Vec3 vertices[8] = {
        Vec3(min_x, min_y, min_z), Vec3(max_x, min_y, min_z),
        Vec3(max_x, min_y, max_z), Vec3(min_x, min_y, max_z),
        Vec3(min_x, max_y, min_z), Vec3(max_x, max_y, min_z),
        Vec3(max_x, max_y, max_z), Vec3(min_x, max_y, max_z)};

    for (Vec3& vertex : vertices) {
        vertex = rotate_y(vertex, angle_radians) + translation;
    }

    AddQuad(Quad(vertices[0], vertices[1] - vertices[0], vertices[3] - vertices[0],
                 material_id));
    AddQuad(Quad(vertices[4], vertices[5] - vertices[4], vertices[7] - vertices[4],
                 material_id));

    AddQuad(Quad(vertices[0], vertices[1] - vertices[0], vertices[4] - vertices[0],
                 material_id));
    AddQuad(Quad(vertices[3], vertices[2] - vertices[3], vertices[7] - vertices[3],
                 material_id));
    AddQuad(Quad(vertices[0], vertices[3] - vertices[0], vertices[4] - vertices[0],
                 material_id));
    AddQuad(Quad(vertices[1], vertices[2] - vertices[1], vertices[5] - vertices[1],
                 material_id));

    return 6;
}

}  // namespace pt
