#include "shading/shading.h"

#include <cmath>
#include <limits>

#include "math/math_utils.h"
#include "math/ray.h"

bool is_in_shadow(
    const Vec3& point,
    const Vec3& normal,
    const Light& light,
    const Hittable& world,
    double epsilon
) {
    HitRecord temp;

    if (light.type == LightType::Point) {
        const Vec3 to_light = light.position - point;
        const double dist_to_light = to_light.length();
        const Vec3 light_dir = to_light / dist_to_light;

        const Ray shadow_ray(point + normal * epsilon, light_dir);
        return world.intersect(shadow_ray, epsilon, dist_to_light - epsilon, temp);
    }

    const Vec3 light_dir = normalize(-light.direction);
    const Ray shadow_ray(point + normal * epsilon, light_dir);
    return world.intersect(
        shadow_ray,
        epsilon,
        std::numeric_limits<double>::infinity(),
        temp
    );
}

Vec3 evaluate_local_lighting(
    const HitRecord& hit,
    const Vec3& view_dir,
    const std::vector<Light>& lights,
    const Hittable& world,
    const LightingConfig& cfg
) {
    const Material& mat = *hit.material;
    const Vec3 base_color = material_base_color(mat, hit.u, hit.v);

    Vec3 result = hadamard(base_color, cfg.ambient_color);

    for (const Light& light : lights) {
        if (is_in_shadow(hit.point, hit.normal, light, world, cfg.shadow_epsilon)) {
            continue;
        }

        Vec3 L;
        Vec3 radiance = light.color * light.intensity;

        if (light.type == LightType::Point) {
            const Vec3 to_light = light.position - hit.point;
            const double r2 = to_light.length_squared();
            const double r = std::sqrt(r2);
            L = to_light / r;
            radiance /= r2; // inverse-square attenuation
        } else {
            L = normalize(-light.direction);
        }

        const double n_dot_l = math::clamp(dot(hit.normal, L), 0.0, 1.0);
        if (n_dot_l <= 0.0) {
            continue;
        }

        const Vec3 diffuse = base_color * n_dot_l;

        const Vec3 H = normalize(L + view_dir);
        const double n_dot_h = math::clamp(dot(hit.normal, H), 0.0, 1.0);
        const double spec_power = std::pow(n_dot_h, mat.shininess);
        const Vec3 specular = mat.specular_color * spec_power;

        result += hadamard(diffuse + specular, radiance);
    }

    return result;
}