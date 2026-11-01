#include "renderer/renderer.h"

#include <algorithm>
#include <cstdint>

#include "math/math_utils.h"
#include "shading/material.h"
#include "shading/optics.h"

Renderer::Renderer(
    const Hittable& world,
    const std::vector<Light>& lights,
    const LightingConfig& lighting_cfg,
    const RenderParams& render_params
)
    : world_(world),
      lights_(lights),
      lighting_cfg_(lighting_cfg),
      render_params_(render_params) {}

Vec3 Renderer::trace_ray(const Ray& ray, int depth) const {
    if (depth <= 0) {
        return render_params_.depth_exhausted_color;
    }

    HitRecord hit;
    if (!world_.intersect(ray, render_params_.ray_epsilon, 1e30, hit)) {
        return render_params_.background_color;
    }

    const Material& mat = *hit.material;
    const Vec3 view_dir = normalize(-ray.direction);
    const Vec3 local = evaluate_local_lighting(hit, view_dir, lights_, world_, lighting_cfg_);

    if (mat.type == MaterialType::Diffuse) {
        return local;
    }

    if (mat.type == MaterialType::Mirror) {
        const Vec3 reflected_dir = normalize(reflect(ray.direction, hit.normal));
        const Ray reflected_ray(hit.point + hit.normal * render_params_.ray_epsilon, reflected_dir);
        const Vec3 reflected_color = trace_ray(reflected_ray, depth - 1);
        return (1.0 - mat.reflectivity) * local + mat.reflectivity * reflected_color;
    }

    // Dielectric: Fresnel blend of reflection/refraction.
    double eta_i = 1.0;
    double eta_t = mat.ior;
    if (!hit.front_face) {
        eta_i = mat.ior;
        eta_t = 1.0;
    }

    const double cos_theta = std::min(dot(-normalize(ray.direction), hit.normal), 1.0);
    const Vec3 reflected_dir = normalize(reflect(ray.direction, hit.normal));
    const Ray reflected_ray(hit.point + hit.normal * render_params_.ray_epsilon, reflected_dir);
    const Vec3 reflected_color = trace_ray(reflected_ray, depth - 1);

    Vec3 refracted_color(0.0, 0.0, 0.0);
    Vec3 refracted_dir;
    const bool has_refract = refract(normalize(ray.direction), hit.normal, eta_i, eta_t, refracted_dir);

    double fresnel = 1.0;
    if (has_refract) {
        fresnel = fresnel_schlick(cos_theta, eta_i, eta_t);
        const Vec3 refr_dir_n = normalize(refracted_dir);
        const Ray refracted_ray(hit.point + refr_dir_n * render_params_.ray_epsilon, refr_dir_n);
        refracted_color = trace_ray(refracted_ray, depth - 1);
    }

    const Vec3 glass = fresnel * reflected_color + (1.0 - fresnel) * refracted_color;
    return (1.0 - mat.transparency) * local + mat.transparency * glass;
}

void Renderer::render_tile(
    Framebuffer& fb,
    const Camera& cam,
    int start_x,
    int start_y,
    int end_x,
    int end_y
) const {
    const int w = fb.width();
    const int h = fb.height();

    for (int y = start_y; y < end_y; ++y) {
        for (int x = start_x; x < end_x; ++x) {
            Vec3 accum(0.0, 0.0, 0.0);

            // Per-pixel deterministic RNG state (parallel-safe by design).
            uint64_t rng = 0x9e3779b97f4a7c15ULL
                         ^ (static_cast<uint64_t>(x) * 0xbf58476d1ce4e5b9ULL)
                         ^ (static_cast<uint64_t>(y) * 0x94d049bb133111ebULL);

            for (int s = 0; s < render_params_.samples_per_pixel; ++s) {
                const double jx = math::rand01(rng);
                const double jy = math::rand01(rng);

                const double u = (static_cast<double>(x) + jx) / static_cast<double>(w);
                const double v = (static_cast<double>(h - 1 - y) + jy) / static_cast<double>(h);

                const Ray primary = cam.generate_ray(u, v);
                accum += trace_ray(primary, render_params_.max_depth);
            }

            accum /= static_cast<double>(render_params_.samples_per_pixel);
            fb.set_pixel(x, y, accum);
        }
    }
}

void Renderer::render_frame(Framebuffer& fb, const Camera& cam) const {
    render_tile(fb, cam, 0, 0, fb.width(), fb.height());
}