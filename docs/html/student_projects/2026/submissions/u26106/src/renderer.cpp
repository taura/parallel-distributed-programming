#include "renderer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "image.h"
#include "scene.h"

namespace pt {

namespace {

constexpr double kPi = 3.14159265358979323846;

const Quad* find_ceiling_light_quad(const Scene& scene) {
    for (const Quad& quad : scene.Quads()) {
        const Material& material = scene.Materials()[quad.material_id];
        if (material.IsEmitter()) {
            return &quad;
        }
    }
    return nullptr;
}

Color estimate_direct_lighting(const HitRecord& hit_record, const Material& material,
                               const Scene& scene, std::mt19937& random_engine) {
    const Quad* light_quad = find_ceiling_light_quad(scene);
    if (light_quad == nullptr) {
        return Color(0.0, 0.0, 0.0);
    }

    const Material& light_material = scene.Materials()[light_quad->material_id];
    const Vec3 light_normal_unnormalized =
        cross_product(light_quad->edge_u, light_quad->edge_v);
    const double light_area = length(light_normal_unnormalized);
    if (light_area <= 0.0) {
        return Color(0.0, 0.0, 0.0);
    }

    std::uniform_real_distribution<double> uniform_distribution(0.0, 1.0);
    const double u = uniform_distribution(random_engine);
    const double v = uniform_distribution(random_engine);
    const Vec3 sampled_light_position =
        light_quad->origin + light_quad->edge_u * u + light_quad->edge_v * v;

    const Vec3 to_light = sampled_light_position - hit_record.hit_position;
    const double distance_squared = length_squared(to_light);
    if (distance_squared <= 0.0) {
        return Color(0.0, 0.0, 0.0);
    }
    const double distance = std::sqrt(distance_squared);
    const Vec3 light_direction = to_light / distance;

    const double cosine_surface =
        std::max(0.0, dot_product(hit_record.surface_normal, light_direction));
    if (cosine_surface <= 0.0) {
        return Color(0.0, 0.0, 0.0);
    }

    const Vec3 light_normal = normalize(light_normal_unnormalized);
    const double cosine_light =
        std::max(0.0, dot_product(light_normal, -1.0 * light_direction));
    if (cosine_light <= 0.0) {
        return Color(0.0, 0.0, 0.0);
    }

    const double shadow_epsilon = 1e-4;
    const Ray shadow_ray(hit_record.hit_position + hit_record.surface_normal * shadow_epsilon,
                         light_direction);
    HitRecord shadow_hit_record;
    if (scene.Intersect(shadow_ray, shadow_epsilon, distance - shadow_epsilon,
                        shadow_hit_record)) {
        return Color(0.0, 0.0, 0.0);
    }

    const double area_pdf = 1.0 / light_area;
    const Color lambertian_bsdf = material.albedo / kPi;
    const double geometry_term = (cosine_surface * cosine_light) / distance_squared;

    return hadamard_product(lambertian_bsdf, light_material.emission) * (geometry_term / area_pdf);
}

void build_local_frame(const Vec3& surface_normal, Vec3& tangent,
                       Vec3& bitangent) {
    if (std::abs(surface_normal.x) > 0.1) {
        tangent = normalize(cross_product(Vec3(0.0, 1.0, 0.0), surface_normal));
    } else {
        tangent = normalize(cross_product(Vec3(1.0, 0.0, 0.0), surface_normal));
    }
    bitangent = cross_product(surface_normal, tangent);
}

}  // namespace

PathTracer::PathTracer(const RenderConfig& config) : config_(config) {}

Color PathTracer::TracePath(const Ray& ray, const Scene& scene,
                            std::mt19937& random_engine) const {
    return path_tracing_recursive(ray, scene, random_engine, 0);
}

Color PathTracer::path_tracing_recursive(const Ray& ray, const Scene& scene,
                                         std::mt19937& random_engine,
                                         int current_depth) const {
    HitRecord hit_record;
    if (!scene.Intersect(ray, 1e-4, 1e30, hit_record)) {
        return Color(0.0, 0.0, 0.0);
    }

    const Material& material = scene.Materials()[hit_record.material_id];
    if (material.IsEmitter()) {
        if (current_depth == 0) {
            return material.emission;
        }
        return Color(0.0, 0.0, 0.0);
    }

    const Color direct_radiance =
        estimate_direct_lighting(hit_record, material, scene, random_engine);

    const double albedo_max =
        std::max(material.albedo.x, std::max(material.albedo.y, material.albedo.z));
    if (albedo_max <= 0.0) {
        return direct_radiance;
    }

    std::uniform_real_distribution<double> uniform_distribution(0.0, 1.0);

    if (current_depth >= config_.russian_roulette_start_depth) {
        const double continue_probability = clamp_value(
            config_.russian_roulette_continue_probability, 1e-6, 1.0);
        if (uniform_distribution(random_engine) > continue_probability) {
            return direct_radiance;
        }
    }

    const Vec3 sampled_direction =
        sample_hemisphere_direction(hit_record.surface_normal, random_engine);

    const Ray bounce_ray(hit_record.hit_position, sampled_direction);
    const Color incoming_radiance =
        path_tracing_recursive(bounce_ray, scene, random_engine, current_depth + 1);

    const double cosine_term =
        std::max(0.0, dot_product(hit_record.surface_normal, sampled_direction));
    const double sampling_pdf = cosine_term / kPi;
    const Color lambertian_bsdf = material.albedo / kPi;

    Color indirect_radiance =
        hadamard_product(lambertian_bsdf, incoming_radiance) *
        (cosine_term / sampling_pdf);

    if (current_depth >= config_.russian_roulette_start_depth) {
        const double continue_probability = clamp_value(
            config_.russian_roulette_continue_probability, 1e-6, 1.0);
        indirect_radiance =
            indirect_radiance / continue_probability;
    }

    return direct_radiance + indirect_radiance;
}

Vec3 PathTracer::sample_hemisphere_direction(
    const Vec3& surface_normal, std::mt19937& random_engine) const {
    std::uniform_real_distribution<double> uniform_distribution(0.0, 1.0);

    const double u1 = uniform_distribution(random_engine);
    const double u2 = uniform_distribution(random_engine);

    const double radius = std::sqrt(u1);
    const double theta = 2.0 * kPi * u2;

    const double x = radius * std::cos(theta);
    const double y = radius * std::sin(theta);
    const double z = std::sqrt(1.0 - u1);

    Vec3 tangent;
    Vec3 bitangent;
    build_local_frame(surface_normal, tangent, bitangent);

    return normalize(tangent * x + bitangent * y + surface_normal * z);
}

Renderer::Renderer(const RenderConfig& config, PathTracer path_tracer)
    : config_(config), path_tracer_(path_tracer) {}

void Renderer::Render(const Scene& scene, const Camera& camera, Image& image) const {
    std::mt19937 random_engine(12345u);
    std::uniform_real_distribution<double> uniform_distribution(0.0, 1.0);
    int strata_x = static_cast<int>(std::sqrt(static_cast<double>(config_.sample_count)));
    if (strata_x < 1) {
        strata_x = 1;
    }
    while (config_.sample_count % strata_x != 0) {
        --strata_x;
    }
    const int strata_y = config_.sample_count / strata_x;

    const int total_pixels = config_.image_width * config_.image_height;
    int completed_pixels = 0;
    const int progress_bar_width = 80;
    const auto render_start_time = std::chrono::steady_clock::now();

    for (int pixel_y = 0; pixel_y < config_.image_height; ++pixel_y) {
        for (int pixel_x = 0; pixel_x < config_.image_width; ++pixel_x) {
            Color accumulated_color(0.0, 0.0, 0.0);

            for (int sample_index = 0; sample_index < config_.sample_count;
                 ++sample_index) {
                const int stratum_x = sample_index % strata_x;
                const int stratum_y = sample_index / strata_x;
                const double sample_u =
                    (static_cast<double>(stratum_x) + uniform_distribution(random_engine)) / static_cast<double>(strata_x);
                const double sample_v =
                    (static_cast<double>(stratum_y) + uniform_distribution(random_engine)) / static_cast<double>(strata_y);

                const Ray camera_ray =
                    camera.GenerateRay(pixel_x, pixel_y, sample_u, sample_v);
                const Color sample_color =
                    path_tracer_.TracePath(camera_ray, scene, random_engine);

                accumulated_color = accumulated_color + sample_color;
            }

            image.SetPixel(pixel_x, pixel_y,
                           accumulated_color / static_cast<double>(config_.sample_count));

            ++completed_pixels;
        }

        const double progress = static_cast<double>(completed_pixels) /
                                static_cast<double>(total_pixels);
        const int percent = static_cast<int>(progress * 100.0);
        const int filled_width = static_cast<int>(progress * progress_bar_width);
        const auto now = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed_seconds = now - render_start_time;

        std::cout << "\r[";
        for (int i = 0; i < progress_bar_width; ++i) {
            if (i < filled_width) {
                std::cout << "+";
            } else if (i == filled_width && filled_width < progress_bar_width) {
                std::cout << ">";
            } else {
                std::cout << "=";
            }
        }
        std::cout << std::setw(3) << percent << "%] Time: "
                  << std::fixed << std::setprecision(1)
                  << elapsed_seconds.count() << "s" << std::flush;
    }

    std::cout << "\n";
}

}  // namespace pt
