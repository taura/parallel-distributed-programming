#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "config.h"
#include "geometry/hittable.h"
#include "geometry/plane.h"
#include "geometry/sphere.h"
#include "renderer/camera.h"
#include "renderer/framebuffer.h"
#include "renderer/renderer.h"
#include "shading/light.h"
#include "shading/material.h"
#include "shading/shading.h"

int main() {
    // Materials
    Material ground;
    ground.type = MaterialType::Diffuse;
    ground.albedo = Vec3(0.8, 0.8, 0.8);
    ground.specular_color = Vec3(0.1, 0.1, 0.1);
    ground.shininess = 16.0;
    ground.use_checker = true;
    ground.checker_a = Vec3(0.92, 0.92, 0.92);
    ground.checker_b = Vec3(0.12, 0.12, 0.12);
    ground.checker_scale = 2.0;

    Material matte_red;
    matte_red.type = MaterialType::Diffuse;
    matte_red.albedo = Vec3(0.85, 0.2, 0.2);
    matte_red.specular_color = Vec3(0.25, 0.25, 0.25);
    matte_red.shininess = 48.0;

    Material mirror;
    mirror.type = MaterialType::Mirror;
    mirror.albedo = Vec3(1.0, 1.0, 1.0);
    mirror.specular_color = Vec3(1.0, 1.0, 1.0);
    mirror.shininess = 256.0;
    mirror.reflectivity = 0.95;

    Material glass;
    glass.type = MaterialType::Dielectric;
    glass.albedo = Vec3(1.0, 1.0, 1.0);
    glass.specular_color = Vec3(1.0, 1.0, 1.0);
    glass.shininess = 128.0;
    glass.transparency = 1.0;
    glass.ior = 1.5;

    Material matte_blue;
    matte_blue.type = MaterialType::Diffuse;
    matte_blue.albedo = Vec3(0.2, 0.35, 0.9);
    matte_blue.specular_color = Vec3(0.2, 0.2, 0.2);
    matte_blue.shininess = 32.0;

    // Geometry
    HittableList world;
    world.add(std::make_shared<Plane>(
        Vec3(0.0, -1.0, 0.0), Vec3(0.0, 1.0, 0.0), &ground
    ));
    world.add(std::make_shared<Sphere>(
        Vec3(-1.7, 0.0, -4.8), 1.0, &matte_red
    ));
    world.add(std::make_shared<Sphere>(
        Vec3(0.0, 0.0, -5.5), 1.0, &glass
    ));
    world.add(std::make_shared<Sphere>(
        Vec3(1.9, 0.0, -4.3), 1.0, &mirror
    ));
    world.add(std::make_shared<Sphere>(
        Vec3(0.0, -0.45, -2.6), 0.55, &matte_blue
    ));

    // Lights
    std::vector<Light> lights;

    Light key;
    key.type = LightType::Point;
    key.position = Vec3(5.0, 7.0, 5.0);
    key.color = Vec3(1.0, 0.98, 0.95);
    key.intensity = 260.0;
    lights.push_back(key);

    Light fill;
    fill.type = LightType::Directional;
    fill.direction = normalize(Vec3(-1.0, -1.0, -0.25)); // light travel direction
    fill.color = Vec3(0.7, 0.75, 0.9);
    fill.intensity = 0.35;
    lights.push_back(fill);

    // Camera
    const double aspect = static_cast<double>(config::kImageWidth) /
                          static_cast<double>(config::kImageHeight);
    Camera camera(
        Vec3(0.0, 1.4, 5.8),   // eye
        Vec3(0.0, -0.1, -4.8), // look-at
        Vec3(0.0, 1.0, 0.0),   // up
        45.0,                  // vertical FOV (deg)
        aspect
    );

    // Renderer settings
    LightingConfig lighting_cfg;
    lighting_cfg.ambient_color = Vec3(0.03, 0.03, 0.03);
    lighting_cfg.shadow_epsilon = config::kShadowEpsilon;

    RenderParams render_params;
    render_params.samples_per_pixel = config::kSamplesPerPixel;
    render_params.max_depth = config::kMaxDepth;
    render_params.background_color = config::kBackgroundColor;
    render_params.depth_exhausted_color = config::kDepthExhaustedColor;
    render_params.ray_epsilon = config::kRayEpsilon;

    Framebuffer framebuffer(config::kImageWidth, config::kImageHeight);
    Renderer renderer(world, lights, lighting_cfg, render_params);

    const auto t0 = std::chrono::high_resolution_clock::now();
    renderer.render_frame(framebuffer, camera); // serial: one full tile
    const auto t1 = std::chrono::high_resolution_clock::now();

    const bool ok = framebuffer.write_ppm(config::kOutputPath, config::kGamma);
    if (!ok) {
        std::cerr << "Failed to write output image: " << config::kOutputPath << "\n";
        return 1;
    }

    // Simple deterministic quality/check number: average luminance.
    double lum_sum = 0.0;
    for (int y = 0; y < framebuffer.height(); ++y) {
        for (int x = 0; x < framebuffer.width(); ++x) {
            const Vec3 c = framebuffer.get_pixel(x, y);
            lum_sum += 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
        }
    }
    const double avg_luminance =
        lum_sum / static_cast<double>(framebuffer.width() * framebuffer.height());

    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    std::cout << "Render time: " << ms << " ms\n";
    std::cout << std::fixed << std::setprecision(6)
              << "Average luminance (linear): " << avg_luminance << "\n";
    std::cout << "Output: " << config::kOutputPath << "\n";

    return 0;
}