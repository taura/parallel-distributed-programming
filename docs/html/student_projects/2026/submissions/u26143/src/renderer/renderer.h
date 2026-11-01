#pragma once

#include <vector>

#include "geometry/hittable.h"
#include "math/ray.h"
#include "math/vec3.h"
#include "renderer/camera.h"
#include "renderer/framebuffer.h"
#include "shading/light.h"
#include "shading/shading.h"

struct RenderParams {
    int samples_per_pixel = 1;
    int max_depth = 5;
    Vec3 background_color = Vec3(0.5, 0.7, 1.0);
    Vec3 depth_exhausted_color = Vec3(0.0, 0.0, 0.0);
    double ray_epsilon = 1e-4;
};

class Renderer {
public:
    Renderer(
        const Hittable& world,
        const std::vector<Light>& lights,
        const LightingConfig& lighting_cfg,
        const RenderParams& render_params
    );

    // Parallel-ready API: render an arbitrary subregion.
    void render_tile(
        Framebuffer& fb,
        const Camera& cam,
        int start_x,
        int start_y,
        int end_x,
        int end_y
    ) const;

    // Serial baseline: just calls one full-frame tile.
    void render_frame(Framebuffer& fb, const Camera& cam) const;

private:
    Vec3 trace_ray(const Ray& ray, int depth) const;

    const Hittable& world_;
    const std::vector<Light>& lights_;
    LightingConfig lighting_cfg_;
    RenderParams render_params_;
};