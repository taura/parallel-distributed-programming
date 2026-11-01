#pragma once

#include <random>

#include "core.h"

namespace pt {

class Image;
class Scene;

class PathTracer {
public:
    explicit PathTracer(const RenderConfig& config);
    Color TracePath(const Ray& ray, const Scene& scene,
                    std::mt19937& random_engine) const;

private:
    Color path_tracing_recursive(const Ray& ray, const Scene& scene,
                    std::mt19937& random_engine,
                    int current_depth) const;
    Vec3 sample_hemisphere_direction(const Vec3& surface_normal,
                                     std::mt19937& random_engine) const;

    RenderConfig config_;
};

class Renderer {
public:
    Renderer(const RenderConfig& config, PathTracer path_tracer);
    void Render(const Scene& scene, const Camera& camera, Image& image) const;

private:
    RenderConfig config_;
    PathTracer path_tracer_;
};

}  // namespace pt
