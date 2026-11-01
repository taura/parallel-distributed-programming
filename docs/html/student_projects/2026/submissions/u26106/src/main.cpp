#include <chrono>
#include <iostream>
#include <vector>

#include "core.h"
#include "image.h"
#include "renderer.h"
#include "scene.h"

int main() {
    pt::RenderConfig config;

    const pt::Scene scene = pt::Scene::CreateCornellBox();
    const pt::Camera camera(pt::Vec3(278.0, 278.0, -800.0),
                            pt::Vec3(278.0, 278.0, 0.0),
                            pt::Vec3(0.0, 1.0, 0.0),
                            40.0,
                            config.image_width,
                            config.image_height);

    const pt::PathTracer path_tracer(config);
    const pt::Renderer renderer(config, path_tracer);

    pt::Image image(config.image_width, config.image_height);

    const auto render_start_time = std::chrono::steady_clock::now();
    renderer.Render(scene, camera, image);
    const auto render_end_time = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed_seconds =
        render_end_time - render_start_time;

    if (!image.SavePpm(config.output_file_name)) {
        std::cerr << "Failed to save image: " << config.output_file_name << "\n";
        return 1;
    }

    std::cout << "Render time (s): " << elapsed_seconds.count() << "\n";
    std::cout << "Saved image: " << config.output_file_name << "\n";
    return 0;
}
