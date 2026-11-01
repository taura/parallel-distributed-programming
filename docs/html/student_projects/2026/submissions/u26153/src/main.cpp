#include "bitmap.h"
#include "material.h"
#include "ray.h"
#include "scene.h"
#include "shape.h"

#include <iostream>
#include <memory>

int main() {
    constexpr int nx = 2000;
    constexpr int ny = 1000;
    constexpr int samples = 16;
    raytracer::Image image(nx, ny);
    raytracer::Scene scene;
    scene.addObject(std::make_unique<raytracer::Terrain>(
        -0.5f,
        std::make_shared<raytracer::Lambertian>(raytracer::vec3(0.65f))));
    scene.addObject(std::make_unique<raytracer::Sphere>(
        raytracer::vec3(0.0f, 0.0f, -1.0f), 0.5f,
        std::make_shared<raytracer::Metal>(raytracer::vec3(1.0f, 0.0f, 0.0f), 0.0f)));

    for (int j = 0; j < ny; ++j) {
        std::cerr << "Rendering (y = " << j << ") "
                  << (100.0 * j / (ny - 1)) << "%\n";
        for (int i = 0; i < nx; ++i) {
            raytracer::vec3 c(0.0f);
            for (int s = 0; s < samples; ++s) {
                float u = float(i + randomFloat()) / float(nx);
                float v = float(j + randomFloat()) / float(ny);
                raytracer::Ray r = scene.getRay(u, v);
                c += scene.render(r);
            }
            c /= float(samples);
            image.write(i, j, c.x, c.y, c.z);
        }
    }

    if (!raytracer::writeBmp("render.bmp", image)) {
        std::cerr << "Failed to write render.bmp\n";
        return 1;
    }
    return 0;
}
