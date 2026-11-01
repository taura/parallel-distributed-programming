#ifndef RAYTRACER_SCENE_H
#define RAYTRACER_SCENE_H

#include <limits>
#include <memory>
#include <utility>
#include <vector>
#include "material.h"

namespace raytracer {

class Scene {
    public:
        Scene() { build(); }

        void build() {
            const vec3 u(4.0f, 0.0f, 0.0f);
            const vec3 v(0.0f, 2.0f, 0.0f);
            const vec3 w(-2.0f, -1.0f, -1.0f);
            m_camera = Camera(u, v, w);
        }

        void addObject(std::unique_ptr<Shape> object) {
            if (object) {
                m_objects.push_back(std::move(object));
            }
        }

        Ray getRay(float u, float v) const { return m_camera.getRay(u, v); }

        vec3 render(const Ray& ray) const {
            return render(ray, 0);
        }

    private:
        vec3 render(const Ray& ray, int depth) const {
            if (depth >= 50) return vec3(0.f);

            HitRec nearest;
            bool hitAnything = false;
            float nearestT = std::numeric_limits<float>::infinity();
            for (const auto& object : m_objects) {
                HitRec candidate;
                // 交差判定を行い、最も近い交点を見つける
                if (object->hit(ray, EPSILON, nearestT, candidate)) {
                    hitAnything = true;
                    nearestT = candidate.t;
                    nearest = candidate;
                }
            }
            if (hitAnything) {
                ScatterRec scattered;
                if (nearest.mat && nearest.mat->scatter(ray, nearest, scattered)) {
                    // 再帰的に反射光線を追跡して色を計算
                    return mulPerElem(scattered.albedo, render(scattered.ray, depth + 1));
                }
                return vec3(0.f);
            }
            return backgroundColor(ray.direction());
        }

        vec3 backgroundColor(const vec3& rayDirection) const {
            const vec3 direction = normalize(rayDirection);
            const float t = 0.5f * (direction.y + 1.0f); // y成分を0から1の範囲に正規化
            const vec3 horizonColor(1.0f); // 白色
            const vec3 zenithColor(0.5f, 0.7f, 1.0f); // 水色
            return lerp(t, horizonColor, zenithColor); // 線形に補間して背景色を計算
        }

        std::vector<std::unique_ptr<Shape>> m_objects;
        Camera m_camera;
};

}  // namespace raytracer

#endif  // RAYTRACER_SCENE_H
