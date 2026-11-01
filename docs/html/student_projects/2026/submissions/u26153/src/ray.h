#ifndef RAYTRACER_RAY_H
#define RAYTRACER_RAY_H

#include <cmath>
#include <cstdlib>
#include "vec3.h"
#define PI 3.14159265359f
#define PI2 6.28318530718f
#define RECIP_PI 0.31830988618f
#define RECIP_PI2 0.15915494f
#define LOG2 1.442695f
#define EPSILON 1e-6f
#define GAMMA_FACTOR 2.2f

inline float randomFloat() {
    return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
}

inline float pow2(float x) { return x*x; }
inline float pow3(float x) { return x*x*x; }
inline float pow4(float x) { return x*x*x*x; }
inline float pow5(float x) { return x*x*x*x*x; }
inline float clamp(float x, float a, float b) { return x < a ? a : x > b ? b : x; }
inline float saturate(float x) { return x < 0.f ? 0.f : x > 1.f ? 1.f : x; }
inline float recip(float x) { return 1.f / x; }
inline float mix(float a, float b, float t) { return a*(1.f - t) + b*t; /* return a + (b-a) * t; */ }
inline float step(float edge, float x) { return (x < edge) ? 0.f : 1.f; }
inline float smoothstep(float a, float b, float t) { if (a >= b) return 0.f; float x = saturate((t - a) / (b - a)); return x*x*(3.f - 2.f * x); }
inline float radians(float deg) { return (deg / 180.f)*PI; }
inline float degrees(float rad) { return (rad / PI) * 180.f; }


namespace raytracer {
class Ray{
    public:
        Ray() {}
        Ray(const vec3& origin, const vec3& direction)
            : orig(origin), dir(direction) {}
        
        const vec3& origin() const { return orig; }
        const vec3& direction() const { return dir; }
        vec3 at(float t) const {
            return orig + t * dir;
        }
    private:
        vec3 orig;
        vec3 dir;
};

class Camera{
    public:
        Camera(){}

        // カメラから見た方向を(絶対座標で)指定するコンストラクタ
        Camera(const vec3& u, const vec3& v, const vec3& w) {
            m_origin = vec3();
            m_uvw[0] = u;
            m_uvw[1] = v;
            m_uvw[2] = w;
        }
        // カメラとスクリーンの位置は1.0fの距離にあると仮定
        // カメラの位置と注視点、上方向ベクトル、垂直視野角、アスペクト比を指定するコンストラクタ
        Camera(const vec3&lookfrom, const vec3& lookat, const vec3& vup, float vfov, float aspect) {
            vec3 u, v, w;
            float halfH = tanf(radians(vfov) / 2.f);
            float halfW = aspect * halfH;
            w = normalize(lookfrom - lookat);
            u = normalize(cross(vup, w));
            v = cross(w, u);
            m_origin = lookfrom;
            m_uvw[0] = u * halfW * 2.f;
            m_uvw[1] = v * halfH * 2.f;
            m_uvw[2] = -halfW * u - halfH * v - w;
        }

        Ray getRay(float u, float v) const {
            return Ray(m_origin, m_uvw[2] + u * m_uvw[0] + v * m_uvw[1]);
        }
    private:
        vec3 m_origin;
        vec3 m_uvw[3];
};

}  // namespace raytracer

#endif  // RAYTRACER_RAY_H
