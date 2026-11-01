#include "renderer/framebuffer.h"

#include <cmath>
#include <fstream>

#include "math/math_utils.h"

Framebuffer::Framebuffer(int width, int height)
    : width_(width), height_(height), pixels_(static_cast<size_t>(width) * static_cast<size_t>(height), Vec3()) {}

void Framebuffer::set_pixel(int x, int y, const Vec3& color) {
    const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(width_) + static_cast<size_t>(x);
    pixels_[idx] = color;
}

const Vec3& Framebuffer::get_pixel(int x, int y) const {
    const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(width_) + static_cast<size_t>(x);
    return pixels_[idx];
}

unsigned char Framebuffer::to_srgb8(double linear, double gamma) {
    // Reinhard tone map + gamma correction.
    const double mapped = linear / (1.0 + linear);
    const double clamped = math::clamp(mapped, 0.0, 1.0);
    const double srgb = std::pow(clamped, 1.0 / gamma);
    const int v = static_cast<int>(255.999 * srgb);
    return static_cast<unsigned char>(math::clamp(static_cast<double>(v), 0.0, 255.0));
}

bool Framebuffer::write_ppm(const std::string& path, double gamma) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }

    out << "P6\n" << width_ << " " << height_ << "\n255\n";

    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            const Vec3& c = get_pixel(x, y);
            const unsigned char r = to_srgb8(c.x, gamma);
            const unsigned char g = to_srgb8(c.y, gamma);
            const unsigned char b = to_srgb8(c.z, gamma);
            out.write(reinterpret_cast<const char*>(&r), 1);
            out.write(reinterpret_cast<const char*>(&g), 1);
            out.write(reinterpret_cast<const char*>(&b), 1);
        }
    }

    return true;
}