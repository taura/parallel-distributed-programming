#pragma once

#include <string>
#include <vector>

#include "math/vec3.h"

class Framebuffer {
public:
    Framebuffer(int width, int height);

    int width() const { return width_; }
    int height() const { return height_; }

    void set_pixel(int x, int y, const Vec3& color);
    const Vec3& get_pixel(int x, int y) const;

    bool write_ppm(const std::string& path, double gamma = 2.2) const;

private:
    int width_;
    int height_;
    std::vector<Vec3> pixels_;

    static unsigned char to_srgb8(double linear, double gamma);
};