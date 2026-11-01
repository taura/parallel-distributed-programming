#ifndef RAYTRACER_BITMAP_H
#define RAYTRACER_BITMAP_H

#include <cstdint>
#include <fstream>
#include <ostream>
#include <vector>
#include "ray.h"

namespace raytracer {

class Image {
    public:
        struct rgb {
            unsigned char r, g, b;
        };
        Image(int width, int height)
            : m_width(width), m_height(height), m_pixels(width * height) {}
        int width() const { return m_width; }
        int height() const { return m_height; }
        const rgb* pixels() const { return m_pixels.data(); }

        void write(int x, int y, float r, float g, float b) {
            int index = y * m_width + x;
            m_pixels[index].r = static_cast<unsigned char>(255.99f * saturate(r));
            m_pixels[index].g = static_cast<unsigned char>(255.99f * saturate(g));
            m_pixels[index].b = static_cast<unsigned char>(255.99f * saturate(b));
        }
    private:
        int m_width;
        int m_height;
        std::vector<rgb> m_pixels;
};

namespace detail {
inline void writeLittleEndian(std::ostream& out, std::uint32_t value, int bytes) {
    for (int i = 0; i < bytes; ++i) {
        out.put(static_cast<char>(value & 0xffu));
        value >>= 8;
    }
}
}  // namespace detail

inline bool writeBmp(const char* filename, const Image& image) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        return false;
    }

    const std::uint32_t width = static_cast<std::uint32_t>(image.width());
    const std::uint32_t height = static_cast<std::uint32_t>(image.height());
    const std::uint32_t rowBytes = (width * 3u + 3u) & ~3u;
    const std::uint32_t imageBytes = rowBytes * height;

    out.put('B');
    out.put('M');
    detail::writeLittleEndian(out, 54u + imageBytes, 4);
    detail::writeLittleEndian(out, 0, 4);
    detail::writeLittleEndian(out, 54, 4);
    detail::writeLittleEndian(out, 40, 4);
    detail::writeLittleEndian(out, width, 4);
    detail::writeLittleEndian(out, height, 4);
    detail::writeLittleEndian(out, 1, 2);
    detail::writeLittleEndian(out, 24, 2);
    detail::writeLittleEndian(out, 0, 4);
    detail::writeLittleEndian(out, imageBytes, 4);
    detail::writeLittleEndian(out, 0, 4);
    detail::writeLittleEndian(out, 0, 4);
    detail::writeLittleEndian(out, 0, 4);
    detail::writeLittleEndian(out, 0, 4);

    const char padding[3] = {};
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const auto& pixel = image.pixels()[y * width + x];
            out.put(static_cast<char>(pixel.b));
            out.put(static_cast<char>(pixel.g));
            out.put(static_cast<char>(pixel.r));
        }
        out.write(padding, rowBytes - width * 3u);
    }

    out.close();
    return static_cast<bool>(out);
}

}  // namespace raytracer

#endif  // RAYTRACER_BITMAP_H
