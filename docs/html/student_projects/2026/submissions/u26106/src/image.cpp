#include "image.h"

#include <cmath>
#include <fstream>

namespace pt {

namespace {

unsigned char to_ppm_channel(double value) {
    constexpr double kGamma = 2.2;
    const double clamped = clamp_value(value, 0.0, 1.0);
    const double gamma_corrected = std::pow(clamped, 1.0 / kGamma);
    return static_cast<unsigned char>(255.999 * gamma_corrected);
}

}  // namespace

Image::Image(int image_width, int image_height)
    : image_width_(image_width),
      image_height_(image_height),
      pixel_buffer_(static_cast<std::size_t>(image_width) *
                    static_cast<std::size_t>(image_height),
                    Color(0.0, 0.0, 0.0)) {}

void Image::SetPixel(int pixel_x, int pixel_y, const Color& pixel_color) {
    pixel_buffer_[pixel_index(pixel_x, pixel_y)] = pixel_color;
}

const Color& Image::GetPixel(int pixel_x, int pixel_y) const {
    return pixel_buffer_[pixel_index(pixel_x, pixel_y)];
}

bool Image::SavePpm(const std::string& file_path) const {
    std::ofstream output_file(file_path, std::ios::binary);
    if (!output_file) {
        return false;
    }

    output_file << "P6\n" << image_width_ << " " << image_height_ << "\n255\n";

    for (int pixel_y = 0; pixel_y < image_height_; ++pixel_y) {
        for (int pixel_x = 0; pixel_x < image_width_; ++pixel_x) {
            const Color& color = GetPixel(pixel_x, pixel_y);

            const unsigned char rgb[3] = {
                to_ppm_channel(color.x),
                to_ppm_channel(color.y),
                to_ppm_channel(color.z),
            };

            output_file.write(reinterpret_cast<const char*>(rgb), 3);
        }
    }

    return static_cast<bool>(output_file);
}

}  // namespace pt
