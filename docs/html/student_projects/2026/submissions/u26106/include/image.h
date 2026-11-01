#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "core.h"

namespace pt {

class Image {
public:
    Image(int image_width, int image_height);

    int Width() const { return image_width_; }
    int Height() const { return image_height_; }

    void SetPixel(int pixel_x, int pixel_y, const Color& pixel_color);
    const Color& GetPixel(int pixel_x, int pixel_y) const;

    bool SavePpm(const std::string& file_path) const;

private:
    std::size_t pixel_index(int pixel_x, int pixel_y) const {
        return static_cast<std::size_t>(pixel_y) *
               static_cast<std::size_t>(image_width_) +
               static_cast<std::size_t>(pixel_x);
    }

    int image_width_;
    int image_height_;
    std::vector<Color> pixel_buffer_;
};

}  // namespace pt
