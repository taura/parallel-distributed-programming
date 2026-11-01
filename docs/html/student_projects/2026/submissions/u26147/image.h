// image.h : output and verification utilities
#pragma once
#include "vec3.h"
#include <cstdint>
#include <string>
#include <vector>

// Tone map (clamp to [0,1] + gamma 2.2) and write a binary PPM (P6)
bool write_ppm(const std::string& path, const std::vector<Vec3>& img, int width, int height);

// Raw linear float image (PFM), used as a reference for error measurement
bool write_pfm(const std::string& path, const std::vector<Vec3>& img, int width, int height);
bool read_pfm(const std::string& path, std::vector<Vec3>& img, int& width, int& height);

// Verification metrics
double rmse(const std::vector<Vec3>& a, const std::vector<Vec3>& b);
Vec3 mean_color(const std::vector<Vec3>& img);
double max_abs_diff(const std::vector<Vec3>& img, const Vec3& c);  // furnace test
uint64_t image_checksum(const std::vector<Vec3>& img);             // serial vs parallel identity
