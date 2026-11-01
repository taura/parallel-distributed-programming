#include "image.h"
#include <cstdio>
#include <cstring>

static unsigned char to_byte(double v) {
  v = v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
  return (unsigned char)(std::pow(v, 1.0 / 2.2) * 255.0 + 0.5);
}

bool write_ppm(const std::string& path, const std::vector<Vec3>& img, int width, int height) {
  FILE* fp = fopen(path.c_str(), "wb");
  if (!fp) return false;
  fprintf(fp, "P6\n%d %d\n255\n", width, height);
  std::vector<unsigned char> buf(3 * img.size());
  for (size_t i = 0; i < img.size(); i++) {
    buf[3 * i] = to_byte(img[i].x);
    buf[3 * i + 1] = to_byte(img[i].y);
    buf[3 * i + 2] = to_byte(img[i].z);
  }
  bool ok = fwrite(buf.data(), 1, buf.size(), fp) == buf.size();
  fclose(fp);
  return ok;
}

// PFM: "PF\n<w> <h>\n-1.0\n" then little-endian floats, bottom row first
bool write_pfm(const std::string& path, const std::vector<Vec3>& img, int width, int height) {
  FILE* fp = fopen(path.c_str(), "wb");
  if (!fp) return false;
  fprintf(fp, "PF\n%d %d\n-1.0\n", width, height);
  std::vector<float> row(3 * width);
  bool ok = true;
  for (int y = height - 1; y >= 0; y--) {
    for (int x = 0; x < width; x++) {
      const Vec3& c = img[(size_t)y * width + x];
      row[3 * x] = (float)c.x;
      row[3 * x + 1] = (float)c.y;
      row[3 * x + 2] = (float)c.z;
    }
    ok &= fwrite(row.data(), sizeof(float), row.size(), fp) == row.size();
  }
  fclose(fp);
  return ok;
}

bool read_pfm(const std::string& path, std::vector<Vec3>& img, int& width, int& height) {
  FILE* fp = fopen(path.c_str(), "rb");
  if (!fp) return false;
  char magic[3] = {0};
  double scale;
  if (fscanf(fp, "%2s %d %d %lf", magic, &width, &height, &scale) != 4 || strcmp(magic, "PF") != 0 || scale >= 0) {
    fclose(fp);
    return false;
  }
  fgetc(fp);  // single whitespace after the header
  img.assign((size_t)width * height, Vec3());
  std::vector<float> row(3 * width);
  for (int y = height - 1; y >= 0; y--) {
    if (fread(row.data(), sizeof(float), row.size(), fp) != row.size()) {
      fclose(fp);
      return false;
    }
    for (int x = 0; x < width; x++) img[(size_t)y * width + x] = Vec3(row[3 * x], row[3 * x + 1], row[3 * x + 2]);
  }
  fclose(fp);
  return true;
}

double rmse(const std::vector<Vec3>& a, const std::vector<Vec3>& b) {
  double s = 0.0;
  for (size_t i = 0; i < a.size(); i++) {
    Vec3 d = a[i] - b[i];
    s += dot(d, d);
  }
  return std::sqrt(s / (3.0 * a.size()));
}

Vec3 mean_color(const std::vector<Vec3>& img) {
  Vec3 s;
  for (const Vec3& c : img) s += c;
  return s / (double)img.size();
}

double max_abs_diff(const std::vector<Vec3>& img, const Vec3& c) {
  double m = 0.0;
  for (const Vec3& v : img) {
    Vec3 d = v - c;
    m = std::max(m, std::max(std::fabs(d.x), std::max(std::fabs(d.y), std::fabs(d.z))));
  }
  return m;
}

// FNV-1a over the raw bytes of the image
uint64_t image_checksum(const std::vector<Vec3>& img) {
  uint64_t h = 0xcbf29ce484222325ull;
  const unsigned char* p = reinterpret_cast<const unsigned char*>(img.data());
  for (size_t i = 0; i < img.size() * sizeof(Vec3); i++) {
    h ^= p[i];
    h *= 0x100000001b3ull;
  }
  return h;
}
