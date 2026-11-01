#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Config {
  int width = 1024;
  int height = 1024;
  int scans = 1500;
  int beams = 720;
  double max_range = 220.0;
  int obstacles = 140;
  double path_radius = 300.0;
  unsigned int seed = 12345;
  std::string out_ppm = "";
  std::string out_gt_ppm = "";
};

struct Pose {
  double x;
  double y;
  double heading;
};

int index_of(int x, int y, int w) { return y * w + x; }

bool in_bounds(int x, int y, int w, int h) {
  return (0 <= x && x < w && 0 <= y && y < h);
}

double clamp_logodds(double v) {
  constexpr double lo = -4.0;
  constexpr double hi = 4.0;
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

bool parse_int(const std::string &s, int *out) {
  try {
    size_t pos = 0;
    long long v = std::stoll(s, &pos, 10);
    if (pos != s.size()) return false;
    if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) {
      return false;
    }
    *out = static_cast<int>(v);
    return true;
  } catch (...) {
    return false;
  }
}

bool parse_uint(const std::string &s, unsigned int *out) {
  try {
    size_t pos = 0;
    unsigned long long v = std::stoull(s, &pos, 10);
    if (pos != s.size()) return false;
    if (v > std::numeric_limits<unsigned int>::max()) return false;
    *out = static_cast<unsigned int>(v);
    return true;
  } catch (...) {
    return false;
  }
}

bool parse_double(const std::string &s, double *out) {
  try {
    size_t pos = 0;
    double v = std::stod(s, &pos);
    if (pos != s.size()) return false;
    *out = v;
    return true;
  } catch (...) {
    return false;
  }
}

void usage(const char *prog) {
  std::cerr << "Usage: " << prog << " [options]\n"
            << "  --width N          grid width (default 1024)\n"
            << "  --height N         grid height (default 1024)\n"
            << "  --scans N          number of robot poses/scans (default 1500)\n"
            << "  --beams N          beams per scan (default 720)\n"
            << "  --max-range R      max beam range in cells (default 220)\n"
            << "  --obstacles N      number of random rectangles (default 140)\n"
            << "  --path-radius R    circular trajectory radius (default 300)\n"
            << "  --seed N           random seed (default 12345)\n"
            << "  --out-ppm FILE     optional output map image (P6 PPM)\n"
            << "  --out-gt-ppm FILE  optional output ground-truth image (P6 PPM)\n"
            << "  --help             show this message\n";
}

bool parse_args(int argc, char **argv, Config *cfg) {
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto need_value = [&](const std::string &key) -> const char * {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << key << "\n";
        return nullptr;
      }
      return argv[++i];
    };

    if (k == "--help") {
      usage(argv[0]);
      return false;
    } else if (k == "--width") {
      const char *v = need_value(k);
      if (!v || !parse_int(v, &cfg->width) || cfg->width <= 16) return false;
    } else if (k == "--height") {
      const char *v = need_value(k);
      if (!v || !parse_int(v, &cfg->height) || cfg->height <= 16) return false;
    } else if (k == "--scans") {
      const char *v = need_value(k);
      if (!v || !parse_int(v, &cfg->scans) || cfg->scans <= 0) return false;
    } else if (k == "--beams") {
      const char *v = need_value(k);
      if (!v || !parse_int(v, &cfg->beams) || cfg->beams <= 0) return false;
    } else if (k == "--max-range") {
      const char *v = need_value(k);
      if (!v || !parse_double(v, &cfg->max_range) || cfg->max_range <= 1.0) return false;
    } else if (k == "--obstacles") {
      const char *v = need_value(k);
      if (!v || !parse_int(v, &cfg->obstacles) || cfg->obstacles < 0) return false;
    } else if (k == "--path-radius") {
      const char *v = need_value(k);
      if (!v || !parse_double(v, &cfg->path_radius) || cfg->path_radius <= 2.0) return false;
    } else if (k == "--seed") {
      const char *v = need_value(k);
      if (!v || !parse_uint(v, &cfg->seed)) return false;
    } else if (k == "--out-ppm") {
      const char *v = need_value(k);
      if (!v) return false;
      cfg->out_ppm = v;
    } else if (k == "--out-gt-ppm") {
      const char *v = need_value(k);
      if (!v) return false;
      cfg->out_gt_ppm = v;
    } else {
      std::cerr << "Unknown option: " << k << "\n";
      return false;
    }
  }

  const double safe_r = std::min(cfg->width, cfg->height) * 0.5 - 8.0;
  if (cfg->path_radius > safe_r) {
    cfg->path_radius = safe_r;
  }
  return true;
}

void add_world_obstacles(std::vector<uint8_t> *occ, const Config &cfg) {
  const int w = cfg.width;
  const int h = cfg.height;

  for (int x = 0; x < w; ++x) {
    (*occ)[index_of(x, 0, w)] = 1;
    (*occ)[index_of(x, h - 1, w)] = 1;
  }
  for (int y = 0; y < h; ++y) {
    (*occ)[index_of(0, y, w)] = 1;
    (*occ)[index_of(w - 1, y, w)] = 1;
  }

  std::mt19937 rng(cfg.seed);
  std::uniform_int_distribution<int> rw(8, std::max(8, w / 12));
  std::uniform_int_distribution<int> rh(8, std::max(8, h / 12));
  std::uniform_int_distribution<int> cx(1, w - 2);
  std::uniform_int_distribution<int> cy(1, h - 2);

  const double mx = 0.5 * static_cast<double>(w);
  const double my = 0.5 * static_cast<double>(h);
  const double forbidden = cfg.path_radius;

  int placed = 0;
  int trials = 0;
  while (placed < cfg.obstacles && trials < cfg.obstacles * 30 + 500) {
    ++trials;
    int ww = rw(rng);
    int hh = rh(rng);
    int cxx = cx(rng);
    int cyy = cy(rng);

    double dx = static_cast<double>(cxx) - mx;
    double dy = static_cast<double>(cyy) - my;
    double dist = std::sqrt(dx * dx + dy * dy);
    if (std::abs(dist - forbidden) < 0.17 * forbidden) {
      continue;
    }

    int x0 = std::max(1, cxx - ww / 2);
    int x1 = std::min(w - 2, cxx + ww / 2);
    int y0 = std::max(1, cyy - hh / 2);
    int y1 = std::min(h - 2, cyy + hh / 2);

    for (int y = y0; y <= y1; ++y) {
      for (int x = x0; x <= x1; ++x) {
        (*occ)[index_of(x, y, w)] = 1;
      }
    }
    ++placed;
  }
}

double cast_ray_range(const std::vector<uint8_t> &occ, int w, int h,
                      double sx, double sy, double angle, double max_range) {
  const double dx = std::cos(angle);
  const double dy = std::sin(angle);

  int x = static_cast<int>(std::floor(sx));
  int y = static_cast<int>(std::floor(sy));
  if (!in_bounds(x, y, w, h)) return max_range;

  const int step_x = (dx > 0.0) ? 1 : (dx < 0.0 ? -1 : 0);
  const int step_y = (dy > 0.0) ? 1 : (dy < 0.0 ? -1 : 0);

  const double inf = std::numeric_limits<double>::infinity();
  double t_max_x = inf;
  double t_max_y = inf;
  double t_delta_x = inf;
  double t_delta_y = inf;

  if (step_x != 0) {
    t_delta_x = std::abs(1.0 / dx);
    if (step_x > 0) {
      t_max_x = (std::floor(sx) + 1.0 - sx) / dx;
    } else {
      t_max_x = (sx - std::floor(sx)) / (-dx);
    }
  }
  if (step_y != 0) {
    t_delta_y = std::abs(1.0 / dy);
    if (step_y > 0) {
      t_max_y = (std::floor(sy) + 1.0 - sy) / dy;
    } else {
      t_max_y = (sy - std::floor(sy)) / (-dy);
    }
  }

  double t = 0.0;
  while (t <= max_range) {
    if (!in_bounds(x, y, w, h)) break;
    if (occ[index_of(x, y, w)] != 0) {
      return t;
    }
    if (t_max_x < t_max_y) {
      x += step_x;
      t = t_max_x;
      t_max_x += t_delta_x;
    } else {
      y += step_y;
      t = t_max_y;
      t_max_y += t_delta_y;
    }
  }
  return max_range;
}

void integrate_beam(std::vector<double> *logodds, int w, int h,
                    double sx, double sy, double angle,
                    double measured_range, double max_range) {
  constexpr double free_update = -0.40;
  constexpr double occ_update = +0.85;

  const double dx = std::cos(angle);
  const double dy = std::sin(angle);

  int x = static_cast<int>(std::floor(sx));
  int y = static_cast<int>(std::floor(sy));
  if (!in_bounds(x, y, w, h)) return;

  const int step_x = (dx > 0.0) ? 1 : (dx < 0.0 ? -1 : 0);
  const int step_y = (dy > 0.0) ? 1 : (dy < 0.0 ? -1 : 0);

  const double inf = std::numeric_limits<double>::infinity();
  double t_max_x = inf;
  double t_max_y = inf;
  double t_delta_x = inf;
  double t_delta_y = inf;

  if (step_x != 0) {
    t_delta_x = std::abs(1.0 / dx);
    if (step_x > 0) {
      t_max_x = (std::floor(sx) + 1.0 - sx) / dx;
    } else {
      t_max_x = (sx - std::floor(sx)) / (-dx);
    }
  }
  if (step_y != 0) {
    t_delta_y = std::abs(1.0 / dy);
    if (step_y > 0) {
      t_max_y = (std::floor(sy) + 1.0 - sy) / dy;
    } else {
      t_max_y = (sy - std::floor(sy)) / (-dy);
    }
  }

  double t = 0.0;
  const double eps = 1e-9;
  while (t < measured_range - eps) {
    if (!in_bounds(x, y, w, h)) break;
    int idx = index_of(x, y, w);
    (*logodds)[idx] = clamp_logodds((*logodds)[idx] + free_update);

    if (t_max_x < t_max_y) {
      x += step_x;
      t = t_max_x;
      t_max_x += t_delta_x;
    } else {
      y += step_y;
      t = t_max_y;
      t_max_y += t_delta_y;
    }
  }

  if (measured_range < max_range - 1e-6) {
    int ex = static_cast<int>(std::floor(sx + dx * measured_range));
    int ey = static_cast<int>(std::floor(sy + dy * measured_range));
    if (in_bounds(ex, ey, w, h)) {
      int eidx = index_of(ex, ey, w);
      (*logodds)[eidx] = clamp_logodds((*logodds)[eidx] + occ_update);
    }
  }
}

std::vector<Pose> make_trajectory(const Config &cfg) {
  std::vector<Pose> poses;
  poses.reserve(cfg.scans);
  const double cx = 0.5 * static_cast<double>(cfg.width);
  const double cy = 0.5 * static_cast<double>(cfg.height);

  for (int i = 0; i < cfg.scans; ++i) {
    double t = (2.0 * kPi * static_cast<double>(i)) / static_cast<double>(cfg.scans);
    double x = cx + cfg.path_radius * std::cos(t);
    double y = cy + cfg.path_radius * std::sin(t);
    double heading = t + kPi;
    poses.push_back({x, y, heading});
  }
  return poses;
}

void write_ppm(const std::string &filename, const std::vector<double> &logodds,
               int w, int h) {
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs) {
    std::cerr << "Warning: cannot open " << filename << " for writing\n";
    return;
  }
  ofs << "P6\n" << w << " " << h << "\n255\n";

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      double lo = logodds[index_of(x, y, w)];
      unsigned char c = 127;
      if (lo > 0.2) {
        c = 0;
      } else if (lo < -0.2) {
        c = 255;
      }
      unsigned char rgb[3] = {c, c, c};
      ofs.write(reinterpret_cast<const char *>(rgb), 3);
    }
  }
}

void write_gt_ppm(const std::string &filename, const std::vector<uint8_t> &occ,
                  int w, int h) {
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs) {
    std::cerr << "Warning: cannot open " << filename << " for writing\n";
    return;
  }
  ofs << "P6\n" << w << " " << h << "\n255\n";

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      unsigned char c = occ[index_of(x, y, w)] ? 0 : 255;
      unsigned char rgb[3] = {c, c, c};
      ofs.write(reinterpret_cast<const char *>(rgb), 3);
    }
  }
}

}  // namespace

int main(int argc, char **argv) {
  Config cfg;
  if (!parse_args(argc, argv, &cfg)) {
    usage(argv[0]);
    return 1;
  }

  std::vector<uint8_t> true_occ(static_cast<size_t>(cfg.width) * cfg.height, 0);
  add_world_obstacles(&true_occ, cfg);

  std::vector<double> est_logodds(static_cast<size_t>(cfg.width) * cfg.height, 0.0);
  std::vector<Pose> poses = make_trajectory(cfg);

  auto t0 = std::chrono::high_resolution_clock::now();

  for (const Pose &p : poses) {
    for (int b = 0; b < cfg.beams; ++b) {
      double rel = (2.0 * kPi * static_cast<double>(b)) / static_cast<double>(cfg.beams);
      double a = p.heading + rel;

      double r = cast_ray_range(true_occ, cfg.width, cfg.height,
                                p.x, p.y, a, cfg.max_range);
      integrate_beam(&est_logodds, cfg.width, cfg.height,
                     p.x, p.y, a, r, cfg.max_range);
    }
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double elapsed_s = std::chrono::duration<double>(t1 - t0).count();

  long long tp = 0, fp = 0, fn = 0;
  double checksum = 0.0;
  for (size_t i = 0; i < est_logodds.size(); ++i) {
    bool pred_occ = est_logodds[i] > 0.0;
    bool gt_occ = true_occ[i] != 0;
    if (pred_occ && gt_occ) ++tp;
    if (pred_occ && !gt_occ) ++fp;
    if (!pred_occ && gt_occ) ++fn;
    checksum += est_logodds[i] * static_cast<double>((i % 1021) + 1);
  }
  double iou = 0.0;
  if (tp + fp + fn > 0) {
    iou = static_cast<double>(tp) / static_cast<double>(tp + fp + fn);
  }

  std::cout << "grid=" << cfg.width << "x" << cfg.height
            << " scans=" << cfg.scans
            << " beams=" << cfg.beams
            << " max_range=" << cfg.max_range
            << " obstacles=" << cfg.obstacles
            << " seed=" << cfg.seed << "\n";
  std::cout << "main_compute_seconds=" << elapsed_s << "\n";
  std::cout << "IoU_occupied=" << iou << " (tp=" << tp
            << ", fp=" << fp << ", fn=" << fn << ")\n";
  std::cout << "checksum=" << checksum << "\n";

  if (!cfg.out_ppm.empty()) {
    write_ppm(cfg.out_ppm, est_logodds, cfg.width, cfg.height);
    std::cout << "wrote_ppm=" << cfg.out_ppm << "\n";
  }
  if (!cfg.out_gt_ppm.empty()) {
    write_gt_ppm(cfg.out_gt_ppm, true_occ, cfg.width, cfg.height);
    std::cout << "wrote_gt_ppm=" << cfg.out_gt_ppm << "\n";
  }

  return 0;
}
