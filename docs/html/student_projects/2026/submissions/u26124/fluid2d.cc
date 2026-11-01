#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Config {
  int nx = 256;
  int ny = 256;
  int steps = 1200;
  int jacobi = 80;
  int diffuse_iters = 20;
  int frame_every = 10;
  int initial_frame = 0;
  double dt = 0.04;
  double visc = 8e-5;
  double gravity = 22.0;
  double drop_radius_ratio = 0.14;
  std::string out_prefix = "drop";
  std::string out_final = "drop_final.ppm";
};

int idx(int x, int y, int nx) { return y * nx + x; }

double clampd(double v, double lo, double hi) {
  return std::max(lo, std::min(hi, v));
}

double smoothstep(double x) {
  const double t = clampd(x, 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

void set_velocity_boundaries(std::vector<double> &u, std::vector<double> &v, int nx,
                             int ny) {
  for (int x = 0; x < nx; ++x) {
    u[idx(x, 0, nx)] = 0.0;
    v[idx(x, 0, nx)] = 0.0;
    u[idx(x, ny - 1, nx)] = 0.0;
    v[idx(x, ny - 1, nx)] = 0.0;
  }
  for (int y = 0; y < ny; ++y) {
    u[idx(0, y, nx)] = 0.0;
    v[idx(0, y, nx)] = 0.0;
    u[idx(nx - 1, y, nx)] = 0.0;
    v[idx(nx - 1, y, nx)] = 0.0;
  }
}

void set_scalar_boundaries(std::vector<double> &f, int nx, int ny) {
  for (int x = 0; x < nx; ++x) {
    f[idx(x, 0, nx)] = f[idx(x, 1, nx)];
    f[idx(x, ny - 1, nx)] = f[idx(x, ny - 2, nx)];
  }
  for (int y = 0; y < ny; ++y) {
    f[idx(0, y, nx)] = f[idx(1, y, nx)];
    f[idx(nx - 1, y, nx)] = f[idx(nx - 2, y, nx)];
  }
}

double sample_bilinear(const std::vector<double> &f, int nx, int ny, double x,
                       double y) {
  x = clampd(x, 0.0, static_cast<double>(nx - 1));
  y = clampd(y, 0.0, static_cast<double>(ny - 1));
  const int x0 = static_cast<int>(x);
  const int y0 = static_cast<int>(y);
  const int x1 = std::min(x0 + 1, nx - 1);
  const int y1 = std::min(y0 + 1, ny - 1);
  const double tx = x - static_cast<double>(x0);
  const double ty = y - static_cast<double>(y0);
  const double f00 = f[idx(x0, y0, nx)];
  const double f10 = f[idx(x1, y0, nx)];
  const double f01 = f[idx(x0, y1, nx)];
  const double f11 = f[idx(x1, y1, nx)];
  const double a = f00 * (1.0 - tx) + f10 * tx;
  const double b = f01 * (1.0 - tx) + f11 * tx;
  return a * (1.0 - ty) + b * ty;
}

void initialize_drop(std::vector<double> &liq, int nx, int ny,
                     double radius_ratio) {
  const double cx = 0.5 * static_cast<double>(nx);
  const double cy = 0.24 * static_cast<double>(ny);
  const double r = radius_ratio * static_cast<double>(std::min(nx, ny));
  const double edge = 2.5;

  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      const double dx = static_cast<double>(x) - cx;
      const double dy = static_cast<double>(y) - cy;
      const double dist = std::sqrt(dx * dx + dy * dy);
      const double t = (r - dist) / edge;
      liq[idx(x, y, nx)] = clampd(0.5 + 0.5 * t, 0.0, 1.0);
    }
  }
  set_scalar_boundaries(liq, nx, ny);
}

void advect_vector(const std::vector<double> &u0, const std::vector<double> &v0,
                   std::vector<double> &u, std::vector<double> &v, int nx, int ny,
                   double dt) {
  for (int y = 1; y < ny - 1; ++y) {
    for (int x = 1; x < nx - 1; ++x) {
      const int k = idx(x, y, nx);
      const double x_prev = static_cast<double>(x) - dt * u0[k];
      const double y_prev = static_cast<double>(y) - dt * v0[k];
      u[k] = sample_bilinear(u0, nx, ny, x_prev, y_prev);
      v[k] = sample_bilinear(v0, nx, ny, x_prev, y_prev);
    }
  }
  set_velocity_boundaries(u, v, nx, ny);
}

void advect_scalar(const std::vector<double> &src, const std::vector<double> &u,
                   const std::vector<double> &v, std::vector<double> &dst, int nx,
                   int ny, double dt) {
  for (int y = 1; y < ny - 1; ++y) {
    for (int x = 1; x < nx - 1; ++x) {
      const int k = idx(x, y, nx);
      const double x_prev = static_cast<double>(x) - dt * u[k];
      const double y_prev = static_cast<double>(y) - dt * v[k];
      dst[k] = sample_bilinear(src, nx, ny, x_prev, y_prev);
    }
  }
  set_scalar_boundaries(dst, nx, ny);
}

void diffuse_velocity(std::vector<double> &u, std::vector<double> &v,
                      std::vector<double> &u_tmp, std::vector<double> &v_tmp,
                      int nx, int ny, double a, int iters) {
  if (a <= 0.0 || iters <= 0) {
    return;
  }
  const double denom = 1.0 + 4.0 * a;
  const std::vector<double> u0 = u;
  const std::vector<double> v0 = v;

  for (int it = 0; it < iters; ++it) {
    for (int y = 1; y < ny - 1; ++y) {
      for (int x = 1; x < nx - 1; ++x) {
        const int k = idx(x, y, nx);
        u_tmp[k] =
            (u0[k] + a * (u[idx(x - 1, y, nx)] + u[idx(x + 1, y, nx)] +
                           u[idx(x, y - 1, nx)] + u[idx(x, y + 1, nx)])) /
            denom;
        v_tmp[k] =
            (v0[k] + a * (v[idx(x - 1, y, nx)] + v[idx(x + 1, y, nx)] +
                           v[idx(x, y - 1, nx)] + v[idx(x, y + 1, nx)])) /
            denom;
      }
    }
    std::swap(u, u_tmp);
    std::swap(v, v_tmp);
    set_velocity_boundaries(u, v, nx, ny);
  }
}

void project(std::vector<double> &u, std::vector<double> &v,
             std::vector<double> &p, std::vector<double> &div, int nx, int ny,
             int jacobi_iters) {
  const double h = 1.0 / static_cast<double>(std::min(nx, ny));
  for (int y = 1; y < ny - 1; ++y) {
    for (int x = 1; x < nx - 1; ++x) {
      div[idx(x, y, nx)] =
          -0.5 * h * (u[idx(x + 1, y, nx)] - u[idx(x - 1, y, nx)] +
                      v[idx(x, y + 1, nx)] - v[idx(x, y - 1, nx)]);
      p[idx(x, y, nx)] = 0.0;
    }
  }
  set_scalar_boundaries(div, nx, ny);
  set_scalar_boundaries(p, nx, ny);

  std::vector<double> p_tmp(p.size(), 0.0);
  for (int it = 0; it < jacobi_iters; ++it) {
    for (int y = 1; y < ny - 1; ++y) {
      for (int x = 1; x < nx - 1; ++x) {
        p_tmp[idx(x, y, nx)] =
            0.25 * (div[idx(x, y, nx)] + p[idx(x - 1, y, nx)] +
                    p[idx(x + 1, y, nx)] + p[idx(x, y - 1, nx)] +
                    p[idx(x, y + 1, nx)]);
      }
    }
    std::swap(p, p_tmp);
    set_scalar_boundaries(p, nx, ny);
  }

  for (int y = 1; y < ny - 1; ++y) {
    for (int x = 1; x < nx - 1; ++x) {
      u[idx(x, y, nx)] -=
          0.5 * (p[idx(x + 1, y, nx)] - p[idx(x - 1, y, nx)]) / h;
      v[idx(x, y, nx)] -=
          0.5 * (p[idx(x, y + 1, nx)] - p[idx(x, y - 1, nx)]) / h;
    }
  }
  set_velocity_boundaries(u, v, nx, ny);
}

void add_body_forces(std::vector<double> &u, std::vector<double> &v,
                     const std::vector<double> &liq, int nx, int ny,
                     const Config &cfg) {
  for (int y = 1; y < ny - 1; ++y) {
    for (int x = 1; x < nx - 1; ++x) {
      const int k = idx(x, y, nx);
      const double w = smoothstep((liq[k] - 0.02) / 0.98);
      v[k] += cfg.dt * cfg.gravity * w;
    }
  }
}

void confine_velocity_to_drop(std::vector<double> &u, std::vector<double> &v,
                              const std::vector<double> &liq, int nx, int ny) {
  for (int y = 1; y < ny - 1; ++y) {
    for (int x = 1; x < nx - 1; ++x) {
      const int k = idx(x, y, nx);
      const double w = smoothstep((liq[k] - 0.02) / 0.98);
      const double damp = 0.04 + 0.96 * w;
      u[k] *= damp;
      v[k] *= damp;
    }
  }
  set_velocity_boundaries(u, v, nx, ny);
}

void sharpen_interface(std::vector<double> &liq, int nx, int ny) {
  for (int y = 1; y < ny - 1; ++y) {
    for (int x = 1; x < nx - 1; ++x) {
      const int k = idx(x, y, nx);
      const double q = clampd(liq[k], 0.0, 1.0);
      const double s = 1.0 / (1.0 + std::exp(-10.0 * (q - 0.5)));
      liq[k] = clampd(0.85 * q + 0.15 * s, 0.0, 1.0);
    }
  }
  set_scalar_boundaries(liq, nx, ny);
}

void write_drop_ppm(const std::string &path, const std::vector<double> &liq, int nx,
                    int ny) {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) {
    throw std::runtime_error("failed to open output file: " + path);
  }
  ofs << "P6\n" << nx << " " << ny << "\n255\n";

  const double lx = -0.35;
  const double ly = -0.45;
  const double lz = 0.82;

  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      const int xm = std::max(0, x - 1);
      const int xp = std::min(nx - 1, x + 1);
      const int ym = std::max(0, y - 1);
      const int yp = std::min(ny - 1, y + 1);

      const double q = clampd(liq[idx(x, y, nx)], 0.0, 1.0);
      const double gx = liq[idx(xp, y, nx)] - liq[idx(xm, y, nx)];
      const double gy = liq[idx(x, yp, nx)] - liq[idx(x, ym, nx)];

      double nxn = -gx;
      double nyn = -gy;
      double nzn = 1.6;
      const double nlen = std::sqrt(nxn * nxn + nyn * nyn + nzn * nzn);
      nxn /= nlen;
      nyn /= nlen;
      nzn /= nlen;

      const double diff = std::max(0.0, nxn * lx + nyn * ly + nzn * lz);
      const double rim = clampd(std::sqrt(gx * gx + gy * gy) * 2.5, 0.0, 1.0);
      const double alpha = smoothstep((q - 0.03) / 0.97);

      double r = 246.0;
      double g = 249.0;
      double b = 252.0;

      const double wr = 28.0 + 92.0 * diff + 45.0 * rim;
      const double wg = 110.0 + 95.0 * diff + 30.0 * rim;
      const double wb = 188.0 + 60.0 * diff + 25.0 * rim;

      r = (1.0 - alpha) * r + alpha * wr;
      g = (1.0 - alpha) * g + alpha * wg;
      b = (1.0 - alpha) * b + alpha * wb;

      const unsigned char rc = static_cast<unsigned char>(clampd(r, 0.0, 255.0));
      const unsigned char gc = static_cast<unsigned char>(clampd(g, 0.0, 255.0));
      const unsigned char bc = static_cast<unsigned char>(clampd(b, 0.0, 255.0));
      ofs.write(reinterpret_cast<const char *>(&rc), 1);
      ofs.write(reinterpret_cast<const char *>(&gc), 1);
      ofs.write(reinterpret_cast<const char *>(&bc), 1);
    }
  }
}

double max_abs_divergence(const std::vector<double> &u, const std::vector<double> &v,
                         int nx, int ny) {
  const double h = 1.0 / static_cast<double>(std::min(nx, ny));
  double m = 0.0;
  for (int y = 1; y < ny - 1; ++y) {
    for (int x = 1; x < nx - 1; ++x) {
      const double d = 0.5 * h * (u[idx(x + 1, y, nx)] - u[idx(x - 1, y, nx)] +
                                  v[idx(x, y + 1, nx)] - v[idx(x, y - 1, nx)]);
      m = std::max(m, std::abs(d));
    }
  }
  return m;
}

double liquid_volume(const std::vector<double> &liq) {
  double s = 0.0;
  for (double v : liq) {
    s += v;
  }
  return s;
}

double max_speed(const std::vector<double> &u, const std::vector<double> &v) {
  double m = 0.0;
  for (std::size_t i = 0; i < u.size(); ++i) {
    m = std::max(m, std::sqrt(u[i] * u[i] + v[i] * v[i]));
  }
  return m;
}

std::string frame_name(const std::string &prefix, int id) {
  std::ostringstream oss;
  oss << prefix << "_" << std::setfill('0') << std::setw(4) << id << ".ppm";
  return oss.str();
}

void print_usage(const char *prog) {
  std::cout
      << "Usage: " << prog << " [options]\n"
      << "  --nx N               grid width (default: 256)\n"
      << "  --ny N               grid height (default: 256)\n"
      << "  --steps N            number of time steps (default: 1200)\n"
      << "  --dt X               timestep (default: 0.04)\n"
      << "  --visc X             kinematic viscosity (default: 8e-5)\n"
      << "  --jacobi N           pressure Jacobi iterations (default: 80)\n"
      << "  --diffuse-iters N    velocity diffusion iterations (default: 20)\n"
      << "  --gravity X          downward acceleration (default: 22.0)\n"
      << "  --drop-radius X      initial drop radius ratio (default: 0.14)\n"
      << "  --frame-every N      save one frame every N steps (default: 10)\n"
      << "  --initial-frame N    index of first frame (default: 0)\n"
      << "  --out-prefix NAME    frame prefix (default: drop)\n"
      << "  --out-final FILE     final image filename (default: drop_final.ppm)\n"
      << "  --help               show this message\n";
}

bool parse_int(const std::string &s, int &v) {
  try {
    std::size_t pos = 0;
    v = std::stoi(s, &pos);
    return pos == s.size();
  } catch (...) {
    return false;
  }
}

bool parse_double(const std::string &s, double &v) {
  try {
    std::size_t pos = 0;
    v = std::stod(s, &pos);
    return pos == s.size();
  } catch (...) {
    return false;
  }
}

bool parse_args(int argc, char **argv, Config &cfg) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need_value = [&](const std::string &name) {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << name << "\n";
        return false;
      }
      return true;
    };

    if (arg == "--help") {
      print_usage(argv[0]);
      return false;
    } else if (arg == "--nx") {
      if (!need_value(arg) || !parse_int(argv[++i], cfg.nx)) return false;
    } else if (arg == "--ny") {
      if (!need_value(arg) || !parse_int(argv[++i], cfg.ny)) return false;
    } else if (arg == "--steps") {
      if (!need_value(arg) || !parse_int(argv[++i], cfg.steps)) return false;
    } else if (arg == "--jacobi") {
      if (!need_value(arg) || !parse_int(argv[++i], cfg.jacobi)) return false;
    } else if (arg == "--diffuse-iters") {
      if (!need_value(arg) || !parse_int(argv[++i], cfg.diffuse_iters)) return false;
    } else if (arg == "--frame-every") {
      if (!need_value(arg) || !parse_int(argv[++i], cfg.frame_every)) return false;
    } else if (arg == "--initial-frame") {
      if (!need_value(arg) || !parse_int(argv[++i], cfg.initial_frame)) return false;
    } else if (arg == "--dt") {
      if (!need_value(arg) || !parse_double(argv[++i], cfg.dt)) return false;
    } else if (arg == "--visc") {
      if (!need_value(arg) || !parse_double(argv[++i], cfg.visc)) return false;
    } else if (arg == "--gravity") {
      if (!need_value(arg) || !parse_double(argv[++i], cfg.gravity)) return false;
    } else if (arg == "--drop-radius") {
      if (!need_value(arg) || !parse_double(argv[++i], cfg.drop_radius_ratio)) {
        return false;
      }
    } else if (arg == "--out-prefix") {
      if (!need_value(arg)) return false;
      cfg.out_prefix = argv[++i];
    } else if (arg == "--out-final") {
      if (!need_value(arg)) return false;
      cfg.out_final = argv[++i];
    } else {
      std::cerr << "unknown option: " << arg << "\n";
      return false;
    }
  }

  if (cfg.nx < 4 || cfg.ny < 4 || cfg.steps < 1 || cfg.dt <= 0.0 ||
      cfg.jacobi < 1 || cfg.diffuse_iters < 0 || cfg.visc < 0.0 ||
      cfg.drop_radius_ratio <= 0.02 || cfg.drop_radius_ratio >= 0.45 ||
      cfg.frame_every < 0 || cfg.initial_frame < 0) {
    std::cerr << "invalid parameter values\n";
    return false;
  }

  return true;
}

}  // namespace

int main(int argc, char **argv) {
  Config cfg;
  if (!parse_args(argc, argv, cfg)) {
    if (argc > 1 && std::string(argv[1]) == "--help") {
      return 0;
    }
    print_usage(argv[0]);
    return 1;
  }

  const int n = cfg.nx * cfg.ny;
  std::vector<double> u(n, 0.0), v(n, 0.0), u_tmp(n, 0.0), v_tmp(n, 0.0);
  std::vector<double> p(n, 0.0), div(n, 0.0), liq(n, 0.0), liq_tmp(n, 0.0);
  initialize_drop(liq, cfg.nx, cfg.ny, cfg.drop_radius_ratio);

  int frames_written = 0;
  auto write_frame = [&](int frame_id) {
    write_drop_ppm(frame_name(cfg.out_prefix, frame_id), liq, cfg.nx, cfg.ny);
    ++frames_written;
  };

  if (cfg.frame_every > 0) {
    write_frame(cfg.initial_frame);
  }

  const auto t0 = std::chrono::steady_clock::now();
  int next_frame = cfg.initial_frame + 1;

  for (int step = 0; step < cfg.steps; ++step) {
    add_body_forces(u, v, liq, cfg.nx, cfg.ny, cfg);

    u_tmp = u;
    v_tmp = v;
    advect_vector(u_tmp, v_tmp, u, v, cfg.nx, cfg.ny, cfg.dt);

    diffuse_velocity(u, v, u_tmp, v_tmp, cfg.nx, cfg.ny, cfg.visc * cfg.dt,
                     cfg.diffuse_iters);

    project(u, v, p, div, cfg.nx, cfg.ny, cfg.jacobi);

    liq_tmp = liq;
    advect_scalar(liq_tmp, u, v, liq, cfg.nx, cfg.ny, cfg.dt);
    sharpen_interface(liq, cfg.nx, cfg.ny);

    confine_velocity_to_drop(u, v, liq, cfg.nx, cfg.ny);
    project(u, v, p, div, cfg.nx, cfg.ny, cfg.jacobi / 2 + 1);

    if (cfg.frame_every > 0 && (step + 1) % cfg.frame_every == 0) {
      write_frame(next_frame);
      ++next_frame;
    }
  }

  const auto t1 = std::chrono::steady_clock::now();
  const double sec =
      std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();

  if (!cfg.out_final.empty()) {
    write_drop_ppm(cfg.out_final, liq, cfg.nx, cfg.ny);
  }

  const double max_div = max_abs_divergence(u, v, cfg.nx, cfg.ny);
  const double vol = liquid_volume(liq);
  const double vmax = max_speed(u, v);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "grid=" << cfg.nx << "x" << cfg.ny << " steps=" << cfg.steps
            << " dt=" << cfg.dt << "\n";
  std::cout << "compute_time_sec=" << sec << "\n";
  std::cout << "max_abs_divergence=" << max_div << "\n";
  std::cout << "liquid_volume=" << vol << "\n";
  std::cout << "max_speed=" << vmax << "\n";
  std::cout << "frames_written=" << frames_written << "\n";
  if (!cfg.out_final.empty()) {
    std::cout << "final_image=" << cfg.out_final << "\n";
  }

  return 0;
}
