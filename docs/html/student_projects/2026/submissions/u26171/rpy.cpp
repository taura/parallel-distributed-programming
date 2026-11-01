#include "rpy.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace rpy {
namespace {
constexpr double pi = 3.14159265358979323846;

bool finite(Vec3 v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

void expect_close(double actual, double expected, const char* label) {
    const double tolerance = 1e-12 * (1.0 + std::abs(expected));
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance)
        throw std::runtime_error(std::string("Check failed: ") + label);
}
} // namespace

void validate_geometry(const std::vector<Vec3>& x, const Parameters& p) {
    if (!(p.radius > 0.0) || !std::isfinite(p.radius) ||
        !(p.viscosity > 0.0) || !std::isfinite(p.viscosity))
        throw std::invalid_argument("Radius and viscosity must be finite and positive.");
    for (const auto v : x)
        if (!finite(v)) throw std::invalid_argument("Non-finite position.");
    const double diameter = 2.0 * p.radius;
    for (std::size_t i = 0; i < x.size(); ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            const double d = std::hypot(x[i].x - x[j].x,
                                       x[i].y - x[j].y, x[i].z - x[j].z);
            if (!std::isfinite(d) || d < diameter)
                throw std::invalid_argument("Overlapping spheres or invalid distance.");
        }
    }
}

void apply_mobility_serial(const std::vector<Vec3>& x,
                           const std::vector<Vec3>& f,
                           const Parameters& p,
                           std::vector<Vec3>& u) {
    if (x.size() != f.size() || x.size() != u.size() ||
        &x == &u || &f == &u)
        throw std::invalid_argument("Invalid array sizes or aliased output.");

    const double self = 1.0 / (6.0 * pi * p.viscosity * p.radius);
    const double prefactor = 1.0 / (8.0 * pi * p.viscosity);
    const double a2 = p.radius * p.radius;

    for (std::size_t i = 0; i < x.size(); ++i) {
        Vec3 sum{self * f[i].x, self * f[i].y, self * f[i].z};
        for (std::size_t j = 0; j < x.size(); ++j) {
            if (i == j) continue;
            const double dx = x[i].x - x[j].x;
            const double dy = x[i].y - x[j].y;
            const double dz = x[i].z - x[j].z;
            const double inv_r = 1.0 / std::sqrt(dx*dx + dy*dy + dz*dz);
            const double ex = dx * inv_r, ey = dy * inv_r, ez = dz * inv_r;
            const double q = a2 * inv_r * inv_r;
            const double A = prefactor * inv_r * (1.0 + 2.0*q/3.0);
            const double B = prefactor * inv_r * (1.0 - 2.0*q);
            const double projection = ex*f[j].x + ey*f[j].y + ez*f[j].z;
            sum.x += A*f[j].x + B*ex*projection;
            sum.y += A*f[j].y + B*ey*projection;
            sum.z += A*f[j].z + B*ez*projection;
        }
        u[i] = sum;
    }
}

void run_self_checks() {
    const Parameters p{};
    const double mu0 = 1.0 / (6.0*pi);
    std::vector<Vec3> x{{0,0,0}}, f{{1,2,-3}}, u(1);
    validate_geometry(x, p);
    apply_mobility_serial(x, f, p, u);
    expect_close(u[0].x, mu0, "single sphere x");
    expect_close(u[0].y, 2*mu0, "single sphere y");
    expect_close(u[0].z, -3*mu0, "single sphere z");

    x = {{0,0,0}, {4,0,0}};
    f = {{1,0,0}, {0,0,0}};
    u.resize(2);
    validate_geometry(x, p);
    apply_mobility_serial(x, f, p, u);
    expect_close(u[1].x, (23.0/64.0)*mu0, "axial pair response");
    expect_close(u[1].y, 0, "axial cross component");
    f[0] = {0,1,0};
    apply_mobility_serial(x, f, p, u);
    expect_close(u[1].y, (25.0/128.0)*mu0, "transverse pair response");

    x[1] = {0,4,0};
    apply_mobility_serial(x, f, p, u);
    expect_close(u[1].y, (23.0/64.0)*mu0, "rotated axial response");
    apply_mobility_serial(x, f, Parameters{1,2}, u);
    expect_close(u[1].y, (23.0/128.0)*mu0, "viscosity scaling");

    x[1] = {1,0,0};
    bool rejected = false;
    try { validate_geometry(x, p); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("Overlap was not rejected.");
}
} // namespace rpy
