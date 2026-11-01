// Self-contained, serial C++17 baseline: fixed NACA airfoil in a 2-D channel.
// Numerical model: D2Q9 TRT, halfway bounce-back, Zou-He inlet/outlet.
// Project inspiration: https://github.com/AndreaTorti-01/Lattice-Boltzmann-Method
// This standalone implementation was generated with AI assistance in Codex.
// No OpenMP, CUDA, external libraries, or downloaded geometry are required.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lbm {
constexpr int Q = 9;
constexpr int cx[Q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
constexpr int cy[Q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
constexpr int opposite[Q] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
constexpr double weight[Q] = {4.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9,
                              1.0/36, 1.0/36, 1.0/36, 1.0/36};
constexpr double pi = 3.14159265358979323846;
using Clock = std::chrono::steady_clock;

struct Config {
    int nx = 400, ny = 160, steps = 5000, sample_every = 100, ramp_steps = 500;
    double chord = 64, reynolds = 100, umax = 0.04, alpha = 5;
    std::string naca = "2412", prefix = "airfoil";
    bool output = true, frames = false;
};

void usage() {
    std::cout <<
        "Serial C++17 D2Q9/TRT airfoil simulation (lattice units)\n"
        "  --nx N --ny N          lattice size (default 400 x 160)\n"
        "  --steps N             time steps (5000)\n"
        "  --naca DDDD           NACA four-digit airfoil (2412)\n"
        "  --chord X             chord length in lattice spacings (64)\n"
        "  --alpha X             nose-up angle in degrees (5)\n"
        "  --re X                Reynolds number Umax*chord/nu (100)\n"
        "  --umax X              final maximum inlet speed, 0..0.1 (0.04)\n"
        "  --ramp-steps N        smooth startup duration; 0 = immediate (500)\n"
        "  --sample-every N      diagnostics/force sampling interval (100)\n"
        "  --output-prefix PATH  filename prefix; parent must exist (airfoil)\n"
        "  --frames              also write sampled velocity PPM frames\n"
        "  --no-output           no files; retain console diagnostics\n"
        "  --self-test           run equilibrium and symmetry checks\n"
        "  --help                show this message\n";
}

int integer(const std::string& s) {
    std::size_t used = 0;
    const long long v = std::stoll(s, &used);
    if (used != s.size() || v < 0 || v > std::numeric_limits<int>::max())
        throw std::runtime_error("Invalid nonnegative integer: " + s);
    return static_cast<int>(v);
}

double number(const std::string& s) {
    std::size_t used = 0;
    const double v = std::stod(s, &used);
    if (used != s.size() || !std::isfinite(v))
        throw std::runtime_error("Invalid finite number: " + s);
    return v;
}

Config parse(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "--no-output") { c.output = false; continue; }
        if (k == "--frames") { c.frames = true; continue; }
        if (i + 1 >= argc) throw std::runtime_error("Missing value for " + k);
        const std::string v = argv[++i];
        if (k == "--nx") c.nx = integer(v);
        else if (k == "--ny") c.ny = integer(v);
        else if (k == "--steps") c.steps = integer(v);
        else if (k == "--sample-every") c.sample_every = integer(v);
        else if (k == "--ramp-steps") c.ramp_steps = integer(v);
        else if (k == "--chord") c.chord = number(v);
        else if (k == "--re") c.reynolds = number(v);
        else if (k == "--umax") c.umax = number(v);
        else if (k == "--alpha") c.alpha = number(v);
        else if (k == "--naca") c.naca = v;
        else if (k == "--output-prefix") c.prefix = v;
        else throw std::runtime_error("Unknown option: " + k);
    }
    return c;
}

struct Point { double x, y; };
struct Macro { double rho, ux, uy; };
struct Force { double drag = 0, lift = 0; };
struct Diagnostics {
    double min_rho = std::numeric_limits<double>::infinity();
    double max_rho = 0, mass = 0, max_speed = 0, kinetic_energy = 0;
};
struct Sample { int step; Force force; };

class Simulation {
public:
    Config c;
    std::size_t cells, fluid_cells = 0, airfoil_cells = 0;
    double nu, tau_plus, tau_minus;
    // SoA layout: f[q*cells + cell]. post is the second distribution buffer.
    std::vector<double> f, post;
    // 0 = fluid; 1 = channel wall; 2 = airfoil. y increases upwards.
    std::vector<unsigned char> solid;

    explicit Simulation(const Config& cfg) : c(cfg), cells(0) {
        validate();
        cells = static_cast<std::size_t>(c.nx) * static_cast<std::size_t>(c.ny);
        if (cells > std::vector<double>().max_size() / Q)
            throw std::runtime_error("Grid exceeds addressable array size");
        // For the zero-flow test only, use a positive reference speed for nu.
        nu = (c.umax > 0 ? c.umax : 0.04) * c.chord / c.reynolds;
        tau_plus = 0.5 + 3.0 * nu;
        tau_minus = 0.5 + 0.25 / (tau_plus - 0.5); // TRT magic parameter 1/4
        if (!std::isfinite(tau_plus) || !std::isfinite(tau_minus) || tau_plus <= 0.5001)
            throw std::runtime_error("Viscosity too small/invalid: lower --re or increase --chord");
        solid.assign(cells, 0);
        make_geometry();
        f.resize(Q * cells);
        post.resize(Q * cells);
        for (int q = 0; q < Q; ++q)
            std::fill(f.begin() + q*cells, f.begin() + (q+1)*cells, weight[q]);
    }

    std::size_t index(int x, int y) const {
        return static_cast<std::size_t>(y) * c.nx + x;
    }

    Macro macro(std::size_t i) const {
        double rho = 0, mx = 0, my = 0;
        for (int q = 0; q < Q; ++q) {
            const double fq = f[q*cells + i];
            rho += fq; mx += fq*cx[q]; my += fq*cy[q];
        }
        if (!(rho > 0) || !std::isfinite(rho) || !std::isfinite(mx) || !std::isfinite(my))
            throw std::runtime_error("Non-finite/nonpositive fluid state; reduce speed or Reynolds number");
        return {rho, mx/rho, my/rho};
    }

    void collide() {
        const double omega_plus = 1.0 / tau_plus, omega_minus = 1.0 / tau_minus;
        for (std::size_t i = 0; i < cells; ++i) {
            if (solid[i]) continue;
            const Macro m = macro(i);
            const double uu = m.ux*m.ux + m.uy*m.uy;
            double eq[Q];
            for (int q = 0; q < Q; ++q) {
                const double cu = cx[q]*m.ux + cy[q]*m.uy;
                eq[q] = weight[q]*m.rho*(1 + 3*cu + 4.5*cu*cu - 1.5*uu);
            }
            for (int q = 0; q < Q; ++q) {
                const int o = opposite[q];
                const double a = f[q*cells+i], b = f[o*cells+i];
                const double even = 0.5*((a+b) - (eq[q]+eq[o]));
                const double odd  = 0.5*((a-b) - (eq[q]-eq[o]));
                post[q*cells+i] = a - omega_plus*even - omega_minus*odd;
            }
        }
    }

    Force stream() {
        Force force;
        for (int y = 1; y < c.ny-1; ++y) {
            for (int x = 0; x < c.nx; ++x) {
                const std::size_t i = index(x,y);
                if (solid[i]) continue;
                for (int q = 0; q < Q; ++q) {
                    const int sx = x-cx[q], sy = y-cy[q];
                    // Missing inlet/outlet populations are reconstructed below.
                    if (sx < 0 || sx >= c.nx) continue;
                    const std::size_t src = index(sx,sy);
                    if (!solid[src]) {
                        f[q*cells+i] = post[q*cells+src];
                    } else {
                        const double reflected = post[opposite[q]*cells+i];
                        f[q*cells+i] = reflected;
                        if (solid[src] == 2) {
                            // Force on the body, opposite to momentum gained by fluid.
                            force.drag -= 2*reflected*cx[q];
                            force.lift -= 2*reflected*cy[q];
                        }
                    }
                }
            }
        }
        return force;
    }

    void boundaries(int step) {
        const double ramp = c.ramp_steps == 0 ? 1.0 :
            0.5*(1-std::cos(pi*std::min(1.0, static_cast<double>(step)/c.ramp_steps)));
        for (int y = 1; y < c.ny-1; ++y) {
            // Halfway wall locations: y=0.5 and y=ny-1.5.
            const double eta = (y-0.5)/(c.ny-2.0);
            const double ux = ramp*c.umax*4*eta*(1-eta);
            const std::size_t left = index(0,y), right = index(c.nx-1,y);
            auto L = [&](int q) -> double& { return f[q*cells+left]; };
            auto R = [&](int q) -> double& { return f[q*cells+right]; };
            // Left velocity boundary, uy=0. Unknown E, NE, SE populations.
            const double rho = (L(0)+L(2)+L(4)+2*(L(3)+L(6)+L(7)))/(1-ux);
            L(1) = L(3) + (2.0/3)*rho*ux;
            L(5) = L(7) + 0.5*(L(4)-L(2)) + rho*ux/6;
            L(8) = L(6) + 0.5*(L(2)-L(4)) + rho*ux/6;
            // Right pressure boundary, rho=1 and uy=0. Unknown W, NW, SW.
            const double outlet_ux = -1 + R(0)+R(2)+R(4)+2*(R(1)+R(5)+R(8));
            R(3) = R(1) - (2.0/3)*outlet_ux;
            R(6) = R(8) + 0.5*(R(4)-R(2)) - outlet_ux/6;
            R(7) = R(5) + 0.5*(R(2)-R(4)) - outlet_ux/6;
        }
    }

    Force advance(int step) {
        collide();
        const Force force = stream();
        boundaries(step);
        return force;
    }

    Diagnostics diagnostics() const {
        Diagnostics d;
        for (std::size_t i = 0; i < cells; ++i) {
            if (solid[i]) continue;
            const Macro m = macro(i);
            const double speed = std::hypot(m.ux,m.uy);
            if (!std::isfinite(speed)) throw std::runtime_error("Invalid velocity");
            d.min_rho = std::min(d.min_rho,m.rho);
            d.max_rho = std::max(d.max_rho,m.rho);
            d.mass += m.rho;
            d.max_speed = std::max(d.max_speed,speed);
            d.kinetic_energy += 0.5*m.rho*speed*speed;
        }
        return d;
    }

    void write_velocity(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("Cannot create " + path);
        out << "P6\n" << c.nx << ' ' << c.ny << "\n255\n";
        const double scale = c.umax > 0 ? 1.5*c.umax : 1.0;
        for (int y = c.ny-1; y >= 0; --y) {
            for (int x = 0; x < c.nx; ++x) {
                const std::size_t i = index(x,y);
                std::array<unsigned char,3> rgb{};
                if (solid[i] == 2) rgb = {245,245,245};
                else if (!solid[i]) {
                    const Macro m = macro(i);
                    const double t = std::min(1.0,std::hypot(m.ux,m.uy)/scale);
                    // Blue (0) -> cyan -> yellow -> red (>=1.5*Umax).
                    auto channel = [](double v) {
                        return static_cast<unsigned char>(255*std::clamp(v,0.0,1.0));
                    };
                    rgb = {channel(3*t-1),channel(1.5-std::abs(3*t-1.5)),channel(2-3*t)};
                }
                out.write(reinterpret_cast<const char*>(rgb.data()),3);
            }
        }
        if (!out) throw std::runtime_error("Failed writing " + path);
    }

    void write_field(const std::string& path) const {
        std::ofstream out(path);
        if (!out) throw std::runtime_error("Cannot create " + path);
        out << "x,y,solid,rho,ux,uy\n" << std::setprecision(17);
        for (int y = 0; y < c.ny; ++y) for (int x = 0; x < c.nx; ++x) {
            const std::size_t i = index(x,y);
            const Macro m = solid[i] ? Macro{0,0,0} : macro(i);
            out << x << ',' << y << ',' << static_cast<int>(solid[i]) << ','
                << m.rho << ',' << m.ux << ',' << m.uy << '\n';
        }
        if (!out) throw std::runtime_error("Failed writing " + path);
    }

private:
    void validate() const {
        if (c.nx < 32 || c.ny < 24 || c.steps < 1 || c.steps == std::numeric_limits<int>::max() || c.sample_every < 1)
            throw std::runtime_error("Require nx>=32, ny>=24, 1<=steps<INT_MAX, sample-every>=1");
        if (!(c.chord >= 8 && c.chord < c.nx && c.reynolds > 0 && c.umax >= 0 && c.umax <= 0.1 && std::abs(c.alpha) <= 45))
            throw std::runtime_error("Require 8<=chord<nx, re>0, 0<=umax<=0.1, |alpha|<=45");
        if (c.naca.size() != 4 || c.naca.find_first_not_of("0123456789") != std::string::npos)
            throw std::runtime_error("--naca requires four digits, e.g. 2412 or 0012");
        if (c.naca[0] != '0' && c.naca[1] == '0')
            throw std::runtime_error("Cambered NACA airfoil requires a nonzero camber position");
        if (std::stoi(c.naca.substr(2)) == 0 || c.prefix.empty())
            throw std::runtime_error("Airfoil thickness and output prefix must be nonzero/nonempty");
    }

    void make_geometry() {
        const double m = (c.naca[0]-'0')/100.0, p = (c.naca[1]-'0')/10.0;
        const double thickness = std::stoi(c.naca.substr(2))/100.0;
        const double a = c.alpha*pi/180, ca = std::cos(a), sa = std::sin(a);
        const double x0 = 0.25*(c.nx-1), y0 = 0.5*(c.ny-1);
        constexpr int segments = 240;
        std::vector<Point> upper, lower, polygon;
        for (int k = 0; k <= segments; ++k) {
            const double x = 0.5*(1-std::cos(pi*k/segments));
            // Closed trailing edge: coefficient -0.1036.
            const double yt = 5*thickness*(0.2969*std::sqrt(x)-0.1260*x-0.3516*x*x+0.2843*x*x*x-0.1036*x*x*x*x);
            double yc = 0, slope = 0;
            if (m > 0) {
                if (x < p) { yc = m*(2*p*x-x*x)/(p*p); slope = 2*m*(p-x)/(p*p); }
                else { yc = m*((1-2*p)+2*p*x-x*x)/((1-p)*(1-p)); slope = 2*m*(p-x)/((1-p)*(1-p)); }
            }
            const double theta = std::atan(slope);
            auto transform = [&](double px, double py) {
                const double dx = (px-0.25)*c.chord, dy = py*c.chord;
                // Positive alpha is nose-up (clockwise rotation of the chord).
                return Point{x0+0.25*c.chord+ca*dx+sa*dy, y0-sa*dx+ca*dy};
            };
            upper.push_back(transform(x-yt*std::sin(theta),yc+yt*std::cos(theta)));
            lower.push_back(transform(x+yt*std::sin(theta),yc-yt*std::cos(theta)));
        }
        polygon = upper;
        polygon.insert(polygon.end(),lower.rbegin(),lower.rend());
        double xmin = c.nx, xmax = 0, ymin = c.ny, ymax = 0;
        for (const Point& pt : polygon) {
            if (pt.x < 2 || pt.x > c.nx-3 || pt.y < 2 || pt.y > c.ny-3)
                throw std::runtime_error("Airfoil touches domain boundary: enlarge grid or reduce chord/angle");
            xmin = std::min(xmin,pt.x); xmax = std::max(xmax,pt.x);
            ymin = std::min(ymin,pt.y); ymax = std::max(ymax,pt.y);
        }
        for (int x = 0; x < c.nx; ++x) { solid[index(x,0)] = 1; solid[index(x,c.ny-1)] = 1; }
        // Only test the bounding box; preprocessing does not scale with full-grid area times polygon size.
        for (int y = static_cast<int>(std::floor(ymin)); y <= static_cast<int>(std::ceil(ymax)); ++y)
            for (int x = static_cast<int>(std::floor(xmin)); x <= static_cast<int>(std::ceil(xmax)); ++x) {
                bool inside = false;
                for (std::size_t i = 0, j = polygon.size()-1; i < polygon.size(); j=i++) {
                    const Point& u = polygon[i]; const Point& v = polygon[j];
                    if ((u.y > y) != (v.y > y))
                        if (x < (v.x-u.x)*(y-u.y)/(v.y-u.y)+u.x) inside = !inside;
                }
                if (inside) { solid[index(x,y)] = 2; ++airfoil_cells; }
            }
        if (airfoil_cells == 0) throw std::runtime_error("Airfoil is unresolved on this grid");
        fluid_cells = static_cast<std::size_t>(std::count(solid.begin(),solid.end(),0));
    }
};

void force_plot(const std::vector<Sample>& samples, const std::string& path) {
    constexpr int w = 800, h = 400, pad = 24;
    std::vector<unsigned char> pixels(w*h*3,255);
    auto put = [&](int x, int y, std::array<unsigned char,3> color) {
        if (x < 0 || x >= w || y < 0 || y >= h) return;
        for (int k = 0; k < 3; ++k) pixels[3*(y*w+x)+k] = color[k];
    };
    auto line = [&](int x0, int y0, int x1, int y1, std::array<unsigned char,3> color) {
        const int n = std::max(std::abs(x1-x0),std::abs(y1-y0));
        for (int k = 0; k <= n; ++k) {
            const double t = n ? static_cast<double>(k)/n : 0;
            put(static_cast<int>(std::lround(x0+t*(x1-x0))),static_cast<int>(std::lround(y0+t*(y1-y0))),color);
        }
    };
    double lo = 0, hi = 0;
    for (const Sample& s : samples) { lo = std::min({lo,s.force.drag,s.force.lift}); hi = std::max({hi,s.force.drag,s.force.lift}); }
    if (hi-lo < 1e-12) { lo = -1e-12; hi = 1e-12; }
    const double extra = 0.05*(hi-lo); lo -= extra; hi += extra;
    const int last = samples.back().step;
    auto px = [&](int step) { return pad + static_cast<int>((w-2*pad)*static_cast<double>(step)/last); };
    auto py = [&](double force) { return h-pad-static_cast<int>((h-2*pad)*(force-lo)/(hi-lo)); };
    line(pad,pad,pad,h-pad,{90,90,90}); line(pad,py(0),w-pad,py(0),{180,180,180});
    for (std::size_t i = 1; i < samples.size(); ++i) {
        line(px(samples[i-1].step),py(samples[i-1].force.drag),px(samples[i].step),py(samples[i].force.drag),{220,45,35});
        line(px(samples[i-1].step),py(samples[i-1].force.lift),px(samples[i].step),py(samples[i].force.lift),{25,150,65});
    }
    std::ofstream out(path,std::ios::binary);
    if (!out) throw std::runtime_error("Cannot create " + path);
    out << "P6\n# red=drag green=lift; x=time step 0.." << last
        << "; y=force " << lo << ".." << hi << " (lattice units); see forces.csv for values\n"
        << w << ' ' << h << "\n255\n";
    out.write(reinterpret_cast<const char*>(pixels.data()),static_cast<std::streamsize>(pixels.size()));
    if (!out) throw std::runtime_error("Failed writing " + path);
}

void run(const Config& c) {
    Simulation sim(c);
    std::ofstream csv;
    if (c.output) {
        csv.open(c.prefix+"_forces.csv");
        if (!csv) throw std::runtime_error("Cannot create output files; check prefix parent directory");
        csv << "step,drag,lift,rho_min,rho_max,mass,max_speed,kinetic_energy\n" << std::setprecision(17);
    }
    std::cout << std::setprecision(10) << "serial C++ / D2Q9 TRT / double precision\n"
        << "grid=" << c.nx << 'x' << c.ny << " steps=" << c.steps << " NACA=" << c.naca
        << " chord=" << c.chord << " alpha=" << c.alpha << " Re(Umax,chord)=" << c.reynolds
        << " Umax=" << c.umax << '\n'
        << "nu=" << sim.nu << " tau_plus=" << sim.tau_plus << " tau_minus=" << sim.tau_minus
        << " fluid_cells=" << sim.fluid_cells << " airfoil_cells=" << sim.airfoil_cells << '\n';
    std::vector<Sample> samples;
    auto sample = [&](int step, Force force) {
        const Diagnostics d = sim.diagnostics();
        if (!std::isfinite(force.drag) || !std::isfinite(force.lift)) throw std::runtime_error("Invalid force");
        samples.push_back({step,force});
        std::cout << "step=" << step << " drag=" << force.drag << " lift=" << force.lift
            << " rho=[" << d.min_rho << ',' << d.max_rho << "] max_speed=" << d.max_speed << '\n';
        if (c.output) {
            csv << step << ',' << force.drag << ',' << force.lift << ',' << d.min_rho << ',' << d.max_rho
                << ',' << d.mass << ',' << d.max_speed << ',' << d.kinetic_energy << '\n';
            if (!csv) throw std::runtime_error("Failed writing forces CSV");
            if (c.frames) {
                std::ostringstream name;
                name << c.prefix << "_frame_" << std::setw(7) << std::setfill('0') << step << ".ppm";
                sim.write_velocity(name.str());
            }
        }
        if (d.max_speed > 0.2) std::cerr << "Warning: speed exceeds 0.2; low-Mach accuracy/stability may be poor\n";
    };
    sample(0,{});
    double solver_seconds = 0;
    const auto wall_start = Clock::now();
    for (int step = 1; step <= c.steps; ++step) {
        const auto start = Clock::now();
        const Force force = sim.advance(step);
        solver_seconds += std::chrono::duration<double>(Clock::now()-start).count();
        if (step % c.sample_every == 0 || step == c.steps) sample(step,force);
    }
    const double loop_wall_seconds = std::chrono::duration<double>(Clock::now()-wall_start).count();
    std::cout << "solver_seconds=" << solver_seconds << " loop_wall_seconds=" << loop_wall_seconds
        << " fluid_MLUPS=" << (static_cast<double>(sim.fluid_cells)*c.steps/1e6/solver_seconds) << '\n';
    if (c.output) {
        sim.write_velocity(c.prefix+"_velocity.ppm");
        sim.write_field(c.prefix+"_field.csv");
        force_plot(samples,c.prefix+"_forces.ppm");
        csv.close();
        if (!csv) throw std::runtime_error("Failed closing force output");
        std::cout << "Wrote " << c.prefix << "_{velocity.ppm,field.csv,forces.csv,forces.ppm}\n";
    }
}

void self_test() {
    Config c;
    c.nx = 120; c.ny = 64; c.chord = 24; c.naca = "0012"; c.alpha = 0;
    c.steps = 400; c.output = false; c.umax = 0;
    Simulation rest(c);
    Force force;
    for (int step = 1; step <= 100; ++step) force = rest.advance(step);
    const Diagnostics d = rest.diagnostics();
    if (d.max_speed > 1e-12 || std::abs(d.min_rho-1) > 1e-12 || std::abs(d.max_rho-1) > 1e-12 ||
        std::abs(force.drag) > 1e-11 || std::abs(force.lift) > 1e-11)
        throw std::runtime_error("FAIL: equilibrium-at-rest test");
    std::cout << "PASS equilibrium at rest: max_speed=" << d.max_speed << '\n';
    c.umax = 0.025; c.ramp_steps = 100;
    Simulation symmetric(c);
    for (int y = 0; y < c.ny; ++y) for (int x = 0; x < c.nx; ++x)
        if (symmetric.solid[symmetric.index(x,y)] != symmetric.solid[symmetric.index(x,c.ny-1-y)])
            throw std::runtime_error("FAIL: symmetric geometry test");
    double max_lift = 0;
    for (int step = 1; step <= c.steps; ++step) {
        force = symmetric.advance(step);
        max_lift = std::max(max_lift,std::abs(force.lift));
    }
    double error = 0;
    for (int y = 1; y < c.ny-1; ++y) for (int x = 0; x < c.nx; ++x) {
        const auto i = symmetric.index(x,y), j = symmetric.index(x,c.ny-1-y);
        if (symmetric.solid[i]) continue;
        const Macro a = symmetric.macro(i), b = symmetric.macro(j);
        error = std::max({error,std::abs(a.rho-b.rho),std::abs(a.ux-b.ux),std::abs(a.uy+b.uy)});
    }
    if (error > 1e-10 || max_lift > 1e-10) throw std::runtime_error("FAIL: flow reflection symmetry test");
    std::cout << "PASS symmetric NACA0012: mirror_error=" << error << " max_abs_lift=" << max_lift << '\n';
    Config bad = c; bad.chord = 1000;
    bool rejected = false;
    try { Simulation invalid(bad); } catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("FAIL: invalid geometry accepted");
    std::cout << "PASS invalid configuration rejection\n";
}
} // namespace lbm

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") { lbm::usage(); return 0; }
        if (argc == 2 && std::string(argv[1]) == "--self-test") { lbm::self_test(); return 0; }
        lbm::run(lbm::parse(argc,argv));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\nUse --help for options.\n";
        return 1;
    }
}
