#include "ppm.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace rpy {
namespace {
struct Color { unsigned char r, g, b; };
constexpr Color ink{34,48,65}, blue{43,116,173}, orange{224,104,44};
constexpr Color grid{225,232,238}, muted{100,115,130};

// Five-bit rows of a small built-in font, avoiding a font/library dependency.
std::array<unsigned char,7> glyph(char c) {
    switch(c) {
    case 'A': return {14,17,17,31,17,17,17};
    case 'B': return {30,17,17,30,17,17,30};
    case 'C': return {14,17,16,16,16,17,14};
    case 'D': return {30,17,17,17,17,17,30};
    case 'E': return {31,16,16,30,16,16,31};
    case 'F': return {31,16,16,30,16,16,16};
    case 'G': return {14,17,16,23,17,17,15};
    case 'H': return {17,17,17,31,17,17,17};
    case 'I': return {14,4,4,4,4,4,14};
    case 'J': return {7,2,2,2,2,18,12};
    case 'K': return {17,18,20,24,20,18,17};
    case 'L': return {16,16,16,16,16,16,31};
    case 'M': return {17,27,21,21,17,17,17};
    case 'N': return {17,25,21,19,17,17,17};
    case 'O': return {14,17,17,17,17,17,14};
    case 'P': return {30,17,17,30,16,16,16};
    case 'Q': return {14,17,17,17,21,18,13};
    case 'R': return {30,17,17,30,20,18,17};
    case 'S': return {15,16,16,14,1,1,30};
    case 'T': return {31,4,4,4,4,4,4};
    case 'U': return {17,17,17,17,17,17,14};
    case 'V': return {17,17,17,17,17,10,4};
    case 'W': return {17,17,17,21,21,21,10};
    case 'X': return {17,17,10,4,10,17,17};
    case 'Y': return {17,17,10,4,4,4,4};
    case 'Z': return {31,1,2,4,8,16,31};
    case '0': return {14,17,19,21,25,17,14};
    case '1': return {4,12,4,4,4,4,14};
    case '2': return {14,17,1,2,4,8,31};
    case '3': return {30,1,1,14,1,1,30};
    case '4': return {2,6,10,18,31,2,2};
    case '5': return {31,16,16,30,1,1,30};
    case '6': return {14,16,16,30,17,17,14};
    case '7': return {31,1,2,4,8,8,8};
    case '8': return {14,17,17,14,17,17,14};
    case '9': return {14,17,17,15,1,1,14};
    case '.': return {0,0,0,0,0,6,6};
    case '-': return {0,0,0,31,0,0,0};
    case '+': return {0,4,4,31,4,4,0};
    case '=': return {0,0,31,0,31,0,0};
    case ':': return {0,6,6,0,6,6,0};
    case '/': return {1,1,2,4,8,16,16};
    case '(': return {2,4,8,8,8,4,2};
    case ')': return {8,4,2,2,2,4,8};
    default: return {};
    }
}

class Canvas {
public:
    static constexpr int width = 1100, height = 960;
    std::vector<unsigned char> pixels;
    Canvas() : pixels(width*height*3, 255) {}
    void dot(int x, int y, Color c) {
        if (x < 0 || y < 0 || x >= width || y >= height) return;
        const std::size_t k = (std::size_t(y)*width+x)*3;
        pixels[k]=c.r; pixels[k+1]=c.g; pixels[k+2]=c.b;
    }
    void disk(int x, int y, int radius, Color c) {
        for (int dy=-radius; dy<=radius; ++dy)
            for (int dx=-radius; dx<=radius; ++dx)
                if (dx*dx+dy*dy <= radius*radius) dot(x+dx,y+dy,c);
    }
    void line(double x0, double y0, double x1, double y1, Color c, int radius=0) {
        const int steps=std::max(1, int(std::ceil(std::max(std::abs(x1-x0),std::abs(y1-y0)))));
        for (int s=0; s<=steps; ++s) {
            const double t=double(s)/steps;
            disk(int(std::lround(x0+t*(x1-x0))),int(std::lround(y0+t*(y1-y0))),radius,c);
        }
    }
    void arrow(double x, double y, double dx, double dy, Color c) {
        const double length=std::hypot(dx,dy);
        if (length < 0.5) return; // Below raster resolution, never enlarge short arrows.
        line(x,y,x+dx,y+dy,c,1);
        const double head=std::min(8.0,0.35*length);
        const double ex=dx/length, ey=dy/length;
        line(x+dx,y+dy,x+dx-head*ex+0.5*head*ey,y+dy-head*ey-0.5*head*ex,c,1);
        line(x+dx,y+dy,x+dx-head*ex-0.5*head*ey,y+dy-head*ey+0.5*head*ex,c,1);
    }
    void text(int x, int y, const std::string& value, Color c, int scale=2) {
        for (char ch : value) {
            const auto rows=glyph(ch);
            for (int j=0;j<7;++j) for (int i=0;i<5;++i)
                if (rows[j] & (1 << (4-i)))
                    for (int sy=0;sy<scale;++sy) for (int sx=0;sx<scale;++sx)
                        dot(x+i*scale+sx,y+j*scale+sy,c);
            x+=6*scale;
        }
    }
};

std::string number(double value) {
    std::ostringstream out;
    out << std::uppercase << std::scientific << std::setprecision(2) << value;
    return out.str();
}
}

void write_velocity_ppm(const std::string& path, const std::vector<Vec3>& x,
                        const std::vector<Vec3>& f, const std::vector<Vec3>& u,
                        bool demo, const ImageOptions& options) {
    if (x.empty() || x.size()!=f.size() || x.size()!=u.size())
        throw std::invalid_argument("Invalid image array sizes.");
    if (!std::isfinite(options.thickness) || options.thickness<=0 ||
        (options.z && !std::isfinite(*options.z)) ||
        options.max_arrows==0 || options.max_arrows>4096)
        throw std::invalid_argument("Invalid image slice options.");
    double xmin=x[0].x, xmax=xmin, ymin=x[0].y, ymax=ymin;
    double zmin=x[0].z, zmax=zmin;
    for (std::size_t i=0;i<x.size();++i) {
        for (double v : {x[i].x,x[i].y,x[i].z,f[i].x,f[i].y,f[i].z,u[i].x,u[i].y,u[i].z})
            if (!std::isfinite(v)) throw std::invalid_argument("Non-finite image data.");
        xmin=std::min(xmin,x[i].x); xmax=std::max(xmax,x[i].x);
        ymin=std::min(ymin,x[i].y); ymax=std::max(ymax,x[i].y);
        zmin=std::min(zmin,x[i].z); zmax=std::max(zmax,x[i].z);
    }
    double zcenter=0.5*zmin+0.5*zmax;
    if (options.z) zcenter=*options.z;
    else {
        // Snap to occupied depth so an even number of lattice layers does not
        // place the automatic slab in an empty gap between layers.
        const double midpoint=zcenter;
        double best=std::abs(x[0].z-midpoint);
        zcenter=x[0].z;
        for (const auto& v : x) {
            const double d=std::abs(v.z-midpoint);
            if (d<best) { best=d; zcenter=v.z; }
        }
    }
    std::vector<std::size_t> candidates;
    for (std::size_t i=0;i<x.size();++i)
        if (std::abs(x[i].z-zcenter)<=options.thickness/2) candidates.push_back(i);
    auto forced = [&](std::size_t i) {
        return f[i].x!=0 || f[i].y!=0 || f[i].z!=0;
    };
    std::sort(candidates.begin(),candidates.end(),[&](std::size_t a,std::size_t b) {
        // Keep the single forced particle visible in the planar demonstration.
        if (demo && forced(a)!=forced(b)) return forced(a)>forced(b);
        const double da=std::abs(x[a].z-zcenter), db=std::abs(x[b].z-zcenter);
        return da==db ? a<b : da<db;
    });
    const double span=std::max({xmax-xmin,ymax-ymin,4.0});
    const double spatial_scale=560.0/span;
    const double cx=(xmin+xmax)/2, cy=(ymin+ymax)/2;
    // Suppress close projected origins, even if a user requests a thick slab.
    // Every displayed arrow still represents an actual individual particle.
    const double separation=std::max(22.0,560.0/std::sqrt(double(options.max_arrows)));
    std::vector<std::size_t> selected;
    for (const auto i : candidates) {
        bool clear=true;
        for (const auto j : selected) {
            if (spatial_scale*std::hypot(x[i].x-x[j].x,x[i].y-x[j].y)<separation) {
                clear=false; break;
            }
        }
        if (clear) selected.push_back(i);
        if (selected.size()==options.max_arrows) break;
    }
    double min_spacing=175.0, max_speed=0.0;
    for (std::size_t a=0;a<selected.size();++a) {
        const auto i=selected[a];
        max_speed=std::max(max_speed,std::hypot(u[i].x,u[i].y));
        for (std::size_t b=0;b<a;++b) {
            const auto j=selected[b];
            min_spacing=std::min(min_spacing,
                spatial_scale*std::hypot(x[i].x-x[j].x,x[i].y-x[j].y));
        }
    }
    // A common linear scale, capped by spacing rather than clipping each arrow.
    // Two opposing arrows then occupy at most 80% of their origin separation.
    const double max_arrow_length=std::min(70.0,0.4*min_spacing);
    const double velocity_scale=max_speed>0 ? max_arrow_length/max_speed : 0.0;
    Canvas canvas;
    canvas.text(44,30,"RPY HYDRODYNAMIC RESPONSE",ink,3);
    canvas.text(44,70,demo ? "PLANAR DEMO - ONE PARTICLE FORCED ALONG +X"
                                  : "XY VIEW OF Z SLAB - LAST LOAD CASE",muted);
    canvas.text(44,100,"TOTAL = "+std::to_string(x.size())+
        "   IN SLAB = "+std::to_string(candidates.size())+
        "   SHOWN = "+std::to_string(selected.size()),muted);
    canvas.text(44,130,"Z CENTER = "+number(zcenter)+
        "   FULL THICKNESS = "+number(options.thickness),muted);
    for (int k=0;k<=4;++k) {
        const int s=190+140*k;
        canvas.line(270,s,830,s,grid);
        canvas.line(270+140*k,190,270+140*k,750,grid);
    }
    for (const auto i : selected) {
        const double px=550+(x[i].x-cx)*spatial_scale;
        const double py=470-(x[i].y-cy)*spatial_scale;
        canvas.disk(int(std::lround(px)),int(std::lround(py)),3,forced(i)?orange:blue);
        canvas.arrow(px,py,velocity_scale*u[i].x,-velocity_scale*u[i].y,ink);
    }
    if (selected.empty()) canvas.text(290,460,"NO PARTICLES IN SELECTED Z SLAB",muted);
    canvas.arrow(100,740,55,0,muted); canvas.text(166,734,"X",muted);
    canvas.arrow(100,740,0,-55,muted); canvas.text(94,662,"Y",muted);
    canvas.disk(50,830,4,orange); canvas.text(67,823,"APPLIED FORCE",ink);
    canvas.disk(285,830,4,blue); canvas.text(302,823,"NO APPLIED FORCE",ink);
    if (max_speed>0) canvas.arrow(560,830,max_arrow_length,0,ink);
    canvas.text(650,823,"XY SPEED = "+number(max_speed),ink);
    canvas.text(44,865,"SPATIALLY SAMPLED PARTICLES. ARROWS ARE NOT CELL AVERAGES.",muted);
    canvas.text(44,891,"COMMON LINEAR SPEED SCALE. Z VELOCITY OMITTED. DOTS NOT TO SCALE.",muted);
    canvas.text(44,917,"DIMENSIONLESS VALUES. FULL 3D RESULTS ARE AVAILABLE VIA --OUTPUT.",muted);
    std::ofstream out(path,std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open image: "+path);
    out << "P6\n# RPY instantaneous XY velocities of spatially sampled Z slab\n"
        << std::setprecision(17)
        << "# z_center=" << zcenter << " thickness=" << options.thickness
        << " total=" << x.size() << " in_slab=" << candidates.size()
        << " shown=" << selected.size() << '\n'
        << "# selected_particle_ids=";
    for (const auto i : selected) out << i << ',';
    out << '\n' << "# pixels_per_speed=" << velocity_scale
        << " minimum_origin_separation_pixels=" << separation << '\n'
        << Canvas::width << ' ' << Canvas::height << "\n255\n";
    out.write(reinterpret_cast<const char*>(canvas.pixels.data()),
              static_cast<std::streamsize>(canvas.pixels.size()));
    out.close();
    if (!out) throw std::runtime_error("Failed writing image: "+path);
}
}
