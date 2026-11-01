// Serial rendering pipeline, frame buffers, timing, and diagnostic work maps.
// Scene data is read-only during rendering; framebuffer ownership is exclusive.
#ifndef PD02_PIPELINE_H_
#define PD02_PIPELINE_H_

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "reporting.h"
#include "scene.h"

inline constexpr std::size_t no_triangle = std::numeric_limits<std::size_t>::max();
inline constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
inline constexpr std::uint64_t fnv_prime = 1099511628211ULL;

// Runs whose estimated memory exceeds this many GiB are refused before the scene
// is generated. Raise it on purpose if a node really has more memory to use.
inline constexpr double memory_limit_gib = 64.0;

struct ScreenVertex {
    // Pixel coordinates: origin at top left, x rightwards, y downwards.
    // inv_z is positive reciprocal camera-space depth in inverse scene units.
    double x, y, inv_z;
    // The normal is in camera space and weighted for perspective interpolation.
    Vec3 normal_over_z;
};
struct Rotation { double cx, sx, cy, sy; };
struct Framebuffer {
    int width, height;
    // Row-major storage: one depth and owner per pixel, three interleaved RGB
    // bytes per pixel. A winning fragment must update all three arrays together.
    std::vector<double> inv_depth;
    std::vector<std::size_t> owner;
    std::vector<std::uint8_t> rgb;

    // Allocates w by h pixels; dimensions must be in [0,32768]. Zero dimensions
    // are allowed for checksum tests. Call clear_framebuffer before rendering.
    Framebuffer(int w, int h) : width(w), height(h),
        inv_depth(static_cast<std::size_t>(w) * h),
        owner(static_cast<std::size_t>(w) * h),
        rgb(static_cast<std::size_t>(w) * h * 3) {}
};

// Returns a rotation about Y followed by X, for finite angles x/y in radians.
// Pure: calls are independent of geometry generation and framebuffer access.
inline Rotation make_rotation(double x, double y) {
    return {std::cos(x), std::sin(x), std::cos(y), std::sin(y)};
}

// Returns p rotated by r without translation; accepts positions or directions.
// Requires finite p and r from make_rotation. Calls share no mutable state.
inline Vec3 rotate(Vec3 p, const Rotation& r) {
    const Vec3 q = {r.cy*p.x + r.sy*p.z, p.y, -r.sy*p.x + r.cy*p.z};
    return {q.x, r.cx*q.y - r.sx*q.z, r.sx*q.y + r.cx*q.z};
}

// Projects scene into projected for a camera at the origin looking along +z.
// frame is a zero-based index in [0,999999]; each instance uses its own spin in
// radians/frame, and the terrain is static. width/height are pixels in [1,32768].
// Requires valid object indices, positive uniform scales,
// finite geometry, and projected.size() == scene.vertices.size(). Throws if a
// transformed vertex has z <= 0.1 scene units; near-plane clipping is unsupported.
// Vertex calculations are independent and write distinct projected[i] entries.
// Keep scene immutable and projected storage fixed; finish before render_frame.
inline void transform_vertices(const Scene& scene, int frame, int width, int height,
                               std::vector<ScreenVertex>& projected) {
    std::vector<Rotation> rotations;
    for (const Object& object : scene.objects)
        rotations.push_back(make_rotation(object.angle_x, object.angle_y + object.spin*frame));
    // One scale for both axes preserves proportions at any image aspect ratio.
    const double focal = 0.9*std::min(width, height);
    for (std::size_t i = 0; i < scene.vertices.size(); ++i) {
        const Vertex& v = scene.vertices[i];
        const Object& object = scene.objects[v.object];
        const Vec3 p = rotate(v.position, rotations[v.object])*object.scale + object.center;
        const Vec3 n = rotate(v.normal, rotations[v.object]);
        // Generated objects stay entirely beyond the near plane for every frame.
        // A general model loader would require triangle clipping here.
        if (p.z <= near_plane) throw std::runtime_error("Scene crosses the near plane");
        const double inv_z = 1.0/p.z;
        projected[i] = {width*0.5 + focal*p.x*inv_z,
                       height*0.5 - focal*p.y*inv_z, inv_z, n*inv_z};
    }
}

// Resets fb to the background with no visible surface. Requires constructor-sized
// arrays and exclusive access to fb. Pixel iterations write disjoint elements.
// Can overlap transform_vertices, but must finish before rendering into fb.
inline void clear_framebuffer(Framebuffer& fb) {
    for (std::size_t i = 0; i < fb.owner.size(); ++i) {
        fb.inv_depth[i] = 0.0; // Farther than every visible positive depth.
        fb.owner[i] = no_triangle;
        fb.rgb[3*i] = 18;
        fb.rgb[3*i+1] = 23;
        fb.rgb[3*i+2] = 33;
    }
}

// Returns the signed edge value for point (x,y) relative to directed edge a->b,
// in squared pixels. Uses only finite screen coordinates; positive means inside
// for clockwise screen-space winding (y downwards). Pure; calls are independent.
inline double edge(const ScreenVertex& a, const ScreenVertex& b, double x, double y) {
    return (b.x-a.x)*(y-a.y) - (b.y-a.y)*(x-a.x);
}

// Returns whether directed edge a->b includes boundary samples under the
// top-left rule. Requires finite pixel coordinates and clockwise winding in
// screen space (y downwards). Pure; calls are independent.
inline bool is_top_left(const ScreenVertex& a, const ScreenVertex& b) {
    const double dy = b.y-a.y, dx = b.x-a.x;
    return dy < 0.0 || (dy == 0.0 && dx > 0.0);
}

// Returns whether an edge test accepts a sample, including boundary ownership.
// value must be the finite result of edge(a,b,x,y) for clockwise screen-space
// winding. Uses exact zero, without an epsilon; calls have no shared writes.
inline bool inside_edge(double value, const ScreenVertex& a, const ScreenVertex& b) {
    return value > 0.0 || (value == 0.0 && is_top_left(a,b));
}

// Returns ambient-plus-diffuse RGB bytes without gamma correction. normal is a
// finite camera-space direction accepted by normalize; color components are
// finite reflectances in [0,1]. A zero normal receives ambient light only.
// Pure: each fragment can be shaded independently; this does not write pixels.
inline std::array<std::uint8_t, 3> shade(Vec3 normal, Vec3 color) {
    const Vec3 light = normalize({-0.4, 0.65, -1.0}); // Surface towards light.
    const double diffuse = std::max(0.0, dot(normalize(normal), light));
    const double intensity = 0.15 + 0.85*diffuse;
    return {{static_cast<std::uint8_t>(std::clamp(color.x*intensity,0.0,1.0)*255.0+0.5),
             static_cast<std::uint8_t>(std::clamp(color.y*intensity,0.0,1.0)*255.0+0.5),
             static_cast<std::uint8_t>(std::clamp(color.z*intensity,0.0,1.0)*255.0+0.5)}};
}

struct TileWork {
    std::uint64_t fragments = 0, triangles = 0;
    std::size_t last_triangle = no_triangle;
};
struct WorkMap {
    static constexpr int tile_size = 32;
    int columns, rows;
    std::vector<TileWork> tiles;
    // Allocates zero counters for positive image dimensions; owns its storage.
    WorkMap(int width, int height) :
        columns((width+tile_size-1)/tile_size), rows((height+tile_size-1)/tile_size),
        tiles(static_cast<std::size_t>(columns)*rows) {}
};

// Draws an opaque triangle into fb using center samples and the top-left rule.
// a/b/c may have either winding; require finite pixel coordinates whose rounded
// bounds fit int, positive inv_z, and finite camera-space normal_over_z.
// color components must be in [0,1]; triangle_id must be a stable unique ID
// below no_triangle. fb must have valid dimensions and initialized arrays.
// Reads and writes shared per-pixel depth, owner, and RGB. Overlapping calls on
// the same fb require synchronization covering the depth test and all writes,
// or exclusive pixel ownership. Separate framebuffers permit independent calls.
// CountWork requires an exclusive map matching fb and one visit per triangle ID.
// The false specialization contains no counter updates or runtime map branches.
template<bool CountWork>
inline void rasterize_triangle_impl(ScreenVertex a, ScreenVertex b, ScreenVertex c,
                                    Vec3 color, std::size_t triangle_id, Framebuffer& fb,
                                    WorkMap* work) {
    double area = edge(a,b,c.x,c.y);
    if (area == 0.0) return;
    if (area < 0.0) {
        std::swap(b,c);
        area = edge(a,b,c.x,c.y);
    }
    const int xmin = std::max(0, static_cast<int>(std::ceil(std::min({a.x,b.x,c.x})-0.5)));
    const int xmax = std::min(fb.width-1, static_cast<int>(std::floor(std::max({a.x,b.x,c.x})-0.5)));
    const int ymin = std::max(0, static_cast<int>(std::ceil(std::min({a.y,b.y,c.y})-0.5)));
    const int ymax = std::min(fb.height-1, static_cast<int>(std::floor(std::max({a.y,b.y,c.y})-0.5)));
    for (int y = ymin; y <= ymax; ++y) {
        for (int x = xmin; x <= xmax; ++x) {
            const double px = x+0.5, py = y+0.5;
            const double e0 = edge(b,c,px,py);
            const double e1 = edge(c,a,px,py);
            const double e2 = edge(a,b,px,py);
            if (!inside_edge(e0,b,c) || !inside_edge(e1,c,a) || !inside_edge(e2,a,b)) continue;
            if constexpr (CountWork) {
                // Count coverage before visibility: hidden fragments still cost
                // work. A triangle contributes only once to each touched tile.
                TileWork& tile = work->tiles[
                    static_cast<std::size_t>(y/WorkMap::tile_size)*work->columns +
                    x/WorkMap::tile_size
                ];
                ++tile.fragments;
                if (tile.last_triangle != triangle_id) {
                    ++tile.triangles;
                    tile.last_triangle = triangle_id;
                }
            }
            const double w0 = e0/area, w1 = e1/area, w2 = e2/area;
            // 1/z is linear in screen-space barycentric coordinates, z is not.
            const double inv_z = w0*a.inv_z + w1*b.inv_z + w2*c.inv_z;
            const std::size_t index = static_cast<std::size_t>(y)*fb.width+x;
            // Stable IDs make exact-depth ties independent of submission order.
            if (inv_z < fb.inv_depth[index] ||
                (inv_z == fb.inv_depth[index] && triangle_id >= fb.owner[index])) continue;
            const Vec3 normal = (a.normal_over_z*w0 + b.normal_over_z*w1 +
                                 c.normal_over_z*w2) * (1.0/inv_z);
            const auto rgb = shade(normal, color);
            // These writes belong to one visibility decision: an atomic depth
            // update alone would not keep the owner and RGB consistent.
            fb.inv_depth[index] = inv_z;
            fb.owner[index] = triangle_id;
            for (int channel = 0; channel < 3; ++channel) fb.rgb[3*index+channel] = rgb[channel];
        }
    }
}

// Normal rendering entry point; requires the rasterize_triangle_impl contract.
// Writes fb exclusively. Diagnostic counters are removed at compile time.
inline void rasterize_triangle(ScreenVertex a, ScreenVertex b, ScreenVertex c,
                               Vec3 color, std::size_t triangle_id, Framebuffer& fb) {
    rasterize_triangle_impl<false>(a,b,c,color,triangle_id,fb,nullptr);
}

// Draws scene using vertices projected for fb's dimensions and the chosen frame.
// Requires valid scene indices, one projected entry per scene vertex, and fb
// cleared before the frame. Scene and vertices are read-only; fb is exclusive.
// Triangle iterations may overlap in fb and are not independent. Frames can be
// rendered independently with separate projected arrays and framebuffers.
inline void render_frame(const Scene& scene, const std::vector<ScreenVertex>& vertices,
                         Framebuffer& fb) {
    for (std::size_t id = 0; id < scene.triangles.size(); ++id) {
        const Triangle& t = scene.triangles[id];
        rasterize_triangle(vertices[t.vertices[0]], vertices[t.vertices[1]],
                           vertices[t.vertices[2]], scene.objects[t.object].color, id, fb);
    }
}

// Returns FNV-1a-64 over fb's row-major RGB bytes, excluding image dimensions
// and the PPM header. fb.rgb must remain unchanged throughout the call.
// Read-only: can overlap write_ppm or checksums of other completed frames.
inline std::uint64_t image_checksum(const Framebuffer& fb) {
    std::uint64_t hash = fnv_offset;
    for (std::uint8_t byte : fb.rgb) hash = (hash ^ byte) * fnv_prime;
    return hash;
}

// Writes fb to filename as binary PPM P6, replacing any existing file.
// Requires positive dimensions, correctly sized RGB storage, and exclusive
// access to the output file. fb.rgb must remain unchanged during the call.
// Throws on open/write failure; can overlap read-only image_checksum calls.
inline void write_ppm(const std::string& filename, const Framebuffer& fb) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open output: " + filename);
    out << "P6\n" << fb.width << ' ' << fb.height << "\n255\n";
    out.write(reinterpret_cast<const char*>(fb.rgb.data()),
              static_cast<std::streamsize>(fb.rgb.size()));
    out.close();
    if (!out) throw std::runtime_error("Cannot write output: " + filename);
}

// Re-renders projected last-frame geometry into scratch storage, outside timing.
// Counts covered samples (including hidden ones) and distinct contributing
// triangles per 32x32 tile. Input scene/vertices are read-only; no shared writes.
inline WorkMap collect_work_map(const Scene& scene, const std::vector<ScreenVertex>& vertices,
                                int width, int height, std::uint64_t expected_hash) {
    WorkMap work(width,height);
    Framebuffer scratch(width,height);
    clear_framebuffer(scratch);
    for (std::size_t id = 0; id < scene.triangles.size(); ++id) {
        const Triangle& t = scene.triangles[id];
        rasterize_triangle_impl<true>(vertices[t.vertices[0]], vertices[t.vertices[1]],
            vertices[t.vertices[2]],scene.objects[t.object].color,id,scratch,&work);
    }
    if (image_checksum(scratch) != expected_hash)
        throw std::runtime_error("Work-map pass differs from normal rendering");
    return work;
}

// Writes every tile, including zero-work and partial edge tiles, in row-major
// order. Coordinates are zero-based tile indices, not pixels. Own out exclusively.
inline void write_work_csv(std::ostream& out, const WorkMap& work) {
    configure_csv(out);
    out << work_header << '\n';
    for (int y = 0; y < work.rows; ++y)
        for (int x = 0; x < work.columns; ++x) {
            const TileWork& tile = work.tiles[static_cast<std::size_t>(y)*work.columns+x];
            out << x << ',' << y << ',' << tile.fragments << ',' << tile.triangles << '\n';
        }
}

// Estimates the resident memory of a run in bytes, from the analytic mesh counts:
// scene vertices and triangles, projected vertices, the framebuffer (plus the
// scratch framebuffer and counters of the work-map pass), frame records and
// instances. Ignores the executable, the C++ runtime and allocator overhead,
// a few MiB. Pure: allocates nothing and reads only o.
inline std::size_t estimate_memory_bytes(const Options& o) {
    const SceneCounts counts = scene_counts(o.subdivisions, o.objects);
    const std::size_t pixels = static_cast<std::size_t>(o.width) * o.height;
    const std::size_t pixel_bytes = sizeof(double) + sizeof(std::size_t) + 3;
    std::size_t bytes = counts.vertices * (sizeof(Vertex) + sizeof(ScreenVertex)) +
        counts.triangles * sizeof(Triangle) + pixels * pixel_bytes +
        static_cast<std::size_t>(o.frames) * sizeof(FrameRecord) +
        (static_cast<std::size_t>(o.objects) + 1) * sizeof(Object);
    if (!o.work_map.empty()) {
        const std::size_t tile = WorkMap::tile_size;
        const std::size_t tiles = ((o.width + tile - 1) / tile) * ((o.height + tile - 1) / tile);
        bytes += pixels * pixel_bytes + tiles * sizeof(TileWork);
    }
    return bytes;
}

// Returns the estimated memory of a run in GiB, or throws if it exceeds
// memory_limit_gib. Called before generating the scene so that an impossible
// combination fails at once instead of after minutes of work. Pure.
inline double check_memory_limit(const Options& o) {
    const double gib = static_cast<double>(estimate_memory_bytes(o)) / (1024.0*1024.0*1024.0);
    if (gib > memory_limit_gib) {
        std::ostringstream message;
        message << std::fixed << std::setprecision(1) << "Estimated memory " << gib
                << " GiB exceeds the limit of " << memory_limit_gib
                << " GiB; reduce --subdiv, --objects, or the image size";
        throw std::runtime_error(message.str());
    }
    return gib;
}

// Runs fixed-count frames with scene immutable, exclusive fb/projected storage,
// and an optional frame writer (injected by tests). Geometry must match o.
// Only clear/transform/raster calls are timed; hashing, records, and frame output
// follow the stop timestamp. Every frame contributes to the sequence checksum.
inline RunResult run_frames(const Options& o, const Scene& scene,
                            std::vector<ScreenVertex>& projected, Framebuffer& fb,
                            const std::function<void(const std::string&,const Framebuffer&)>& save = write_ppm) {
    RunResult r;
    r.vertices = scene.vertices.size();
    r.triangles = scene.triangles.size();
    r.records.resize(o.frames);
    r.sequence_hash = fnv_offset;
    for (int frame = 0; frame < o.frames; ++frame) {
        const auto start = std::chrono::steady_clock::now();
        clear_framebuffer(fb);
        const auto after_clear = std::chrono::steady_clock::now();
        transform_vertices(scene,frame,o.width,o.height,projected);
        const auto after_transform = std::chrono::steady_clock::now();
        render_frame(scene,projected,fb);
        const auto stop = std::chrono::steady_clock::now();
        const double clear_s = std::chrono::duration<double>(after_clear-start).count();
        const double transform_s = std::chrono::duration<double>(after_transform-after_clear).count();
        const double raster_s = std::chrono::duration<double>(stop-after_transform).count();
        r.clear_s += clear_s;
        r.transform_s += transform_s;
        r.raster_s += raster_s;
        r.last_hash = image_checksum(fb);
        r.records[frame] = {frame,clear_s,transform_s,raster_s,r.last_hash};
        // This explicit byte order preserves the pre-instrumentation checksum.
        for (int byte = 0; byte < 8; ++byte)
            r.sequence_hash = (r.sequence_hash ^ ((r.last_hash >> (8*byte)) & 255ULL)) * fnv_prime;
        if (should_save_frame(o,frame))
            save((std::filesystem::path(o.save_frames)/frame_filename(frame)).string(),fb);
    }
    for (std::size_t id : fb.owner) if (id != no_triangle) ++r.covered;
    return r;
}

#endif  // PD02_PIPELINE_H_
