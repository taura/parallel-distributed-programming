// Scene geometry, shared vector types, and deterministic procedural generation.
// All positions use scene units; generators own their PRNG and mesh storage.
#ifndef PD02_SCENE_H_
#define PD02_SCENE_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

inline constexpr double pi = 3.14159265358979323846;
inline constexpr double near_plane = 0.1;
inline constexpr double ground_y = -1.8;
inline constexpr std::uint32_t scene_seed = 0x12345678u;

struct Vec3 { double x, y, z; };
// These arithmetic helpers are pure and have no shared mutable state.
inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline Vec3 operator*(Vec3 a, double s) { return {a.x*s, a.y*s, a.z*s}; }
inline double dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline bool operator==(Vec3 a, Vec3 b) { return a.x == b.x && a.y == b.y && a.z == b.z; }

// Returns a normalized direction, or zero if the computed length is zero.
// Requires finite components and a finite squared length. Calls are independent.
inline Vec3 normalize(Vec3 a) {
    const double length = std::sqrt(dot(a, a));
    return length > 0.0 ? a * (1.0 / length) : Vec3{0.0, 0.0, 0.0};
}

// Positions use arbitrary scene units; normals are dimensionless directions.
// Vertex positions and normals are object-local; object indexes Scene::objects.
struct Vertex { Vec3 position, normal; std::size_t object; };
struct Triangle { std::array<std::size_t, 3> vertices; std::size_t object; };
enum class Shape { Terrain, Torus, Cube, Sphere };

// Centers/radii use scene units, colors [0,1], and angles radians. Scale is
// positive and uniform; spin is radians/frame. Radius bounds the scaled shape
// under every rotation; it is unused for the unrotated terrain.
struct Object {
    Vec3 center, color;
    double angle_x, angle_y, scale, spin, radius;
    Shape shape;
};
struct Scene {
    std::vector<Vertex> vertices;
    std::vector<Triangle> triangles;
    std::vector<Object> objects;
};

// Fixed-width xorshift32 state belongs to its generator, not to the process.
// Separate generators are independent; calls on the same generator are ordered.
struct Xorshift32 {
    std::uint32_t state;

    // Starts the reproducible stream at seed. Zero is an absorbing state and
    // is rejected rather than silently producing a constant sequence.
    explicit Xorshift32(std::uint32_t seed) : state(seed) {
        if (seed == 0) throw std::runtime_error("Xorshift seed must be nonzero");
    }

    // Returns the next 32-bit value and advances this generator's nonzero state.
    // Requires exclusive access to this generator; does not touch scene data.
    std::uint32_t next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }

    // Returns a sample in [0,1) and advances this generator. The power-of-two
    // conversion is exact in double and independent of standard RNG libraries.
    double unit() { return static_cast<double>(next()) / 4294967296.0; }
};

// Appends a torus centered at the local origin, with ring/tube radii 0.78/0.26
// scene units. Requires s in [2,1024] and object indexing scene.objects.
// Adds 8*s*s vertices and 16*s*s triangles. Requires exclusive access to scene;
// vector growth can invalidate references held by other geometry builders.
inline void add_torus(Scene& scene, int s, std::size_t object) {
    const int nu = 4*s, nv = 2*s;
    const std::size_t base = scene.vertices.size();
    for (int i = 0; i < nu; ++i) {
        const double u = 2.0*pi*i/nu;
        for (int j = 0; j < nv; ++j) {
            const double v = 2.0*pi*j/nv;
            const double radius = 0.78 + 0.26*std::cos(v);
            scene.vertices.push_back({
                {radius*std::cos(u), radius*std::sin(u), 0.26*std::sin(v)},
                {std::cos(v)*std::cos(u), std::cos(v)*std::sin(u), std::sin(v)},
                object});
        }
    }
    // Shared seam indices avoid roundoff differences between duplicate endpoints.
    for (int i = 0; i < nu; ++i) {
        for (int j = 0; j < nv; ++j) {
            const std::size_t a = base + i*nv + j;
            const std::size_t b = base + ((i+1)%nu)*nv + j;
            const std::size_t c = base + ((i+1)%nu)*nv + (j+1)%nv;
            const std::size_t d = base + i*nv + (j+1)%nv;
            scene.triangles.push_back({{a,b,c}, object});
            scene.triangles.push_back({{a,c,d}, object});
        }
    }
}

// Appends a cube centered at the local origin, with finite half_size > 0 in
// scene units and object indexing scene.objects. Adds 24 vertices and 12
// triangles. Requires exclusive access to scene, as does add_torus.
inline void add_cube(Scene& scene, double half_size, std::size_t object) {
    const Vec3 axes[3] = {{1,0,0}, {0,1,0}, {0,0,1}};
    const int signs[4][2] = {{-1,-1}, {1,-1}, {1,1}, {-1,1}};
    // Separate face vertices preserve discontinuous normals at sharp edges.
    for (int axis = 0; axis < 3; ++axis) {
        for (int side : {-1, 1}) {
            const Vec3 normal = axes[axis] * side;
            const std::size_t base = scene.vertices.size();
            for (const auto& sign : signs) {
                const Vec3 position = (normal + axes[(axis+1)%3]*sign[0] +
                    axes[(axis+2)%3]*sign[1]) * half_size;
                scene.vertices.push_back({position, normal, object});
            }
            scene.triangles.push_back({{base,base+1,base+2}, object});
            scene.triangles.push_back({{base,base+2,base+3}, object});
        }
    }
}

// Appends a unit-radius sphere with outward smooth normals. Requires s in
// [2,1024] and object indexing scene.objects; writes scene exclusively.
// Adds 4*s*(2*s-1)+2 vertices and 8*s*(2*s-1) triangles without degenerate poles.
inline void add_sphere(Scene& scene, int s, std::size_t object) {
    const int longitude = 4*s, latitude = 2*s;
    const std::size_t top = scene.vertices.size();
    scene.vertices.push_back({{0,1,0}, {0,1,0}, object});
    for (int i = 1; i < latitude; ++i) {
        const double theta = pi*i/latitude;
        for (int j = 0; j < longitude; ++j) {
            const double phi = 2.0*pi*j/longitude;
            const Vec3 p{std::sin(theta)*std::cos(phi), std::cos(theta),
                         std::sin(theta)*std::sin(phi)};
            scene.vertices.push_back({p, p, object});
        }
    }
    const std::size_t bottom = scene.vertices.size();
    scene.vertices.push_back({{0,-1,0}, {0,-1,0}, object});
    // Single pole vertices avoid zero-area triangles and inconsistent seam data.
    for (int j = 0; j < longitude; ++j) {
        const int next = (j+1)%longitude;
        scene.triangles.push_back({{top,top+1+next,top+1+j}, object});
        const std::size_t last_ring = top+1+(latitude-2)*longitude;
        scene.triangles.push_back({{bottom,last_ring+j,last_ring+next}, object});
        for (int i = 0; i < latitude-2; ++i) {
            const std::size_t a = top+1+i*longitude+j;
            const std::size_t b = top+1+i*longitude+next;
            const std::size_t c = b+longitude, d = a+longitude;
            scene.triangles.push_back({{a,b,c}, object});
            scene.triangles.push_back({{a,c,d}, object});
        }
    }
}

// Returns a strictly increasing axis with the specified segment counts between
// four finite increasing boundaries, in scene units. Counts must be positive.
// Pure: callers own the returned vector; no geometry or PRNG state is shared.
inline std::vector<double> terrain_axis(const std::array<double,4>& boundaries,
                                       const std::array<int,3>& segments) {
    std::vector<double> axis{boundaries[0]};
    for (int interval = 0; interval < 3; ++interval) {
        for (int i = 1; i <= segments[interval]; ++i) {
            const double t = static_cast<double>(i)/segments[interval];
            // Exact boundary reuse prevents cracks at changes of density.
            axis.push_back(i == segments[interval] ? boundaries[interval+1] :
                boundaries[interval] + (boundaries[interval+1]-boundaries[interval])*t);
        }
    }
    return axis;
}

// Appends a static plane at ground_y over x=[-12,12], z=[1,30] scene units.
// Requires s in [2,1024] and object identifying an identity transform in scene.
// The dense patch is x=[-4,-1], z=[4,9]. Adds (4*s+13)*(2*s+13) vertices and
// 2*(4*s+12)*(2*s+12) triangles. Requires exclusive access to scene.
inline void add_terrain(Scene& scene, int s, std::size_t object) {
    const auto xs = terrain_axis({-12,-4,-1,12}, {4,4*s,8});
    const auto zs = terrain_axis({1,4,9,30}, {4,2*s,8});
    const std::size_t base = scene.vertices.size();
    // One conforming grid avoids T-junctions between coarse and fine regions.
    for (double z : zs)
        for (double x : xs)
            scene.vertices.push_back({{x,ground_y,z}, {0,1,0}, object});
    for (std::size_t i = 0; i+1 < zs.size(); ++i) {
        for (std::size_t j = 0; j+1 < xs.size(); ++j) {
            const std::size_t a = base+i*xs.size()+j, b = a+1;
            const std::size_t c = b+xs.size(), d = a+xs.size();
            scene.triangles.push_back({{a,c,b}, object});
            scene.triangles.push_back({{a,d,c}, object});
        }
    }
}

// Returns one clustered instance with index >= 0 selecting torus/cube/sphere
// cyclically. Advances rng exclusively; no scene writes. Centers and bounding
// radii guarantee z-radius >= near_plane+1 and clearance above the ground under
// every rotation. The same index/stream prefix gives identical instance data.
inline Object make_instance(int index, Xorshift32& rng) {
    const Shape shapes[] = {Shape::Torus, Shape::Cube, Shape::Sphere};
    const Shape shape = shapes[index%3];
    // Keep draws in explicit sequence so expression evaluation order cannot
    // change the layout. Mesh tessellation never consumes this stream.
    const double cluster = rng.unit();
    const double u = rng.unit();
    const double v = rng.unit();
    const double scale = 0.3 + 0.55*rng.unit();
    const double local_radius = shape == Shape::Torus ? 1.04 :
                                shape == Shape::Cube ? std::sqrt(3.0)*0.6 : 1.0;
    const double radius = scale*local_radius + 1e-9;
    double x, z;
    // Unequal populations and separated ranges leave a central empty corridor.
    if (cluster < 0.70) { x = -3.8 + 2.8*u; z = 4.5 + 3.5*v; }
    else if (cluster < 0.90) { x = 2.0 + 2.5*u; z = 7.0 + 4.0*v; }
    else { x = -7.0 + 3.0*u; z = 13.0 + 7.0*v; }
    z = std::max(z, near_plane + radius + 1.0);
    const double y = ground_y + radius + 0.1 + 1.6*rng.unit();
    const double red = 0.25 + 0.7*rng.unit();
    const double green = 0.25 + 0.7*rng.unit();
    const double blue = 0.25 + 0.7*rng.unit();
    const double angle_x = 0.8*rng.unit();
    const double angle_y = 2.0*pi*rng.unit();
    const double spin = 0.004 + 0.012*rng.unit();
    return {{x,y,z}, {red,green,blue}, angle_x,angle_y,scale,spin,radius,shape};
}

struct SceneCounts { std::size_t vertices, triangles; };

// Returns the exact vertex and triangle totals of generate_scene(subdivisions,
// count) without building anything. Instance types cycle torus/cube/sphere, so
// there are (count+2)/3 tori, (count+1)/3 cubes and count/3 spheres. The formulas
// are those of add_terrain, add_torus, add_cube and add_sphere. Pure.
inline SceneCounts scene_counts(int subdivisions, int count) {
    const std::size_t s = subdivisions, n = count;
    const std::size_t tori = (n+2)/3, cubes = (n+1)/3, spheres = n/3;
    return {(4*s+13)*(2*s+13) + tori*8*s*s + cubes*24 + spheres*(4*s*(2*s-1)+2),
            2*(4*s+12)*(2*s+12) + tori*16*s*s + cubes*12 + spheres*8*s*(2*s-1)};
}

// Returns nonuniform terrain plus exactly count instances (excluding terrain).
// Requires subdivisions in [2,1024], count in [0,10000]. Resets the fixed PRNG
// seed on each call, so increasing count preserves the existing instance prefix.
// Each call owns all mesh data; generated geometry stays beyond the near plane.
inline Scene generate_scene(int subdivisions, int count) {
    Scene scene;
    const SceneCounts expected = scene_counts(subdivisions, count);
    scene.vertices.reserve(expected.vertices);
    scene.triangles.reserve(expected.triangles);
    scene.objects.reserve(static_cast<std::size_t>(count) + 1);
    scene.objects.push_back({{0,0,0}, {0.34,0.43,0.24}, 0,0,1,0,0,Shape::Terrain});
    add_terrain(scene, subdivisions, 0);
    Xorshift32 rng(scene_seed);
    for (int i = 0; i < count; ++i) {
        const Object object = make_instance(i, rng);
        const std::size_t id = scene.objects.size();
        scene.objects.push_back(object);
        if (object.shape == Shape::Torus) add_torus(scene, subdivisions, id);
        else if (object.shape == Shape::Cube) add_cube(scene, 0.6, id);
        else add_sphere(scene, subdivisions, id);
    }
    if (scene.vertices.size() != expected.vertices || scene.triangles.size() != expected.triangles)
        throw std::runtime_error("scene_counts disagrees with the generated mesh");
    return scene;
}

#endif  // PD02_SCENE_H_
