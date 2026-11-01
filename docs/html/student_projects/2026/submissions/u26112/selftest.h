// Small analytic, resource, and reproducibility tests for the serial baseline.
// Tests own their temporary buffers and use memory streams for diagnostic output.
#ifndef PD02_SELFTEST_H_
#define PD02_SELFTEST_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pipeline.h"
#include "reporting.h"
#include "scene.h"

// Raises a test failure with message as diagnostic context when condition fails.
// Uses exceptions so main reports failures consistently; writes no shared output.
inline void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("Self-test failed: " + message);
}

// Compares instance attributes exactly for reproducibility tests, not approximate
// geometric equivalence. Reads a/b only; calls are independent.
inline bool same_object(const Object& a, const Object& b) {
    return a.center == b.center && a.color == b.color && a.angle_x == b.angle_x &&
        a.angle_y == b.angle_y && a.scale == b.scale && a.spin == b.spin &&
        a.radius == b.radius && a.shape == b.shape;
}

// Checks that a is an exact prefix of b, including geometry and instance data.
// Both scenes must use the same subdivision count. Throws on any mismatch;
// only reads the scenes, so completed scenes can be checked independently.
inline void require_scene_prefix(const Scene& a, const Scene& b) {
    require(a.objects.size() <= b.objects.size() && a.vertices.size() <= b.vertices.size()
            && a.triangles.size() <= b.triangles.size(), "scene prefix sizes");
    for (std::size_t i = 0; i < a.objects.size(); ++i)
        require(same_object(a.objects[i], b.objects[i]), "deterministic instance attributes");
    for (std::size_t i = 0; i < a.vertices.size(); ++i)
        require(a.vertices[i].position == b.vertices[i].position &&
                a.vertices[i].normal == b.vertices[i].normal &&
                a.vertices[i].object == b.vertices[i].object, "deterministic mesh vertices");
    for (std::size_t i = 0; i < a.triangles.size(); ++i)
        require(a.triangles[i].vertices == b.triangles[i].vertices &&
                a.triangles[i].object == b.triangles[i].object, "deterministic mesh indices");
}

// Tests the fixed PRNG, scene contracts, and all-angle radius bounds on small
// meshes. Owns all temporary state; throws on failure and writes no files.
inline void test_scene_generation() {
    Xorshift32 known(1);
    require(known.next() == 270369u && known.next() == 67634689u &&
            known.next() == 2647435461u, "xorshift32 reference sequence");
    bool rejected_zero = false;
    try { Xorshift32 invalid(0); }
    catch (const std::runtime_error&) { rejected_zero = true; }
    require(rejected_zero, "zero xorshift seed rejected");
    Xorshift32 rng(scene_seed);
    int dense = 0, secondary = 0, sparse = 0;
    for (int i = 0; i < 300; ++i) {
        const Object object = make_instance(i, rng);
        if (object.center.z < 8.0 && object.center.x < 0.0) ++dense;
        else if (object.center.x > 0.0) ++secondary;
        else ++sparse;
        require(object.center.z-object.radius >= near_plane+1.0 &&
                object.center.y-object.radius > ground_y, "all-angle placement clearance");
    }
    require(dense > secondary+sparse && secondary > 0 && sparse > 0,
            "unequal cluster populations and separate sparse region");

    const Scene scene = generate_scene(4,12);
    require(scene.objects.size() == 13, "instance count excludes terrain");
    require(scene.vertices.size() == 1673 && scene.triangles.size() == 3088,
            "terrain plus four instances of each shape: analytic counts");
    require_scene_prefix(scene, generate_scene(4,12));
    require_scene_prefix(scene, generate_scene(4,13));
    const Scene finer = generate_scene(8,12);
    for (std::size_t i = 0; i < scene.objects.size(); ++i)
        require(same_object(scene.objects[i], finer.objects[i]), "layout independent of tessellation");
    for (const Vertex& vertex : scene.vertices) {
        require(vertex.object < scene.objects.size(), "valid vertex object index");
        const Object& object = scene.objects[vertex.object];
        if (object.shape == Shape::Terrain) {
            require(vertex.position.y == ground_y && vertex.position.z >= 1.0 &&
                    object.spin == 0.0, "static terrain above near plane");
        } else {
            require(std::sqrt(dot(vertex.position,vertex.position))*object.scale <= object.radius,
                    "bounding sphere contains every scaled vertex");
            require(object.center.z-object.radius > near_plane, "rotation-invariant near bound");
        }
        if (object.shape == Shape::Sphere)
            require(std::abs(dot(vertex.position,vertex.position)-1.0) < 1e-12 &&
                    vertex.position == vertex.normal, "unit sphere with radial normals");
    }
    double area_sum = 0.0, min_area = std::numeric_limits<double>::infinity(), max_area = 0.0;
    for (const Triangle& t : scene.triangles) {
        for (std::size_t index : t.vertices)
            require(index < scene.vertices.size() && scene.vertices[index].object == t.object,
                    "triangle indices and object ownership");
        const Vec3 a = scene.vertices[t.vertices[0]].position;
        const Vec3 ab = scene.vertices[t.vertices[1]].position + a*(-1.0);
        const Vec3 ac = scene.vertices[t.vertices[2]].position + a*(-1.0);
        const Vec3 cross{ab.y*ac.z-ab.z*ac.y, ab.z*ac.x-ab.x*ac.z, ab.x*ac.y-ab.y*ac.x};
        require(dot(cross,cross) > 0.0, "nondegenerate generated triangles including sphere poles");
        if (t.object == 0) {
            const double area = 0.5*std::abs(cross.y);
            area_sum += area;
            min_area = std::min(min_area,area);
            max_area = std::max(max_area,area);
        }
    }
    require(std::abs(area_sum-24.0*29.0) < 1e-9 && max_area > 16.0*min_area,
            "complete plane area and deliberately nonuniform tessellation");

    Scene invalid = generate_scene(2,1);
    invalid.objects[1].center.z = 0.0;
    std::vector<ScreenVertex> projected(invalid.vertices.size());
    bool rejected_near = false;
    try { transform_vertices(invalid,0,96,72,projected); }
    catch (const std::runtime_error&) { rejected_near = true; }
    require(rejected_near, "near-plane runtime guard retained");
}

// Checks terrain-only visibility/depth against analytic ray-plane intersections.
// Uses local buffers and a tiny image; throws on gaps or inconsistent depth.
inline void test_terrain_projection() {
    const Scene scene = generate_scene(4,0);
    Framebuffer fb(96,72);
    std::vector<ScreenVertex> projected(scene.vertices.size());
    transform_vertices(scene,0,fb.width,fb.height,projected);
    clear_framebuffer(fb);
    render_frame(scene,projected,fb);
    const double focal = 0.9*std::min(fb.width,fb.height);
    for (int y = 0; y < fb.height; ++y) {
        for (int x = 0; x < fb.width; ++x) {
            const std::size_t index = static_cast<std::size_t>(y)*fb.width+x;
            bool expected = false;
            double z = 0.0;
            if (y+0.5 > fb.height*0.5) {
                z = -ground_y*focal/(y+0.5-fb.height*0.5);
                const double world_x = (x+0.5-fb.width*0.5)*z/focal;
                expected = z > 1.0 && z < 30.0 && world_x > -12.0 && world_x < 12.0;
            }
            require((fb.owner[index] != no_triangle) == expected, "terrain coverage without cracks");
            if (expected)
                require(std::abs(fb.inv_depth[index]-1.0/z) < 1e-12, "analytic terrain depth");
        }
    }
}

// Checks the published CSV field order, precision, locale, and hash spelling
// using memory streams; owns all output and does not create files.
inline void test_csv_format() {
    Options o;
    o.width = 65; o.height = 33; o.subdivisions = 4; o.objects = 9; o.frames = 1;
    RunResult r;
    r.records = {{0,1.0,2.0,3.0,10}};
    r.clear_s = 1; r.transform_s = 2; r.raster_s = 3;
    r.vertices = 1407; r.triangles = 2596; r.covered = 1007;
    r.last_hash = 10; r.sequence_hash = 11;
    const auto fields = [](const std::string& row) {
        std::istringstream in(row);
        std::vector<std::string> values;
        std::string value;
        while (std::getline(in,value,',')) values.push_back(value);
        return values;
    };
    std::ostringstream timings;
    write_timings_csv(timings,o,r);
    std::istringstream in(timings.str());
    std::string header, row;
    std::getline(in,header); std::getline(in,row);
    require(header == "variant,width,height,subdiv,objects,frames,frame,clear_s,transform_s,raster_s,total_s,frame_hash",
            "fixed timings CSV schema");
    const auto f = fields(row);
    require(f.size() == 12 && f[0] == "baseline" && f[1] == "65" && f[6] == "0" &&
            f[7] == "1.00000000000000000e+00" && f[10] == "6.00000000000000000e+00" &&
            f[11] == "000000000000000a", "timings values and numeric format");
    require(csv_field("base,\"quoted\"") == "\"base,\"\"quoted\"\"\"", "CSV escaping");
    std::ostringstream summary;
    write_summary_csv(summary,o,r,true);
    write_summary_csv(summary,o,r,false);
    std::istringstream sin(summary.str());
    std::getline(sin,header); std::getline(sin,row);
    require(header == "variant,threads,width,height,subdiv,objects,frames,vertices,triangles_per_frame,covered_pixels_last,clear_total_s,transform_total_s,raster_total_s,render_seconds,seconds_per_frame,triangles_per_second,checksum_rgb,checksum_sequence",
            "fixed summary CSV schema");
    const auto s = fields(row);
    require(s.size() == 18 && s[1] == "1" && s[7] == "1407" && s[8] == "2596" &&
            s[9] == "1007" && std::stod(s[13]) == 6.0 &&
            std::abs(std::stod(s[15])-2596.0/6.0) < 1e-12 &&
            s[16] == "000000000000000a" && s[17] == "000000000000000b",
            "summary units and hash columns");
    std::string appended;
    require(static_cast<bool>(std::getline(sin,appended)) && appended == row &&
            !std::getline(sin,appended), "one header for appended rows");
}

// Tests observable options with in-memory frame output, including unsaved frame
// hashes. Uses the production frame loop and serializers without filesystem I/O.
inline void test_observability() {
    Options o;
    o.width = 65; o.height = 33; o.subdivisions = 4; o.objects = 9; o.frames = 4;
    const Scene scene = generate_scene(o.subdivisions,o.objects);
    std::vector<ScreenVertex> projected(scene.vertices.size());
    Framebuffer fb(o.width,o.height);
    const RunResult plain = run_frames(o,scene,projected,fb);
    const auto pixels = fb.rgb;
    o.variant = "instrumented,label"; o.threads = 8;
    o.timings = "memory.csv"; o.summary = "summary.csv"; o.work_map = "work.csv";
    o.save_frames = "memory"; o.save_every = 2;
    std::vector<std::string> names;
    std::vector<std::uint64_t> saved;
    const RunResult measured = run_frames(o,scene,projected,fb,
        [&](const std::string& name, const Framebuffer& image) {
            names.push_back(std::filesystem::path(name).filename().string());
            saved.push_back(image_checksum(image));
        });
    require(names == std::vector<std::string>{"frame_00000.ppm","frame_00002.ppm","frame_00003.ppm"},
            "save every K plus last frame exactly once");
    require(saved == std::vector<std::uint64_t>{plain.records[0].hash,plain.records[2].hash,plain.records[3].hash},
            "saved frames match normal rendering");
    require(plain.sequence_hash == measured.sequence_hash && plain.last_hash == measured.last_hash
            && pixels == fb.rgb, "observability options preserve checksums and RGB");
    for (std::size_t i = 0; i < plain.records.size(); ++i)
        require(plain.records[i].hash == measured.records[i].hash, "all per-frame hashes unchanged");
    std::ostringstream out;
    write_timings_csv(out,o,measured);
    write_summary_csv(out,o,measured,true);
    const WorkMap work = collect_work_map(scene,projected,o.width,o.height,measured.last_hash);
    write_work_csv(out,work);
    require(pixels == fb.rgb, "diagnostics preserve final framebuffer");

    // Two full-screen layers expose pre-depth counting, deduplication, and
    // partial tiles; the hidden layer contributes just as many fragments.
    WorkMap quad(65,33);
    clear_framebuffer(fb);
    for (int layer = 0; layer < 2; ++layer) {
        const double z = layer == 0 ? 0.5 : 0.25;
        const ScreenVertex a{0,0,z,{0,0,-z}}, b{65,0,z,{0,0,-z}};
        const ScreenVertex c{65,33,z,{0,0,-z}}, d{0,33,z,{0,0,-z}};
        rasterize_triangle_impl<true>(a,b,c,{1,0,0},2*layer,fb,&quad);
        rasterize_triangle_impl<true>(a,c,d,{0,1,0},2*layer+1,fb,&quad);
    }
    const int ts = WorkMap::tile_size;
    for (int y = 0; y < quad.rows; ++y)
        for (int x = 0; x < quad.columns; ++x) {
            const auto& tile = quad.tiles[y*quad.columns+x];
            require(tile.fragments == static_cast<std::uint64_t>(2*std::min(ts,65-ts*x)*std::min(ts,33-ts*y))
                    && tile.triangles >= 2 && tile.triangles <= 4, "tile work includes occluded samples");
        }
    require(quad.tiles[0].triangles == 4 && quad.tiles.back().triangles == 2,
            "triangle counted once per tile");
    std::ostringstream work_csv;
    write_work_csv(work_csv,quad);
    require(work_csv.str().rfind("tile_x,tile_y,fragments,triangles\n0,0," +
        std::to_string(2*ts*ts) + ",4\n",0) == 0,
            "work-map CSV schema and tile order");
}

// Checks the analytic mesh counts against real generation, and that the memory
// estimate accepts the benchmark size but refuses the largest CLI combination.
inline void test_resource_estimates() {
    const std::vector<std::pair<int,int>> cases{{2,0},{2,1},{2,2},{3,3},{4,12},{5,7}};
    for (const auto& [s, n] : cases) {
        const Scene scene = generate_scene(s,n);
        const SceneCounts counts = scene_counts(s,n);
        require(counts.vertices == scene.vertices.size() &&
                counts.triangles == scene.triangles.size(), "analytic mesh counts match generation");
    }
    const SceneCounts benchmark = scene_counts(300,150);
    require(benchmark.vertices == 72684869 && benchmark.triangles == 145364088,
            "mesh counts of the S=300, N=150 benchmark");
    Options measured;
    measured.width = 3840; measured.height = 2160; measured.subdivisions = 300;
    measured.objects = 150; measured.frames = 20;
    const double gib = check_memory_limit(measured);
    require(gib > 8.0 && gib < 16.0, "memory estimate of the S=300 benchmark is plausible");
    Options with_map = measured; with_map.work_map = "work.csv";
    require(estimate_memory_bytes(with_map) > estimate_memory_bytes(measured),
            "work-map pass needs extra memory");
    Options huge; huge.subdivisions = 1024; huge.objects = 10000;
    bool refused = false;
    try { check_memory_limit(huge); }
    catch (const std::runtime_error&) { refused = true; }
    require(refused, "memory guard refuses the largest CLI combination");
}

// Checks that unusable output destinations are rejected before rendering.
// Only inspects filesystem metadata: no file or directory is created.
inline void test_output_validation() {
    const auto rejected = [](const Options& options) {
        try { validate_outputs(options); }
        catch (const std::runtime_error&) { return true; }
        return false;
    };
    const Options valid;
    require(!rejected(valid), "default output in the current directory is accepted");
    const std::string missing = "no_such_directory_for_self_test/file";
    Options bad = valid; bad.output = missing + ".ppm";
    require(rejected(bad), "missing directory of --output rejected");
    bad = valid; bad.timings = missing + ".csv";
    require(rejected(bad), "missing directory of --timings rejected");
    bad = valid; bad.summary = missing + ".csv";
    require(rejected(bad), "missing directory of --summary rejected");
    bad = valid; bad.work_map = missing + ".csv";
    require(rejected(bad), "missing directory of --work-map rejected");
    bad = valid; bad.output = ".";
    require(rejected(bad), "a directory is not a valid output file");
    bad = valid; bad.timings = valid.output;
    require(rejected(bad), "output files must have distinct paths");
}

// Checks coverage, interpolation, and visibility against small known cases.
// Uses local scenes and framebuffers, throws on failure, and writes a success
// message to shared std::cout. No rendering setup or output file is required.
inline void run_self_tests() {
    test_csv_format();
    test_resource_estimates();
    test_output_validation();
    test_observability();
    test_scene_generation();
    test_terrain_projection();
    const Vec3 red{1,0,0}, green{0,1,0};
    const ScreenVertex a{0,0,0.5,{0,0,-0.5}}, b{4,0,0.5,{0,0,-0.5}};
    const ScreenVertex c{4,4,0.5,{0,0,-0.5}}, d{0,4,0.5,{0,0,-0.5}};
    Framebuffer first(4,4), second(4,4), combined(4,4);
    clear_framebuffer(first); clear_framebuffer(second); clear_framebuffer(combined);
    rasterize_triangle(a,b,c,red,0,first);
    rasterize_triangle(a,c,d,green,1,second);
    for (std::size_t i = 0; i < first.owner.size(); ++i)
        require((first.owner[i] != no_triangle) != (second.owner[i] != no_triangle),
                "shared diagonal must have exactly one owner, without holes");
    rasterize_triangle(a,b,c,red,0,combined);
    rasterize_triangle(a,c,d,green,1,combined);
    const auto expected = combined.rgb;
    clear_framebuffer(combined);
    rasterize_triangle(d,c,a,green,1,combined);
    rasterize_triangle(c,b,a,red,0,combined);
    require(combined.rgb == expected, "winding and submission order");

    // At pixel (0,0), weights (0.75, 0.125, 0.125) and inverse depths are binary
    // fractions, so an exact comparison can isolate interpolation mistakes.
    const ScreenVertex p{0,0,1.0,{1,0,0}}, q{4,0,0.5,{0,0.5,0}};
    const ScreenVertex r{0,4,0.25,{0,0,-0.25}};
    clear_framebuffer(combined);
    rasterize_triangle(p,q,r,red,5,combined);
    require(combined.inv_depth[0] == 0.84375, "perspective inverse depth");
    const auto expected_color = shade({0.75/0.84375,0.0625/0.84375,-0.03125/0.84375},red);
    for (int ch = 0; ch < 3; ++ch)
        require(combined.rgb[ch] == expected_color[ch], "perspective normal interpolation");
    rasterize_triangle(p,q,r,green,2,combined);
    require(combined.owner[0] == 2, "equal-depth tie: lower ID wins");
    rasterize_triangle(p,q,r,red,5,combined);
    require(combined.owner[0] == 2, "equal-depth tie independent of order");
    rasterize_triangle(a,b,d,red,0,combined);
    require(combined.owner[0] == 2, "farther fragment must not overwrite");
    clear_framebuffer(combined);
    rasterize_triangle(a,b,d,red,0,combined);
    rasterize_triangle(p,q,r,green,2,combined);
    require(combined.owner[0] == 2, "closer fragment must overwrite");
    const auto before = combined.rgb;
    rasterize_triangle(a,a,a,red,0,combined);
    rasterize_triangle({-8,-8,1,{0,0,-1}}, {-4,-8,1,{0,0,-1}},
                       {-4,-4,1,{0,0,-1}},red,0,combined);
    require(combined.rgb == before, "degenerate and off-screen triangles");
    require(image_checksum(Framebuffer(0,0)) == fnv_offset, "empty FNV-1a hash");

    const Scene scene = generate_scene(4,12);
    std::vector<ScreenVertex> projected(scene.vertices.size());
    Framebuffer forward(96,72), backward(96,72);
    for (int frame : {0,73,157,999999}) {
        transform_vertices(scene,frame,96,72,projected);
        clear_framebuffer(forward); clear_framebuffer(backward);
        render_frame(scene,projected,forward);
        for (std::size_t i = scene.triangles.size(); i > 0; --i) {
            const Triangle& t = scene.triangles[i-1];
            rasterize_triangle(projected[t.vertices[0]], projected[t.vertices[1]],
                projected[t.vertices[2]],scene.objects[t.object].color,i-1,backward);
        }
        require(forward.rgb == backward.rgb && forward.owner == backward.owner &&
                forward.inv_depth == backward.inv_depth, "scene in reverse triangle order");
        const auto visible = std::count_if(forward.owner.begin(),forward.owner.end(),
            [](std::size_t id) { return id != no_triangle; });
        require(visible > 0 && static_cast<std::size_t>(visible) < forward.owner.size(),
                "visible scene with remaining sky");
        bool visible_instance = false;
        for (std::size_t id : forward.owner)
            if (id != no_triangle && scene.triangles[id].object != 0) visible_instance = true;
        require(visible_instance, "instances visible above terrain");
    }
    std::cout << "Self-tests passed: PRNG, scene counts/prefix, sphere, nonuniform terrain, "
                 "near plane, coverage, depth, interpolation, ties, scene order, CSV, "
                 "frame selection, work map, instrumentation invariance, output paths, memory estimate.\n";
}

#endif  // PD02_SELFTEST_H_
