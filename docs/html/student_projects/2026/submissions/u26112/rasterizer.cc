// Command-line entry point for the deterministic serial C++17 rasterizer.
// Parses options and coordinates scene generation, rendering, tests, and output.
#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pipeline.h"
#include "reporting.h"
#include "scene.h"
#include "selftest.h"

// Returns unsigned decimal text as an int in [minimum,maximum]. Requires
// 0 <= minimum <= maximum; signs and whitespace are not accepted.
// Throws on malformed or out-of-range input. Has no shared mutable state.
int bounded_integer(const std::string& text, int minimum, int maximum) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Expected an unsigned decimal integer: " + text);
    const unsigned long long value = std::stoull(text);
    if (value < static_cast<unsigned long long>(minimum) ||
        value > static_cast<unsigned long long>(maximum))
        throw std::runtime_error("Integer out of range: " + text);
    return static_cast<int>(value);
}

// Returns validated options from argc/argv using the standard main convention.
// Dimensions are pixels in [1,32768], subdivisions in [2,1024], frames in
// [1,1000000], and instance counts in [0,10000] excluding terrain. Throws on
// invalid options; reads argv without modifying it.
Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") { o.help = true; continue; }
        if (key == "--self-test") { o.self_test = true; continue; }
        if (key != "--width" && key != "--height" && key != "--subdiv" &&
            key != "--frames" && key != "--objects" && key != "--output" &&
            key != "--timings" && key != "--summary" && key != "--variant" &&
            key != "--threads" && key != "--save-frames" && key != "--save-every" &&
            key != "--work-map")
            throw std::runtime_error("Unknown option: " + key);
        if (++i == argc) throw std::runtime_error("Missing value for " + key);
        const std::string value = argv[i];
        if (value.empty()) throw std::runtime_error("Empty value for " + key);
        if (key == "--width") o.width = bounded_integer(value, 1, 32768);
        if (key == "--height") o.height = bounded_integer(value, 1, 32768);
        if (key == "--subdiv") o.subdivisions = bounded_integer(value, 2, 1024);
        if (key == "--frames") o.frames = bounded_integer(value, 1, 1000000);
        if (key == "--objects") o.objects = bounded_integer(value, 0, 10000);
        if (key == "--output") o.output = value;
        if (key == "--threads") o.threads = bounded_integer(value, 1, 1000000);
        if (key == "--save-every") o.save_every = bounded_integer(value, 1, 1000000);
        if (key == "--variant") o.variant = value;
        if (key == "--timings") o.timings = value;
        if (key == "--summary") o.summary = value;
        if (key == "--save-frames") o.save_frames = value;
        if (key == "--work-map") o.work_map = value;
    }
    if (o.output.empty()) throw std::runtime_error("Empty output filename");
    for (unsigned char c : o.variant)
        if (c < 32 || c == 127) throw std::runtime_error("Variant must be a nonempty single-line label");
    return o;
}

// Prints the CLI contract and resource limits to shared stdout. No rendering or
// filesystem access occurs; callers must serialize access to the output stream.
void print_help() {
    std::cout << "Usage: ./rasterizer [--width W] [--height H] [--subdiv S]\n"
                 "                    [--objects N] [--frames F] [--output image.ppm]\n"
                 "                    [--timings FILE.csv] [--summary FILE.csv]\n"
                 "                    [--variant NAME] [--threads N]\n"
                 "                    [--save-frames DIR] [--save-every K] [--work-map FILE.csv]\n"
                 "                    [--self-test] [--help]\n"
                 "Defaults: W=960 H=720 S=64 N=24 F=60.\n"
                 "Ranges: W,H=1..32768 S=2..1024 N=0..10000 F=1..1000000.\n"
                 "N counts instances, excluding the static nonuniform terrain.\n"
                 "Types cycle torus/cube/sphere; N=0 renders terrain only.\n"
                 "Layout uses handwritten xorshift32 with fixed seed 0x12345678.\n"
                 "S controls torus/sphere resolution and the dense terrain patch.\n"
                 "Geometry and memory grow roughly as (N+1)*S*S.\n"
                 "Instances rotate deterministically and stay beyond z=0.1.\n"
                 "Times: clear_seconds, transform_seconds, raster_seconds (with shading).\n"
                 "render_seconds is their sum; setup, checksums and file I/O are excluded.\n"
                 "estimated_frames_180s suggests a frame count from the measured mean\n"
                 "(clamped to 1..1000000); it is an estimate, not a time limit.\n"
                 "--variant defaults to baseline; --threads is 1..1000000, default 1 (LABEL ONLY).\n"
                 "--timings overwrites a per-frame CSV; --summary appends one run row.\n"
                 "CSV schema is fixed, decimal point '.', hashes 16 lowercase hex digits.\n"
                 "Use one writer per summary file; incompatible headers are rejected.\n"
                 "--save-frames requires an existing directory. K defaults to 1 (1..1000000).\n"
                 "Saves frames divisible by K and the last as frame_00000.ppm, etc.\n"
                 "Use a fresh frame directory; old files are not removed.\n"
                 "--work-map overwrites a 32x32 tile CSV from an untimed last-frame replay.\n"
                 "Fragments count coverage before depth; triangles count distinct IDs\n"
                 "with at least one covered sample in the tile (not bounding boxes).\n"
                 "All frame hashes contribute even when only some images are saved.\n"
                 "--output always saves the last image as binary PPM P6.\n"
                 "estimated_memory_gib is printed before the scene is generated; runs\n"
                 "estimated above " << memory_limit_gib << " GiB are refused.\n";
}

// Runs the CLI with the standard argc/argv contract. Returns 0 on success or 1
// on a reported exception. Owns render buffers and writes the selected output
// file plus shared stdout/stderr; render timings exclude checksums and file I/O.
int main(int argc, char** argv) {
    try {
        const Options o = parse_options(argc, argv);
        if (o.help) { print_help(); return 0; }
        if (o.self_test) { run_self_tests(); return 0; }
        validate_outputs(o);
        if (!o.summary.empty()) summary_needs_header(o.summary);
        // Refuse impossible sizes now, and flush the estimate so it is in the log
        // even if the job is killed for running out of memory later.
        const double estimated_gib = check_memory_limit(o);
        std::cout << "estimated_memory_gib=" << std::fixed << std::setprecision(2)
                  << estimated_gib << std::defaultfloat << std::setprecision(6) << std::endl;
        const Scene scene = generate_scene(o.subdivisions, o.objects);
        std::vector<ScreenVertex> vertices(scene.vertices.size());
        Framebuffer fb(o.width,o.height);
        const RunResult r = run_frames(o,scene,vertices,fb);
        write_timings_file(o,r);
        write_ppm(o.output,fb);
        if (!o.work_map.empty()) {
            const WorkMap work = collect_work_map(scene,vertices,o.width,o.height,r.last_hash);
            std::ofstream out(o.work_map,std::ios::binary);
            if (!out) throw std::runtime_error("Cannot open work map: " + o.work_map);
            write_work_csv(out,work);
            close_checked(out,o.work_map);
        }
        write_summary_file(o,r);
        const double render_seconds = r.total();
        std::cout << "resolution=" << o.width << 'x' << o.height
                  << " subdivisions=" << o.subdivisions << " objects=" << o.objects
                  << " frames=" << o.frames
                  << " vertices=" << scene.vertices.size()
                  << " triangles_per_frame=" << scene.triangles.size() << '\n'
                  << "scene=clustered_terrain_v2 seed=" << scene_seed
                  << " torus_instances=" << (o.objects+2)/3
                  << " cube_instances=" << (o.objects+1)/3
                  << " sphere_instances=" << o.objects/3 << '\n'
                  << std::fixed << std::setprecision(6)
                  << "clear_seconds=" << r.clear_s << '\n'
                  << "transform_seconds=" << r.transform_s << '\n'
                  << "raster_seconds=" << r.raster_s << '\n'
                  << "render_seconds=" << render_seconds << '\n'
                  << "seconds_per_frame=" << render_seconds/o.frames << '\n'
                  << "covered_pixels_last=" << r.covered << '/' << fb.owner.size() << '\n'
                  << "checksum_rgb_fnv1a64=" << std::hex << std::setfill('0') << std::setw(16)
                  << r.last_hash << '\n'
                  << "checksum_sequence_fnv1a64=" << std::setw(16) << r.sequence_hash << std::dec
                  << "\noutput=" << o.output << '\n';
        if (render_seconds > 0.0) {
            // Keep the workload fixed while calibrating duration; timer readings
            // never influence geometry, animation, frame counts, or checksums.
            const double estimated_frames = std::ceil(180.0*o.frames/render_seconds);
            std::cout << std::setprecision(0) << "estimated_frames_180s="
                      << std::clamp(estimated_frames, 1.0, 1000000.0) << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
