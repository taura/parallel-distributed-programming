// Stable CSV and output contracts for the serial rasterizer and future variants.
// Serialization is separate from rendering and uses only the C++17 library.
#ifndef PD02_REPORTING_H_
#define PD02_REPORTING_H_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Options {
    int width = 960, height = 720, subdivisions = 64, frames = 60, objects = 24;
    int threads = 1, save_every = 1;
    std::string output = "image.ppm", variant = "baseline";
    std::string timings, summary, save_frames, work_map;
    bool self_test = false, help = false;
};

struct FrameRecord {
    int frame;
    double clear_s, transform_s, raster_s;
    std::uint64_t hash;
    double total() const { return clear_s + transform_s + raster_s; }
};

struct RunResult {
    std::vector<FrameRecord> records;
    std::size_t vertices = 0, triangles = 0, covered = 0;
    double clear_s = 0.0, transform_s = 0.0, raster_s = 0.0;
    std::uint64_t last_hash = 0, sequence_hash = 0;
    double total() const { return clear_s + transform_s + raster_s; }
};

inline constexpr char timings_header[] =
    "variant,width,height,subdiv,objects,frames,frame,clear_s,transform_s,raster_s,total_s,frame_hash";
inline constexpr char summary_header[] =
    "variant,threads,width,height,subdiv,objects,frames,vertices,triangles_per_frame,"
    "covered_pixels_last,clear_total_s,transform_total_s,raster_total_s,render_seconds,"
    "seconds_per_frame,triangles_per_second,checksum_rgb,checksum_sequence";
inline constexpr char work_header[] = "tile_x,tile_y,fragments,triangles";

// Returns exactly 16 lowercase hexadecimal digits; pure and locale independent.
inline std::string hash_hex(std::uint64_t hash) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::hex << std::setfill('0') << std::setw(16) << hash;
    return out.str();
}

// Quotes a CSV field containing commas or quotes. Embedded quotes are doubled.
// CLI labels exclude line breaks; this pure helper also supports general text.
inline std::string csv_field(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string escaped = "\"";
    for (char c : value) {
        if (c == '"') escaped += '"';
        escaped += c;
    }
    return escaped + '"';
}

// Sets the stable numeric representation on out, which the caller owns
// exclusively. Scientific notation with 17 decimals preserves double values.
inline void configure_csv(std::ostream& out) {
    out.imbue(std::locale::classic());
    out << std::dec << std::scientific << std::setprecision(17);
}

// Writes one header and one row per zero-based frame. Requires records from one
// completed run with parameters o; seconds exclude output and hashing costs.
// Reads inputs only; writes out exclusively and must run outside timed phases.
inline void write_timings_csv(std::ostream& out, const Options& o, const RunResult& r) {
    configure_csv(out);
    out << timings_header << '\n';
    for (const auto& f : r.records)
        out << csv_field(o.variant) << ',' << o.width << ',' << o.height << ','
            << o.subdivisions << ',' << o.objects << ',' << o.frames << ',' << f.frame << ','
            << f.clear_s << ',' << f.transform_s << ',' << f.raster_s << ',' << f.total()
            << ',' << hash_hex(f.hash) << '\n';
}

// Writes one completed-run row, optionally preceded by its header. Throughput
// counts submitted triangles over all timed frames, divided by rendering time.
// threads is metadata, not a request for parallel execution. Own out exclusively.
inline void write_summary_csv(std::ostream& out, const Options& o, const RunResult& r,
                              bool header) {
    configure_csv(out);
    if (header) out << summary_header << '\n';
    const double throughput = r.total() > 0.0 ?
        static_cast<double>(r.triangles)*o.frames/r.total() : 0.0;
    out << csv_field(o.variant) << ',' << o.threads << ',' << o.width << ',' << o.height
        << ',' << o.subdivisions << ',' << o.objects << ',' << o.frames << ','
        << r.vertices << ',' << r.triangles << ',' << r.covered << ',' << r.clear_s
        << ',' << r.transform_s << ',' << r.raster_s << ',' << r.total() << ','
        << r.total()/o.frames << ',' << throughput << ',' << hash_hex(r.last_hash)
        << ',' << hash_hex(r.sequence_hash) << '\n';
}

// Returns the relative filename for a nonnegative frame index. Five digits is
// the minimum width; indices >= 100000 retain all their digits. Pure.
inline std::string frame_filename(int frame) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "frame_" << std::setfill('0') << std::setw(5) << frame << ".ppm";
    return out.str();
}

// Selects frame multiples of save_every and the last frame, exactly once.
// Requires 0 <= frame < o.frames and save_every >= 1. Pure.
inline bool should_save_frame(const Options& o, int frame) {
    return !o.save_frames.empty() && (frame % o.save_every == 0 || frame == o.frames-1);
}

// Validates output destinations before rendering. Frame directories must exist;
// files must have distinct paths to avoid overwriting a CSV with image data.
// Each output file needs an existing parent directory and must not be a directory.
// Reads filesystem metadata only; does not create directories or touch files.
inline void validate_outputs(const Options& o) {
    namespace fs = std::filesystem;
    if (!o.save_frames.empty() && !fs::is_directory(o.save_frames))
        throw std::runtime_error("--save-frames directory must already exist: " + o.save_frames);
    const std::vector<std::string> files{o.output, o.timings, o.summary, o.work_map};
    for (const std::string& file : files) {
        if (file.empty()) continue;
        fs::path parent = fs::path(file).parent_path();
        if (parent.empty()) parent = ".";
        if (!fs::is_directory(parent))
            throw std::runtime_error("Directory of output file does not exist: " + file);
        if (fs::is_directory(file))
            throw std::runtime_error("Output path is a directory: " + file);
    }
    for (std::size_t i = 0; i < files.size(); ++i) {
        if (files[i].empty()) continue;
        for (std::size_t j = 0; j < i; ++j) {
            if (files[j].empty()) continue;
            if (fs::weakly_canonical(files[i]) == fs::weakly_canonical(files[j]) ||
                (fs::exists(files[i]) && fs::exists(files[j]) && fs::equivalent(files[i],files[j])))
                throw std::runtime_error("Output paths must be distinct: " + files[i]);
        }
        if (!o.save_frames.empty() && fs::weakly_canonical(fs::path(files[i]).parent_path()) ==
            fs::weakly_canonical(o.save_frames) &&
            fs::path(files[i]).filename().string().rfind("frame_",0) == 0)
            throw std::runtime_error("Output path conflicts with saved frames: " + files[i]);
    }
}

// Checks an existing summary's schema and complete final line. Returns true for
// a missing/empty file, which needs a header. Requires a single external writer.
inline bool summary_needs_header(const std::string& filename) {
    namespace fs = std::filesystem;
    if (!fs::exists(filename)) return true;
    if (!fs::is_regular_file(filename)) throw std::runtime_error("Summary must be a regular file");
    if (fs::file_size(filename) == 0) return true;
    std::ifstream in(filename, std::ios::binary);
    std::string header;
    if (!std::getline(in,header) || header != summary_header)
        throw std::runtime_error("Summary CSV header mismatch: " + filename);
    in.seekg(-1,std::ios::end);
    if (in.get() != '\n') throw std::runtime_error("Incomplete summary CSV: " + filename);
    return false;
}

// Flushes/closes an owned output file and reports delayed write failures.
inline void close_checked(std::ofstream& out, const std::string& filename) {
    out.close();
    if (!out) throw std::runtime_error("Cannot write output: " + filename);
}

// Writes the per-frame CSV of a completed run, replacing any existing file.
// Does nothing when --timings is unset.
inline void write_timings_file(const Options& o, const RunResult& r) {
    if (o.timings.empty()) return;
    std::ofstream out(o.timings, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open timings: " + o.timings);
    write_timings_csv(out,o,r);
    close_checked(out,o.timings);
}

// Appends one summary row for a completed run, with a header for a new file.
// Does nothing when --summary is unset. Requires a single external writer.
inline void write_summary_file(const Options& o, const RunResult& r) {
    if (o.summary.empty()) return;
    const bool header = summary_needs_header(o.summary);
    std::ofstream out(o.summary, std::ios::binary | std::ios::app);
    if (!out) throw std::runtime_error("Cannot open summary: " + o.summary);
    write_summary_csv(out,o,r,header);
    close_checked(out,o.summary);
}

// Writes both report files, timings first.
inline void write_report_files(const Options& o, const RunResult& r) {
    write_timings_file(o,r);
    write_summary_file(o,r);
}

#endif  // PD02_REPORTING_H_
