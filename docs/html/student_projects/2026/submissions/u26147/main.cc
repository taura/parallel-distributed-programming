// main.cc : command line parsing, timing, output
//
// usage: ./pt [-w width] [-h height] [-s spp] [-d max_depth]
//             [-S scene_id] [-D detail] [-o out.ppm] [-p out.pfm] [-r reference.pfm]
#include "render.h"
#include "image.h"
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void usage(const char* prog) {
  fprintf(stderr,
          "usage: %s [options]\n"
          "  -w W     image width (default 256)\n"
          "  -h H     image height (default 256)\n"
          "  -s N     samples per pixel (default 16)\n"
          "  -d N     max bounces (default 8)\n"
          "  -S ID    scene: 0 Cornell box + spheres, 1 Cornell box + meshes, 2 furnace test (default 0)\n"
          "  -D N     mesh detail, 20*4^N triangles per object (default 3)\n"
          "  -o FILE  output PPM (default image.ppm)\n"
          "  -p FILE  also write the linear float image as PFM\n"
          "  -r FILE  reference PFM; print RMSE against it\n",
          prog);
  exit(1);
}

int main(int argc, char** argv) {
  RenderParams p;
  int scene_id = 0, detail = 3;
  std::string out = "image.ppm", pfm_out, ref;

  for (int i = 1; i < argc; i += 2) {
    if (i + 1 >= argc) usage(argv[0]);
    const char* opt = argv[i];
    const char* val = argv[i + 1];
    if (!strcmp(opt, "-w")) p.width = atoi(val);
    else if (!strcmp(opt, "-h")) p.height = atoi(val);
    else if (!strcmp(opt, "-s")) p.spp = atoi(val);
    else if (!strcmp(opt, "-d")) p.max_depth = atoi(val);
    else if (!strcmp(opt, "-S")) scene_id = atoi(val);
    else if (!strcmp(opt, "-D")) detail = atoi(val);
    else if (!strcmp(opt, "-o")) out = val;
    else if (!strcmp(opt, "-p")) pfm_out = val;
    else if (!strcmp(opt, "-r")) ref = val;
    else usage(argv[0]);
  }
  if (p.width <= 0 || p.height <= 0 || p.spp <= 0 || p.max_depth <= 0 || detail < 0 || detail > 8) usage(argv[0]);

  using clk = std::chrono::steady_clock;
  auto t0 = clk::now();
  Scene scene = build_scene(scene_id, detail);
  auto t1 = clk::now();
  uint64_t rays = 0;
  std::vector<Vec3> img = render(scene, p, rays);
  auto t2 = clk::now();

  double build_s = std::chrono::duration<double>(t1 - t0).count();
  double render_s = std::chrono::duration<double>(t2 - t1).count();
  printf("scene=%d detail=%d triangles=%zu spheres=%zu bvh_nodes=%zu bvh_depth=%d\n", scene_id, detail,
         scene.triangles.size(), scene.spheres.size(), scene.bvh.nodes.size(), scene.bvh.depth());
  printf("size=%dx%d spp=%d max_depth=%d\n", p.width, p.height, p.spp, p.max_depth);
  printf("build %.3f s, render %.3f s\n", build_s, render_s);
  printf("rays=%" PRIu64 " (%.2f per sample), %.3f Mrays/s\n", rays,
         (double)rays / ((double)p.width * p.height * p.spp), rays / render_s * 1e-6);

  // verification info
  Vec3 mean = mean_color(img);
  printf("mean color = (%.6f, %.6f, %.6f)\n", mean.x, mean.y, mean.z);
  printf("checksum = %016" PRIx64 "\n", image_checksum(img));
  if (scene_id == 2) printf("furnace test: max |pixel - 1| = %.3e (should be ~0)\n", max_abs_diff(img, Vec3(1, 1, 1)));
  if (!ref.empty()) {
    std::vector<Vec3> ref_img;
    int rw, rh;
    if (!read_pfm(ref, ref_img, rw, rh) || rw != p.width || rh != p.height) {
      fprintf(stderr, "cannot read reference %s (or size mismatch)\n", ref.c_str());
    } else {
      printf("RMSE vs %s = %.6f\n", ref.c_str(), rmse(img, ref_img));
    }
  }

  if (!write_ppm(out, img, p.width, p.height)) fprintf(stderr, "failed to write %s\n", out.c_str());
  if (!pfm_out.empty() && !write_pfm(pfm_out, img, p.width, p.height)) fprintf(stderr, "failed to write %s\n", pfm_out.c_str());
  return 0;
}
