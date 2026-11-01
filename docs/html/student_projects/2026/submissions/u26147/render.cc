#include "render.h"

Vec3 trace_path(const Scene& scene, Ray r, const RenderParams& p, Rng& rng, uint64_t& rays) {
  Vec3 throughput(1, 1, 1);
  Vec3 radiance;
  for (int depth = 0; depth < p.max_depth; depth++) {
    Hit hit;
    rays++;
    if (!scene.intersect(r, hit)) {
      radiance += throughput * scene.background;
      break;
    }
    const Material& m = scene.materials[hit.material];
    radiance += throughput * m.emission;
    if (m.type == MaterialType::Light) break;

    Vec3 weight;
    Vec3 dir = sample_bsdf(m, r.d, hit.n, rng, weight);
    throughput *= weight;

    // Russian roulette: terminate dim paths randomly, boost survivors (unbiased)
    if (depth >= p.rr_depth) {
      double q = std::min(max_component(throughput), 0.95);
      if (rng.uniform() >= q) break;
      throughput *= 1.0 / q;
    }
    r.o = hit.p;
    r.d = dir;
  }
  return radiance;
}

std::vector<Vec3> render(const Scene& scene, const RenderParams& p, uint64_t& rays) {
  std::vector<Vec3> img(p.width * p.height);
  rays = 0;
  for (int py = 0; py < p.height; py++) {
    for (int px = 0; px < p.width; px++) {
      uint64_t pixel = (uint64_t)py * p.width + px;
      Vec3 sum;
      for (int s = 0; s < p.spp; s++) {
        Rng rng(make_seed(pixel, s));
        double jx = rng.uniform(), jy = rng.uniform();  // anti-aliasing jitter
        Ray r = scene.camera.generate_ray(px, py, p.width, p.height, jx, jy);
        sum += trace_path(scene, r, p, rng, rays);
      }
      img[pixel] = sum / p.spp;
    }
  }
  return img;
}
