#ifndef PATH_TRACE_H
#define PATH_TRACE_H

// Serial RGB path tracer, using the geometry and vector operations in raytrace.cc.
struct PathOptions {
  int max_depth = 8;
  int light_samples = 1;
  int rr_depth = 3;
  bool nee = true;
  bool ggx = false;
  double roughness = -1.0;  // Negative: use each material's roughness.
  double environment = 0.15;
};

struct PathStats {
  std::uint64_t segments = 0, shadows = 0, diffuse = 0, mirror = 0;
  std::uint64_t refraction = 0, roulette = 0, glossy = 0;
};

static double path_random(std::mt19937& rng) {
  return std::uniform_real_distribution<double>(0.0, 1.0)(rng);
}

static double power_weight(double a, double b) {
  // Scaling avoids overflow when a light's solid-angle PDF is large.
  const double scale = std::max(a, b);
  if (scale == 0.0) return 0.0;
  a /= scale;
  b /= scale;
  return a * a / (a * a + b * b);
}

static Vec3 cosine_direction(const Vec3& n, std::mt19937& rng) {
  const double u = path_random(rng);
  const double phi = 6.28318530717958647692 * path_random(rng);
  const Vec3 tangent = normalize(cross(std::abs(n.y) < 0.9 ? Vec3(0, 1, 0) : Vec3(1, 0, 0), n));
  const Vec3 bitangent = cross(n, tangent);
  return normalize(tangent * (std::sqrt(u) * std::cos(phi)) +
                   bitangent * (std::sqrt(u) * std::sin(phi)) + n * std::sqrt(1.0 - u));
}

#include "microfacet.h"

static double dielectric_fresnel(double cos_i, double eta_i, double eta_t) {
  cos_i = std::clamp(cos_i, 0.0, 1.0);
  const double ratio = eta_i / eta_t;
  const double sin_t2 = ratio * ratio * (1.0 - cos_i * cos_i);
  if (sin_t2 >= 1.0) return 1.0;
  const double cos_t = std::sqrt(1.0 - sin_t2);
  const double rs = (eta_i * cos_i - eta_t * cos_t) / (eta_i * cos_i + eta_t * cos_t);
  const double rp = (eta_t * cos_i - eta_i * cos_t) / (eta_t * cos_i + eta_i * cos_t);
  return 0.5 * (rs * rs + rp * rp);
}

static bool refract_direction(const Vec3& dir, const Vec3& n, double eta, Vec3& out) {
  const double cos_i = std::clamp(-dot(dir, n), 0.0, 1.0);
  const double sin_t2 = eta * eta * (1.0 - cos_i * cos_i);
  if (sin_t2 >= 1.0) return false;
  out = normalize(dir * eta + n * (eta * cos_i - std::sqrt(1.0 - sin_t2)));
  return true;
}

struct EmitterHit {
  int index = -1;
  double t = std::numeric_limits<double>::infinity();
};

static EmitterHit intersect_emitters(const Ray& ray, const std::vector<Light>& lights) {
  EmitterHit result;
  if (std::abs(ray.dir.y) < 1e-12) return result;
  for (size_t i = 0; i < lights.size(); ++i) {
    const Light& light = lights[i];
    if (light.radius <= 0.0) continue;  // Point lights have zero hittable area.
    const double t = (light.position.y - ray.origin.y) / ray.dir.y;
    const Vec3 p = ray.origin + ray.dir * t;
    if (t > 1e-6 && t < result.t && std::abs(p.x - light.position.x) <= light.radius &&
        std::abs(p.z - light.position.z) <= light.radius) {
      result = {static_cast<int>(i), t};
    }
  }
  return result;
}

static Vec3 emitter_radiance(const Light& light) {
  // In path mode, old light intensity values specify Le / 20. Emission is
  // downward from a horizontal square; area changes total power, not Le.
  return light.color * (20.0 * light.intensity);
}

static Vec3 trace_path(const Ray& input_ray,
                       const std::vector<Sphere>& spheres, const std::vector<Box>& boxes,
                       const std::vector<Ellipsoid>& ellipsoids, const std::vector<Triangle>& triangles,
                       const std::vector<Material>& materials, const std::vector<Light>& lights,
                       const PathOptions& options, std::mt19937& rng, PathStats& stats,
                       std::uint64_t& primary_hits, bool count_primary) {
  const double pi = 3.14159265358979323846;
  Vec3 radiance, beta(1, 1, 1);
  Ray ray = input_ray;
  bool previous_delta = true;
  double previous_pdf = 0.0;
  Vec3 previous_point;
  double eta_scale = 1.0;  // Cancel refractive radiance scaling for roulette only.

  for (int depth = 0; depth <= options.max_depth; ++depth) {
    ++stats.segments;
    const Hit hit = intersect_scene(ray, spheres, boxes, ellipsoids, triangles);
    const EmitterHit emitter = intersect_emitters(ray, lights);
    if (emitter.index >= 0 && (!hit.hit || emitter.t < hit.t)) {
      // The back of an area light is opaque, but does not emit.
      if (ray.dir.y > 0.0) {
        const Light& light = lights[emitter.index];
        double weight = 1.0;
        if (options.nee && !previous_delta) {
          const Vec3 p = ray.origin + ray.dir * emitter.t;
          const Vec3 delta = p - previous_point;
          const double light_pdf = dot(delta, delta) / (4.0 * light.radius * light.radius * ray.dir.y);
          weight = power_weight(previous_pdf, options.light_samples * light_pdf);
        }
        radiance = radiance + mul(beta, emitter_radiance(light)) * weight;
      }
      break;
    }
    if (!hit.hit) {
      radiance = radiance + mul(beta, sky_color(ray.dir)) * options.environment;
      break;
    }
    if (depth == 0 && count_primary) ++primary_hits;
    const Material m = surface_material(hit, materials);
    const bool entering = dot(ray.dir, hit.normal) < 0.0;
    const Vec3 n = entering ? hit.normal : hit.normal * -1.0;
    const Vec3 wo = ray.dir * -1.0;
    const double roughness = options.roughness > 0.0 ? options.roughness : m.roughness;
    const double diffuse_probability = 1.0 - m.reflectivity;

    // Next-event estimation: each light is sampled independently, in area
    // measure. Convert to solid angle before combining with the BSDF PDF.
    if (m.ior <= 1.0 && (options.ggx || diffuse_probability > 0.0)) {
      for (const Light& light : lights) {
        const bool point = light.radius == 0.0;
        if (!options.nee && !point) continue;
        const int samples = point ? 1 : options.light_samples;
        for (int sample = 0; sample < samples; ++sample) {
          Vec3 p = light.position;
          if (!point) {
            const double dx = (2.0 * path_random(rng) - 1.0) * light.radius;
            const double dz = (2.0 * path_random(rng) - 1.0) * light.radius;
            p = p + Vec3(dx, 0, dz);
          }
          const Vec3 delta = p - hit.point;
          const double distance2 = dot(delta, delta);
          if (distance2 < 1e-8) continue;
          const double distance = std::sqrt(distance2);
          const Vec3 wi = delta / distance;
          const double cosine = dot(n, wi);
          const double emitter_cosine = wi.y;
          if (cosine <= 0.0 || (!point && emitter_cosine <= 0.0)) continue;
          ++stats.shadows;
          if (occluded(hit.point, wi, distance - 2e-4, spheres, boxes, ellipsoids, triangles)) continue;
          const EmitterHit blocker = intersect_emitters({hit.point + wi * 1e-4, wi}, lights);
          if (blocker.index >= 0 && blocker.t < distance - 2e-4) continue;
          const BsdfEvaluation bsdf = options.ggx ? evaluate_ggx(m, n, wo, wi, roughness) :
              BsdfEvaluation{m.color * (diffuse_probability / pi), diffuse_probability * cosine / pi};
          const double light_pdf = point ? 1.0 : distance2 / (4.0 * light.radius * light.radius * emitter_cosine);
          // At the hard depth cap no BSDF sample is taken, so NEE gets weight 1.
          const double weight = point || depth == options.max_depth ? 1.0 :
                                power_weight(samples * light_pdf, bsdf.pdf);
          const double factor = cosine * weight / (samples * light_pdf * (point ? distance2 : 1.0));
          radiance = radiance + mul(mul(beta, bsdf.f), emitter_radiance(light)) * factor;
        }
      }
    }
    if (depth == options.max_depth) break;

    Vec3 next_direction;
    previous_point = hit.point;
    if (m.ior > 1.0) {
      // Non-overlapping closed glass solids in air, with no volume absorption.
      const double eta_i = entering ? 1.0 : m.ior;
      const double eta_t = entering ? m.ior : 1.0;
      const double eta = eta_i / eta_t;
      const double fresnel = dielectric_fresnel(-dot(ray.dir, n), eta_i, eta_t);
      Vec3 refracted;
      const bool transmits = refract_direction(ray.dir, n, eta, refracted);
      if (!transmits || path_random(rng) < fresnel) {
        next_direction = reflect(ray.dir, n);
        ++stats.mirror;
      } else {
        next_direction = refracted;
        beta = beta * (eta * eta);  // Radiance-mode refractive Jacobian.
        eta_scale /= eta * eta;
        ++stats.refraction;
      }
      previous_delta = true;
    } else if (options.ggx) {
      if (path_random(rng) < ggx_specular_probability(m, dot(n, wo))) {
        const Vec3 h = sample_ggx_visible_normal(n, wo, roughness, rng);
        next_direction = reflect(ray.dir, h);
        ++stats.glossy;
      } else {
        next_direction = cosine_direction(n, rng);
        ++stats.diffuse;
      }
      const BsdfEvaluation bsdf = evaluate_ggx(m, n, wo, next_direction, roughness);
      if (bsdf.pdf <= 0.0) break;
      previous_pdf = bsdf.pdf;
      previous_delta = false;
      beta = mul(beta, bsdf.f) * (dot(n, next_direction) / bsdf.pdf);
    } else if (path_random(rng) < m.reflectivity) {
      // The ideal mirror weight cancels its branch probability.
      next_direction = reflect(ray.dir, n);
      previous_delta = true;
      ++stats.mirror;
    } else {
      next_direction = cosine_direction(n, rng);
      previous_pdf = diffuse_probability * std::max(0.0, dot(n, next_direction)) / pi;
      beta = mul(beta, m.color);  // f*cos/pdf; both cosine and mixture weight cancel.
      previous_delta = false;
      ++stats.diffuse;
    }
    const double magnitude = std::max({beta.x, beta.y, beta.z});
    if (magnitude == 0.0) break;
    if (depth + 1 >= options.rr_depth) {
      const double survive = std::clamp(magnitude * eta_scale, 0.05, 0.95);
      if (path_random(rng) >= survive) {
        ++stats.roulette;
        break;
      }
      beta = beta / survive;
    }
    ray = {hit.point + next_direction * 1e-4, normalize(next_direction)};
  }
  return radiance;
}

static void make_transport_scene(std::vector<Sphere>& spheres, std::vector<Box>& boxes,
                                  std::vector<Material>& materials, std::vector<Light>& lights,
                                  double light_radius) {
  const int white = static_cast<int>(materials.size());
  materials.push_back({Vec3(0.75, 0.75, 0.75), 0.02, 0.9, 0.0, 1.0, 0.0});
  const int red = static_cast<int>(materials.size());
  materials.push_back({Vec3(0.75, 0.08, 0.06), 0.02, 0.9, 0.0, 1.0, 0.0});
  const int green = static_cast<int>(materials.size());
  materials.push_back({Vec3(0.08, 0.60, 0.12), 0.02, 0.9, 0.0, 1.0, 0.0});
  const int mirror = static_cast<int>(materials.size());
  materials.push_back({Vec3(1, 1, 1), 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.3});
  const int glass = static_cast<int>(materials.size());
  materials.push_back({Vec3(1, 1, 1), 0.0, 0.8, 0.5, 100.0, 0.04, 1.5});
  boxes = {
      {Vec3(-3.1, -2, -7), Vec3(-3, 4, 0), red},
      {Vec3(3, -2, -7), Vec3(3.1, 4, 0), green},
      {Vec3(-3, -2, -7.1), Vec3(3, 4, -7), white},
      {Vec3(-3, -2, -7), Vec3(3, -1.9, 0), white},
      {Vec3(-3, 4, -7), Vec3(3, 4.1, 0), white},
  };
  spheres = {
      {Vec3(-1.35, -0.85, -4.7), 1.05, white},
      {Vec3(1.30, -0.95, -4.9), 0.95, mirror},
      {Vec3(0.55, -1.15, -2.4), 0.75, glass},
  };
  lights = {{Vec3(0, 3.85, -3.5), Vec3(1.0, 0.95, 0.86), 0.75, light_radius}};
}

static void check_ggx_transport() {
  const std::vector<Material> materials(4, {Vec3(0.7, 0.5, 0.3), 0, 0, 0, 1, 0, 0, 0.4, 0.7});
  const std::vector<Sphere> spheres = {{Vec3(0, -1, 0), 0.5, 0}};
  const std::vector<Light> lights = {{Vec3(0, 2, 0), Vec3(1, 1, 1), 1.0, 1.0}};
  const Ray primary{Vec3(0, 1, 0), Vec3(0, -1, 0)};
  PathOptions options;
  options.ggx = true;
  options.environment = 0.0;
  options.rr_depth = 100;
  PathStats stats;
  std::uint64_t hits = 0;
  std::mt19937 rng(821);
  Vec3 direct, combined;
  for (int i = 0; i < 20000; ++i) {
    options.max_depth = 0;
    direct = direct + trace_path(primary, spheres, {}, {}, {}, materials, lights,
                                options, rng, stats, hits, false) / 20000.0;
    options.max_depth = 1;
    combined = combined + trace_path(primary, spheres, {}, {}, {}, materials, lights,
                                    options, rng, stats, hits, false) / 20000.0;
  }
  if (length(direct - combined) > 0.08) {
    throw std::runtime_error("GGX mixture NEE versus MIS transport check failed");
  }
  std::cout << "GGX mixture transport passed, RGB estimate difference: " << length(direct - combined) << "\n";
}

static void check_transport() {
  Vec3 refracted;
  if (std::abs(dielectric_fresnel(1.0, 1.0, 1.5) - 0.04) > 1e-12 ||
      dielectric_fresnel(0.5, 1.5, 1.0) != 1.0 ||
      !refract_direction(Vec3(0, 0, -1), Vec3(0, 0, 1), 1.0 / 1.5, refracted) ||
      length(refracted - Vec3(0, 0, -1)) > 1e-12 ||
      refract_direction(normalize(Vec3(0.866025403784, 0, -0.5)), Vec3(0, 0, 1), 1.5, refracted) ||
      std::abs(power_weight(0.2, 3) + power_weight(3, 0.2) - 1.0) > 1e-12) {
    throw std::runtime_error("Fresnel/Snell/TIR/MIS check failed");
  }
  std::mt19937 rng(123);
  double mean_cosine = 0.0;
  for (int i = 0; i < 10000; ++i) {
    const Vec3 d = cosine_direction(Vec3(0, 1, 0), rng);
    if (d.y < 0.0 || std::abs(length(d) - 1.0) > 1e-12) {
      throw std::runtime_error("Hemisphere direction check failed");
    }
    mean_cosine += d.y / 10000.0;
  }
  if (std::abs(mean_cosine - 2.0 / 3.0) > 0.015) {
    throw std::runtime_error("Cosine-weighted sampling moment check failed");
  }
  PathOptions options;
  options.max_depth = 1;
  options.rr_depth = 100;
  options.environment = 0;
  PathStats stats;
  std::uint64_t hits = 0;
  const std::vector<Material> mirror_materials(4, {Vec3(1, 1, 1), 0, 0, 0, 1, 1});
  const std::vector<Light> overhead = {{Vec3(0, 2, 0), Vec3(1, 1, 1), 1.0, 1.0}};
  const Vec3 reflected = trace_path({Vec3(0, 1, 0), Vec3(0, -1, 0)},
      {{Vec3(0, -1, 0), 0.5, 0}}, {}, {}, {}, mirror_materials, overhead, options, rng, stats, hits, false);
  if (length(reflected - Vec3(20, 20, 20)) > 1e-10) {
    throw std::runtime_error("Mirror-to-emitter radiance check failed");
  }
  std::vector<Material> glass_materials = mirror_materials;
  glass_materials[0].ior = 1.5;
  options.max_depth = 16;
  options.environment = 1.0;
  for (int i = 0; i < 100; ++i) {
    const Vec3 through_glass = trace_path({Vec3(), Vec3(0, 0, -1)},
        {{Vec3(0, 0, -3), 1.0, 0}}, {}, {}, {}, glass_materials, {}, options, rng, stats, hits, false);
    if (length(through_glass - sky_color(Vec3(0, 0, -1))) > 1e-10) {
      throw std::runtime_error("Closed-glass eta scaling check failed");
    }
  }
  // Independent estimators of the same one-bounce diffuse lighting integral.
  // NEE at depth=0 versus pure BSDF sampling with one emitter-hit segment.
  const std::vector<Material> diffuse_materials(4, {Vec3(0.6, 0.6, 0.6), 0, 1, 0, 1, 0});
  const std::vector<Sphere> sphere = {{Vec3(0, -1, 0), 0.5, 0}};
  const Ray primary{Vec3(0, 1, 0), Vec3(0, -1, 0)};
  double direct_mean = 0, bsdf_mean = 0, mis_mean = 0, roulette_mean = 0;
  options.environment = 0;
  for (int i = 0; i < 20000; ++i) {
    options.rr_depth = 100;
    options.max_depth = 0;
    options.nee = true;
    direct_mean += trace_path(primary, sphere, {}, {}, {}, diffuse_materials,
                             overhead, options, rng, stats, hits, false).x / 20000.0;
    options.max_depth = 1;
    options.nee = false;
    bsdf_mean += trace_path(primary, sphere, {}, {}, {}, diffuse_materials,
                           overhead, options, rng, stats, hits, false).x / 20000.0;
    options.nee = true;
    mis_mean += trace_path(primary, sphere, {}, {}, {}, diffuse_materials,
                          overhead, options, rng, stats, hits, false).x / 20000.0;
    options.rr_depth = 1;
    roulette_mean += trace_path(primary, sphere, {}, {}, {}, diffuse_materials,
                               overhead, options, rng, stats, hits, false).x / 20000.0;
  }
  if (std::abs(direct_mean - bsdf_mean) > 0.08 || std::abs(direct_mean - mis_mean) > 0.04 ||
      std::abs(direct_mean - roulette_mean) > 0.04) {
    throw std::runtime_error("NEE, BSDF, MIS and roulette diffuse estimates disagree");
  }
  std::cout << "Transport checks passed: Fresnel/Snell/TIR, glass eta scaling, cosine sampling, mirror radiance\n"
            << "Diffuse estimates (NEE / BSDF / MIS / MIS+RR): " << direct_mean << " / " << bsdf_mean
            << " / " << mis_mean << " / " << roulette_mean << "\n";
}

#endif
