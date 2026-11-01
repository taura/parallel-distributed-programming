#ifndef MICROFACET_H
#define MICROFACET_H

// Isotropic GGX, separable Smith masking-shadowing, RGB Schlick Fresnel.
// All vectors point away from the surface, except the incident camera ray.
struct BsdfEvaluation {
  Vec3 f;
  double pdf = 0.0;
};

static double ggx_distribution(double cosine, double alpha) {
  if (cosine <= 0.0) return 0.0;
  const double a2 = alpha * alpha;
  const double d = cosine * cosine * (a2 - 1.0) + 1.0;
  return a2 / (3.14159265358979323846 * d * d);
}

static double ggx_g1(double cosine, double alpha) {
  if (cosine <= 0.0) return 0.0;
  return 2.0 * cosine / (cosine + std::sqrt(alpha * alpha + (1.0 - alpha * alpha) * cosine * cosine));
}

static Vec3 schlick_fresnel(const Vec3& f0, double cosine) {
  const double t = std::pow(1.0 - std::clamp(cosine, 0.0, 1.0), 5.0);
  return f0 + (Vec3(1, 1, 1) - f0) * t;
}

static Vec3 ggx_f0(const Material& m) {
  return Vec3(0.04, 0.04, 0.04) * (1.0 - m.metallic) + m.color * m.metallic;
}

static double ggx_specular_probability(const Material& m, double cosine_o) {
  if (m.metallic == 1.0) return 1.0;
  const Vec3 f = schlick_fresnel(ggx_f0(m), cosine_o);
  return std::clamp(std::max({f.x, f.y, f.z}), 0.1, 0.9);
}

static BsdfEvaluation evaluate_ggx(const Material& m, const Vec3& n,
                                  const Vec3& wo, const Vec3& wi, double roughness) {
  const double co = dot(n, wo), ci = dot(n, wi);
  if (co <= 0.0 || ci <= 0.0) return {};
  const double alpha = roughness * roughness;
  const Vec3 h = normalize(wo + wi);
  const double d = ggx_distribution(std::max(0.0, dot(n, h)), alpha);
  const Vec3 f = schlick_fresnel(ggx_f0(m), dot(wo, h));
  const Vec3 specular = f * (d * ggx_g1(co, alpha) * ggx_g1(ci, alpha) / (4.0 * co * ci));
  const Vec3 diffuse = mul(Vec3(1, 1, 1) - f, m.color) * ((1.0 - m.metallic) / 3.14159265358979323846);
  // VNDF reflection PDF includes the half-vector -> direction Jacobian.
  const double specular_pdf = d * ggx_g1(co, alpha) / (4.0 * co);
  const double probability = ggx_specular_probability(m, co);
  return {diffuse + specular, probability * specular_pdf +
                             (1.0 - probability) * ci / 3.14159265358979323846};
}

static Vec3 sample_ggx_visible_normal(const Vec3& n, const Vec3& wo,
                                      double roughness, std::mt19937& rng) {
  // Heitz's stretched-view disk construction for the GGX visible-normal distribution.
  const double alpha = roughness * roughness;
  const Vec3 x = normalize(cross(std::abs(n.y) < 0.9 ? Vec3(0, 1, 0) : Vec3(1, 0, 0), n));
  const Vec3 y = cross(n, x);
  const Vec3 view = normalize(Vec3(alpha * dot(wo, x), alpha * dot(wo, y), dot(wo, n)));
  const double planar_length2 = view.x * view.x + view.y * view.y;
  const Vec3 t1 = planar_length2 > 0.0 ? Vec3(-view.y, view.x, 0) / std::sqrt(planar_length2) : Vec3(1, 0, 0);
  const Vec3 t2 = cross(view, t1);
  const double radius = std::sqrt(path_random(rng));
  const double phi = 6.28318530717958647692 * path_random(rng);
  const double a = radius * std::cos(phi);
  double b = radius * std::sin(phi);
  const double s = 0.5 * (1.0 + view.z);
  b = (1.0 - s) * std::sqrt(std::max(0.0, 1.0 - a * a)) + s * b;
  const Vec3 disk_normal = t1 * a + t2 * b + view * std::sqrt(std::max(0.0, 1.0 - a * a - b * b));
  const Vec3 local = normalize(Vec3(alpha * disk_normal.x, alpha * disk_normal.y, std::max(0.0, disk_normal.z)));
  return normalize(x * local.x + y * local.y + n * local.z);
}

static void check_microfacet() {
  const Vec3 n(0, 0, 1);
  const Vec3 wo = normalize(Vec3(0.7, 0.1, 1));
  const Vec3 wi = normalize(Vec3(-0.2, 0.4, 1));
  const Material metal{Vec3(1, 1, 1), 0, 0, 0, 1, 1, 0, 1, 0.7};
  const auto forward = evaluate_ggx(metal, n, wo, wi, 0.7);
  const auto reverse = evaluate_ggx(metal, n, wi, wo, 0.7);
  if (length(forward.f - reverse.f) > 1e-12 || forward.pdf <= 0.0 ||
      evaluate_ggx(metal, n, wo, n * -1.0, 0.7).pdf != 0.0) {
    throw std::runtime_error("GGX reciprocity/hemisphere check failed");
  }
  // Compare uniform quadrature-in-expectation with VNDF importance sampling.
  // Rejected below-surface reflections are zero contributions, never resampled.
  std::mt19937 rng(371);
  double uniform_energy = 0.0, vndf_energy = 0.0, integrated_pdf = 0.0, acceptance = 0.0;
  const int count = 50000;
  for (int i = 0; i < count; ++i) {
    const double z = path_random(rng), phi = 6.28318530717958647692 * path_random(rng);
    const double r = std::sqrt(1.0 - z * z);
    const Vec3 direction(r * std::cos(phi), r * std::sin(phi), z);
    const auto value = evaluate_ggx(metal, n, wo, direction, 0.7);
    uniform_energy += value.f.x * z * 6.28318530717958647692 / count;
    integrated_pdf += value.pdf * 6.28318530717958647692 / count;
    const Vec3 h = sample_ggx_visible_normal(n, wo, 0.7, rng);
    const Vec3 sampled = reflect(wo * -1.0, h);
    if (dot(n, h) < 0.0 || dot(wo, h) <= 0.0 || std::abs(length(h) - 1.0) > 1e-12) {
      throw std::runtime_error("GGX visible-normal check failed");
    }
    if (sampled.z > 0.0) {
      acceptance += 1.0 / count;
      const auto evaluated = evaluate_ggx(metal, n, wo, sampled, 0.7);
      const double weight = evaluated.f.x * sampled.z / evaluated.pdf;
      if (!std::isfinite(weight) || weight < 0.0 || weight > 1.00000001) {
        throw std::runtime_error("GGX unit-metal sample weight check failed");
      }
      vndf_energy += weight / count;
    }
  }
  if (std::abs(uniform_energy - vndf_energy) > 0.02 || std::abs(integrated_pdf - acceptance) > 0.02 ||
      uniform_energy > 1.02) {
    throw std::runtime_error("GGX furnace/PDF consistency check failed");
  }
  const Vec3 grazing = normalize(Vec3(1, 0, 0.001));
  for (double roughness : {0.02, 0.4, 1.0}) {
    for (int i = 0; i < 1000; ++i) {
      const Vec3 h = sample_ggx_visible_normal(n, grazing, roughness, rng);
      const Vec3 reflected = reflect(grazing * -1.0, h);
      if (reflected.z <= 0.0) continue;
      const auto value = evaluate_ggx(metal, n, grazing, reflected, roughness);
      const double weight = value.f.x * reflected.z / value.pdf;
      if (!std::isfinite(weight) || weight < 0.0 || weight > 1.0000001) {
        throw std::runtime_error("GGX grazing/roughness-boundary check failed");
      }
    }
  }
  std::cout << "GGX checks passed: reciprocity, visible normals, furnace energy, PDF consistency\n"
            << "Unit-metal energy (uniform / VNDF): " << uniform_energy << " / " << vndf_energy << "\n"
            << "PDF integral / accepted samples: " << integrated_pdf << " / " << acceptance << "\n";
}

#endif
