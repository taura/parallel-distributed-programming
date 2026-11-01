#include "material.h"

static const double kPi = 3.14159265358979323846;

static Vec3 reflect(const Vec3& d, const Vec3& n) { return d - n * (2.0 * dot(d, n)); }

// cosine-weighted direction around n (pdf = cos(theta) / pi)
static Vec3 sample_cosine_hemisphere(const Vec3& n, Rng& rng) {
  double r1 = rng.uniform(), r2 = rng.uniform();
  double phi = 2.0 * kPi * r1;
  double r = std::sqrt(r2);
  // orthonormal basis (t, b, n)
  Vec3 a = std::fabs(n.x) > 0.9 ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
  Vec3 t = normalize(cross(a, n));
  Vec3 b = cross(n, t);
  return normalize(t * (r * std::cos(phi)) + b * (r * std::sin(phi)) + n * std::sqrt(1.0 - r2));
}

Vec3 sample_bsdf(const Material& m, const Vec3& wi, const Vec3& n, Rng& rng, Vec3& weight) {
  bool outside = dot(wi, n) < 0.0;
  Vec3 nf = outside ? n : -n;  // normal facing the incoming ray
  switch (m.type) {
    case MaterialType::Diffuse:
      // BSDF = albedo/pi, pdf = cos/pi -> weight = albedo
      weight = m.albedo;
      return sample_cosine_hemisphere(nf, rng);
    case MaterialType::Mirror:
      weight = m.albedo;
      return reflect(wi, nf);
    case MaterialType::Glass: {
      weight = m.albedo;
      double eta = outside ? 1.0 / m.ior : m.ior;  // n_from / n_to
      double cos_i = -dot(wi, nf);
      double sin2_t = eta * eta * (1.0 - cos_i * cos_i);
      if (sin2_t >= 1.0) return reflect(wi, nf);  // total internal reflection
      double cos_t = std::sqrt(1.0 - sin2_t);
      // Schlick's approximation of the Fresnel reflectance
      double r0 = (1.0 - m.ior) / (1.0 + m.ior);
      r0 *= r0;
      double c = outside ? cos_i : cos_t;  // cosine on the air side
      double fr = r0 + (1.0 - r0) * std::pow(1.0 - c, 5.0);
      if (rng.uniform() < fr) return reflect(wi, nf);
      return normalize(wi * eta + nf * (eta * cos_i - cos_t));
    }
    case MaterialType::Light:
    default:
      weight = Vec3();
      return Vec3(0, 1, 0);
  }
}
