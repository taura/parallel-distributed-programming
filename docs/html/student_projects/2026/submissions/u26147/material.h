// material.h : surface materials and BSDF sampling
#pragma once
#include "vec3.h"
#include "rng.h"

enum class MaterialType { Diffuse, Mirror, Glass, Light };

struct Material {
  MaterialType type;
  Vec3 albedo;    // reflectance (Diffuse / Mirror / Glass tint)
  Vec3 emission;  // radiance emitted (Light only)
  double ior;     // index of refraction (Glass only)
};

// Sample an outgoing direction for incoming direction `wi` (pointing towards
// the surface) at a surface with normal `n`. Returns the new direction and
// writes the throughput weight (BSDF * cos / pdf) into `weight`.
//   Diffuse : cosine-weighted hemisphere sampling (weight = albedo)
//   Mirror  : perfect reflection (weight = albedo)
//   Glass   : Fresnel (Schlick) chooses reflection or refraction (weight = albedo)
//   Light   : absorbs (weight = 0)
Vec3 sample_bsdf(const Material& m, const Vec3& wi, const Vec3& n, Rng& rng, Vec3& weight);
