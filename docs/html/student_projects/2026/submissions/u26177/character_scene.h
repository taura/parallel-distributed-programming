#ifndef CHARACTER_SCENE_H
#define CHARACTER_SCENE_H

// Procedural, original chibi figurine. Included after the renderer's geometry
// types. All dimensions and colors live here; no model files or textures needed.
static void make_character(std::vector<Sphere>& spheres, std::vector<Box>& boxes,
                           std::vector<Ellipsoid>& ellipsoids,
                           std::vector<Triangle>& triangles,
                           std::vector<Material>& materials, int decorations, int detail) {
  const auto material = [&](Vec3 color, double shine = 0.15, double reflection = 0.0, double metallic = 0.0) {
    const int id = static_cast<int>(materials.size());
    materials.push_back({color, 0.15, 0.85, shine, 70.0, reflection, 0.0, metallic, 0.4});
    return id;
  };
  const int skin = material(Vec3(1.0, 0.72, 0.57), 0.12);
  const int hair = material(Vec3(0.025, 0.48, 0.52), 0.4);
  const int hair_light = material(Vec3(0.055, 0.68, 0.65), 0.4);
  const int ink = material(Vec3(0.018, 0.025, 0.045), 0.08);
  const int white = material(Vec3(0.95, 0.97, 1.0), 0.3);
  const int iris = material(Vec3(0.08, 0.35, 0.80), 0.4);
  const int pink = material(Vec3(0.95, 0.18, 0.38), 0.25);
  const int blush = material(Vec3(1.0, 0.38, 0.40), 0.05);
  const int navy = material(Vec3(0.065, 0.09, 0.19), 0.3);
  const int gold = material(Vec3(0.95, 0.62, 0.20), 0.55, 0.18, 1.0);
  const int backdrop = material(Vec3(0.18, 0.24, 0.36), 0.0);
  const int stage = material(Vec3(0.22, 0.34, 0.44), 0.4, 0.15);
  const int violet = material(Vec3(0.30, 0.10, 0.46), 0.18);
  const int iris_light = material(Vec3(0.10, 0.65, 0.85), 0.18);
  const double pi = 3.14159265358979323846;

  const auto oval = [&](Vec3 center, Vec3 radii, int id, double tilt = 0.0) {
    ellipsoids.push_back({center, radii, id, tilt});
  };
  const auto triangle = [&](Vec3 a, Vec3 b, Vec3 c, int id) {
    triangles.push_back({a, b, c, id, {}, {}, {}});
  };
  const auto quad = [&](Vec3 a, Vec3 b, Vec3 c, Vec3 d, int id) {
    triangle(a, b, c, id);
    triangle(a, c, d, id);
  };
  // Curved facial color patches follow the analytic head surface. Two radial
  // rings keep the flat facets above the skin; each layer is slightly lifted.
  const auto face_patch = [&](double cx, double cy, double rx, double ry, double lift, int id) {
    const auto vertex = [&](double r, double angle) {
      const double x = cx + rx * r * std::cos(angle);
      const double y = cy + ry * r * std::sin(angle);
      const double z = 0.9 * std::sqrt(std::max(0.0, 1.0 - x * x / 1.44 - (y - 1.6) * (y - 1.6) / 1.0404));
      return Vec3(x, y, -5.0 + z + lift);
    };
    const auto normal = [&](const Vec3& p) {
      return normalize(Vec3(p.x / 1.44, (p.y - 1.6) / 1.0404, (p.z + 5.0 - lift) / 0.81));
    };
    const auto patch_triangle = [&](Vec3 a, Vec3 b, Vec3 c) {
      triangles.push_back({a, b, c, id, normal(a), normal(b), normal(c)});
    };
    const int segments = 8 + 8 * detail;
    for (int i = 0; i < segments; ++i) {
      const double a = 2.0 * pi * i / segments;
      const double b = 2.0 * pi * (i + 1) / segments;
      const Vec3 inner_a = vertex(0.5, a), inner_b = vertex(0.5, b);
      const Vec3 outer_a = vertex(1.0, a), outer_b = vertex(1.0, b);
      patch_triangle(vertex(0, 0), inner_a, inner_b);
      patch_triangle(inner_a, outer_a, outer_b);
      patch_triangle(inner_a, outer_b, inner_b);
    }
  };
  // Smooth studio backdrop and a low oval plinth, resting on the ground.
  boxes.push_back({Vec3(-18, -2, -10), Vec3(18, 14, -9.8), backdrop});
  oval(Vec3(0, -1.95, -5), Vec3(3.0, 0.21, 2.1), stage);
  oval(Vec3(0, -2.02, -5), Vec3(3.08, 0.12, 2.18), gold);

  // Head, hair cap, neck and a fully clothed small body.
  oval(Vec3(0, 1.60, -5), Vec3(1.20, 1.02, 0.90), skin);
  oval(Vec3(0, 1.95, -5.25), Vec3(1.28, 0.88, 0.89), hair);
  oval(Vec3(0, 0.63, -5), Vec3(0.23, 0.30, 0.25), skin);
  oval(Vec3(0, 0.0, -5), Vec3(0.56, 0.60, 0.38), white);
  // Eight-fold pleated skirt. The angular resolution changes with detail,
  // while the physical fold count and garment size remain fixed.
  const int skirt_segments = 16 * detail;
  const auto skirt_point = [&](double angle, double t) {
    const double r = 0.49 + 0.44 * t + 0.075 * t * std::cos(8.0 * angle);
    return Vec3(r * std::cos(angle), -0.31 - 0.62 * t,
                -5.0 + 0.72 * r * std::sin(angle));
  };
  for (int i = 0; i < skirt_segments; ++i) {
    const double a = 2.0 * pi * i / skirt_segments;
    const double b = 2.0 * pi * (i + 1) / skirt_segments;
    for (int row = 0; row < 3; ++row) {
      const double t0 = row / 3.0, t1 = (row + 1) / 3.0;
      quad(skirt_point(a, t0), skirt_point(b, t0), skirt_point(b, t1), skirt_point(a, t1),
           row == 2 ? hair_light : navy);
    }
    triangle(Vec3(0, -0.31, -5), skirt_point(a, 0), skirt_point(b, 0), navy);
    triangle(Vec3(0, -0.93, -5), skirt_point(b, 1), skirt_point(a, 1), navy);
  }
  // A draped cape behind the shoulders, widening toward a scalloped hem.
  const auto cape_point = [&](double u, double v) {
    return Vec3(u * (0.54 + 0.9 * v), 0.42 - 1.42 * v + 0.07 * v * std::cos(3 * pi * u),
                -5.37 - 0.25 * v + 0.10 * std::sin(3 * pi * u) * v);
  };
  for (int i = 0; i < 8 * detail; ++i) {
    for (int j = 0; j < 3; ++j) {
      const double u0 = -1.0 + 2.0 * i / (8 * detail), u1 = -1.0 + 2.0 * (i + 1) / (8 * detail);
      const double v0 = j / 3.0, v1 = (j + 1) / 3.0;
      quad(cape_point(u0, v0), cape_point(u1, v0), cape_point(u1, v1), cape_point(u0, v1),
           j == 2 ? gold : violet);
    }
  }
  // A small collar bow and gold center button.
  oval(Vec3(-0.14, 0.37, -4.65), Vec3(0.19, 0.11, 0.06), pink, -0.35);
  oval(Vec3(0.14, 0.37, -4.65), Vec3(0.19, 0.11, 0.06), pink, 0.35);
  oval(Vec3(0, 0.35, -4.58), Vec3(0.07, 0.07, 0.04), gold);
  oval(Vec3(0, 0.09, -4.61), Vec3(0.075, 0.23, 0.045), hair);

  for (double side : {-1.0, 1.0}) {
    // Ears, sleeves, arms, hands, stockings and rounded shoes.
    oval(Vec3(side * 1.15, 1.48, -5), Vec3(0.17, 0.27, 0.18), skin);
    oval(Vec3(side * 0.59, 0.21, -5), Vec3(0.26, 0.32, 0.29), white, side * 0.4);
    const bool holds_staff = side > 0.0;
    oval(Vec3(side * 0.79, -0.14, holds_staff ? -4.80 : -4.97),
         Vec3(0.15, 0.36, holds_staff ? 0.23 : 0.17), skin, side * 0.35);
    oval(Vec3(side * (holds_staff ? 1.00 : 0.90), -0.47, holds_staff ? -4.62 : -4.91),
         Vec3(0.18, 0.20, 0.18), skin);
    oval(Vec3(side * 0.35, -1.12, -5), Vec3(0.19, 0.46, 0.20), navy);
    oval(Vec3(side * 0.35, -1.50, -4.87), Vec3(0.25, 0.24, 0.40), ink);
    oval(Vec3(side * 0.35, -1.39, -4.59), Vec3(0.19, 0.055, 0.10), white);

    // Twin tails: overlapping tilted locks create a curved silhouette.
    oval(Vec3(side * 1.25, 2.02, -5.19), Vec3(0.22, 0.24, 0.32), pink);
    oval(Vec3(side * 1.53, 1.71, -5.30), Vec3(0.38, 0.77, 0.41), hair, side * 0.28);
    oval(Vec3(side * 1.77, 0.80, -5.28), Vec3(0.34, 0.73, 0.35), hair, side * 0.20);
    oval(Vec3(side * 1.85, 0.08, -5.15), Vec3(0.26, 0.48, 0.28), hair_light, -side * 0.20);
    oval(Vec3(side * 1.81, 0.95, -4.98), Vec3(0.065, 0.67, 0.055), hair_light, side * 0.20);
    // Side bangs frame the face, leaving eyes and cheeks visible.
    oval(Vec3(side * 1.01, 1.84, -4.76), Vec3(0.24, 0.66, 0.23), hair, side * 0.13);

    // Curved eye patches: thin eyeliner, white, iris, pupil and catchlights.
    const double x = side * 0.43;
    face_patch(x, 1.58, 0.31, 0.36, 0.014, ink);
    face_patch(x, 1.56, 0.28, 0.32, 0.028, white);
    face_patch(x, 1.56, 0.18, 0.26, 0.042, iris);
    face_patch(x, 1.44, 0.135, 0.105, 0.055, iris_light);
    face_patch(x, 1.59, 0.072, 0.18, 0.068, ink);
    face_patch(x - 0.055, 1.69, 0.050, 0.060, 0.080, white);
    face_patch(x + 0.065, 1.43, 0.024, 0.030, 0.080, white);
    face_patch(side * 0.76, 1.23, 0.17, 0.060, 0.014, blush);
    // Ribbon tails on both hair ties, and slender highlight strands.
    quad(Vec3(side * 1.22, 2.04, -4.85), Vec3(side * 1.43, 2.05, -4.84),
         Vec3(side * 1.62, 1.49, -4.80), Vec3(side * 1.43, 1.62, -4.82), pink);
    for (int strand = 0; strand < 3; ++strand) {
      oval(Vec3(side * (1.61 + 0.08 * strand), 0.96, -4.97),
           Vec3(0.018, 0.56 - 0.06 * strand, 0.025), hair_light, side * 0.20);
    }
  }

  // Separate fringe locks rather than a single helmet-shaped hair mass.
  oval(Vec3(-0.65, 2.27, -4.40), Vec3(0.32, 0.47, 0.20), hair_light, -0.40);
  oval(Vec3(-0.23, 2.30, -4.27), Vec3(0.29, 0.47, 0.20), hair, -0.28);
  oval(Vec3(0.24, 2.31, -4.27), Vec3(0.28, 0.43, 0.20), hair_light, 0.32);
  oval(Vec3(0.65, 2.27, -4.40), Vec3(0.31, 0.47, 0.20), hair, 0.42);
  oval(Vec3(0, 1.30, -4.115), Vec3(0.085, 0.085, 0.065), skin);
  oval(Vec3(0, 1.08, -4.185), Vec3(0.15, 0.05, 0.025), pink);

  // Gold five-pointed stars with raised centers (front/back and edge facets).
  const auto star = [&](Vec3 center, double radius) {
    for (int i = 0; i < 10; ++i) {
      const double a = pi / 2.0 + 2.0 * pi * i / 10;
      const double b = pi / 2.0 + 2.0 * pi * (i + 1) / 10;
      const double ra = radius * (i % 2 == 0 ? 1.0 : 0.45);
      const double rb = radius * (i % 2 == 0 ? 0.45 : 1.0);
      const Vec3 p = center + Vec3(ra * std::cos(a), ra * std::sin(a), 0);
      const Vec3 q = center + Vec3(rb * std::cos(b), rb * std::sin(b), 0);
      triangle(center + Vec3(0, 0, 0.10), p, q, gold);
      triangle(center - Vec3(0, 0, 0.07), q, p, gold);
    }
  };
  star(Vec3(-0.82, 2.36, -4.12), 0.26);  // hair ornament
  // Staff placed beside the right hand, with a jeweled star and two ribbons.
  oval(Vec3(1.04, -0.03, -4.48), Vec3(0.055, 1.40, 0.055), gold);
  star(Vec3(1.04, 1.62, -4.43), 0.43);
  oval(Vec3(1.04, 1.62, -4.30), Vec3(0.12, 0.13, 0.07), pink);
  quad(Vec3(1.03, 1.28, -4.43), Vec3(1.18, 1.24, -4.43),
       Vec3(1.55, 0.72, -4.39), Vec3(1.32, 0.86, -4.43), pink);
  quad(Vec3(0.93, 1.27, -4.44), Vec3(1.05, 1.24, -4.43),
       Vec3(0.89, 0.72, -4.41), Vec3(0.79, 0.88, -4.44), violet);

  // Tiny beads decorate the edge of the stand; changing their count does not
  // alter the character's anatomy. Workload also scales with resolution/spp.
  const double radius = std::min(0.085, 4.0 / decorations);
  for (int i = 0; i < decorations; ++i) {
    const double angle = 2.0 * pi * i / decorations;
    spheres.push_back({Vec3(2.85 * std::cos(angle), -1.87, -5.0 + 1.93 * std::sin(angle)),
                       radius, i % 2 == 0 ? gold : pink});
  }
}

#endif
