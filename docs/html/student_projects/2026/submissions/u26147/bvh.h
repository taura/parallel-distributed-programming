// bvh.h : bounding volume hierarchy over triangles
#pragma once
#include "geometry.h"
#include <vector>

struct BVHNode {
  AABB box;
  int left = -1, right = -1;  // child node indices (-1 for leaf)
  int first = 0, count = 0;   // range in BVH::tri_index (leaf only)
};

struct BVH {
  std::vector<BVHNode> nodes;  // nodes[0] is the root
  std::vector<int> tri_index;  // permutation of triangle indices

  // Build over `tris` with a median split on the longest axis of the centroid
  // bounds. leaf_size: max triangles per leaf. (SAH is a possible optimization.)
  void build(const std::vector<Triangle>& tris, int leaf_size);

  // Closest hit (iterative traversal with an explicit stack);
  // returns true if anything was hit
  bool intersect(const Ray& r, const std::vector<Triangle>& tris, Hit& hit) const;

  // statistics for the performance analysis
  int depth() const;

 private:
  int build_rec(const std::vector<Triangle>& tris, const std::vector<Vec3>& centroids,
                int first, int count, int leaf_size);
  int depth_rec(int node) const;
};
