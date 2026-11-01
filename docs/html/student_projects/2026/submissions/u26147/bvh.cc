#include "bvh.h"
#include <algorithm>

void BVH::build(const std::vector<Triangle>& tris, int leaf_size) {
  nodes.clear();
  tri_index.resize(tris.size());
  if (tris.empty()) return;
  std::vector<Vec3> centroids(tris.size());
  for (size_t i = 0; i < tris.size(); i++) {
    tri_index[i] = (int)i;
    centroids[i] = (tris[i].v0 + tris[i].v1 + tris[i].v2) / 3.0;
  }
  nodes.reserve(2 * tris.size() / leaf_size + 1);
  build_rec(tris, centroids, 0, (int)tris.size(), leaf_size);
}

int BVH::build_rec(const std::vector<Triangle>& tris, const std::vector<Vec3>& centroids,
                   int first, int count, int leaf_size) {
  int id = (int)nodes.size();
  nodes.push_back(BVHNode());
  AABB box, cbox;
  for (int i = first; i < first + count; i++) {
    box.expand(bounds(tris[tri_index[i]]));
    cbox.expand(centroids[tri_index[i]]);
  }
  nodes[id].box = box;
  if (count <= leaf_size) {
    nodes[id].first = first;
    nodes[id].count = count;
    return id;
  }
  // split at the median along the longest axis of the centroid bounds
  Vec3 ext = cbox.hi - cbox.lo;
  int axis = (ext.x > ext.y && ext.x > ext.z) ? 0 : (ext.y > ext.z ? 1 : 2);
  int mid = first + count / 2;
  std::nth_element(tri_index.begin() + first, tri_index.begin() + mid, tri_index.begin() + first + count,
                   [&](int a, int b) { return centroids[a][axis] < centroids[b][axis]; });
  int l = build_rec(tris, centroids, first, mid - first, leaf_size);
  int r = build_rec(tris, centroids, mid, first + count - mid, leaf_size);
  nodes[id].left = l;  // nodes may have been reallocated; index again
  nodes[id].right = r;
  return id;
}

bool BVH::intersect(const Ray& r, const std::vector<Triangle>& tris, Hit& hit) const {
  if (nodes.empty()) return false;
  bool found = false;
  int stack[64];
  int sp = 0;
  stack[sp++] = 0;
  while (sp > 0) {
    const BVHNode& node = nodes[stack[--sp]];
    if (!node.box.intersect(r, kEps, hit.t)) continue;
    if (node.left < 0) {
      for (int i = node.first; i < node.first + node.count; i++)
        found |= ::intersect(r, tris[tri_index[i]], hit);
    } else {
      stack[sp++] = node.right;
      stack[sp++] = node.left;
    }
  }
  return found;
}

int BVH::depth() const { return nodes.empty() ? 0 : depth_rec(0); }

int BVH::depth_rec(int node) const {
  if (nodes[node].left < 0) return 1;
  return 1 + std::max(depth_rec(nodes[node].left), depth_rec(nodes[node].right));
}
