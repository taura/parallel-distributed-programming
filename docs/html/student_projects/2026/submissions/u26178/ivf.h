// ivf.h : a mini vector database (IVF-Flat index), public API
//
// An IVF (inverted file) index answers approximate k-nearest-neighbor queries:
//   build : k-means clusters the vectors into nlist clusters; each cluster keeps
//           an inverted list of the vectors nearest to its centroid
//   search: a query scans only the lists of its nprobe nearest centroids
//
// Conventions
//   - a set of n vectors of dimension d is a flat float array of n * d elements;
//     vector i occupies [i * d, i * d + d)
//   - distance is the squared Euclidean (L2) distance
//   - top-k results are sorted by ascending distance; missing slots are id -1
//   - everything is serial and deterministic for a given seed
#pragma once

#include <cstdint>
#include <vector>

namespace ivf {

// Deterministic pseudo-random generator (splitmix64).
// Implemented by hand so that the same seed gives the same data on every
// machine and standard library.
class Rng {
 public:
  explicit Rng(uint64_t seed);
  uint64_t next();   // uniform 64-bit integer
  double uniform();  // uniform in [0, 1)
  double normal();   // standard normal N(0, 1) (Box-Muller)

 private:
  uint64_t state_;
};

// Synthetic data: a mixture of Gaussian blobs.
// Each of the n output vectors X is centers[b] + sigma * N(0, I) for a blob b
// chosen uniformly from the ncenters rows of centers.
void generate_blobs(float* X, long n, long d, const float* centers, long ncenters, double sigma, Rng& rng);

// Statistics of one k-means iteration.
struct KmeansIter {
  double objective;  // mean squared distance to the nearest centroid (assignment step)
  long empty;        // clusters that received no vector (re-seeded randomly)
};

// Lloyd's k-means on the n vectors X.
// Writes nlist centroids (nlist * d floats) to centroids; returns one entry
// per iteration. The objective should be non-increasing over iterations.
// Requires 1 <= nlist <= n.
std::vector<KmeansIter> train_kmeans(const float* X, long n, long d, long nlist, long iters, float* centroids, Rng& rng);

// The IVF index.
// List c holds entries [offsets[c], offsets[c + 1]); entry e is the database
// vector ids[e], whose coordinates are copied to vecs[e * d, e * d + d) so that
// each list is contiguous in memory.
struct Index {
  long d = 0;
  long nlist = 0;
  std::vector<float> centroids;  // nlist * d
  std::vector<long> offsets;     // nlist + 1
  std::vector<long> ids;         // n
  std::vector<float> vecs;       // n * d

  long list_size(long c) const { return offsets[c + 1] - offsets[c]; }
};

// Fills index.offsets, index.ids and index.vecs by assigning each of the n
// vectors X to its nearest centroid. index.d, index.nlist and index.centroids
// must already be set.
void build_lists(Index& index, const float* X, long n);

// Approximate top-k search for nq queries Q, scanning nprobe lists per query.
// Writes nq * k results to out_ids / out_dists (row q = query q).
// Returns the total number of database vectors scanned (for statistics).
long search(const Index& index, const float* Q, long nq, long nprobe, long k, long* out_ids, float* out_dists);

// Exact top-k search by scanning all n vectors X. Same output format as search.
void brute_force(const float* X, long n, long d, const float* Q, long nq, long k, long* out_ids, float* out_dists);

// recall@k: the fraction of the true top-k ids (truth) that also appear in the
// approximate top-k ids (found), averaged over nq queries. 1.0 = exact.
double recall_at_k(const long* found, const long* truth, long nq, long k);

}  // namespace ivf
