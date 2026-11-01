#include "ivf.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ivf {

namespace {

inline float dist2(const float* a, const float* b, long d) {
  float sum = 0.0f;
  for (long j = 0; j < d; j++) {
    float diff = a[j] - b[j];
    sum += diff * diff;
  }
  return sum;
}

// k smallest (dist, id) pairs seen so far, sorted ascending
class TopK {
 public:
  TopK(long k, float* dist, long* id) : k_(k), dist_(dist), id_(id) {}

  void push(float dist, long id) {
    if (size_ == k_ && dist >= dist_[k_ - 1]) return;
    long slot = size_ < k_ ? size_++ : k_ - 1;
    for (; slot > 0 && dist_[slot - 1] > dist; slot--) {
      dist_[slot] = dist_[slot - 1];
      id_[slot] = id_[slot - 1];
    }
    dist_[slot] = dist;
    id_[slot] = id;
  }

  void pad() {
    for (long slot = size_; slot < k_; slot++) {
      dist_[slot] = INFINITY;
      id_[slot] = -1;
    }
  }

 private:
  long k_;
  long size_ = 0;
  float* dist_;
  long* id_;
};

long nearest(const float* x, const float* C, long nc, long d, float* min_dist) {
  long best = 0;
  float best_dist = INFINITY;
  for (long c = 0; c < nc; c++) {
    float dist = dist2(x, &C[c * d], d);
    if (dist < best_dist) {
      best_dist = dist;
      best = c;
    }
  }
  *min_dist = best_dist;
  return best;
}

void copy_vec(float* dst, const float* src, long d) { memcpy(dst, src, d * sizeof(float)); }

void init_centroids(const float* X, long n, long d, long nlist, float* C, Rng& rng) {
  std::vector<long> perm(n);
  for (long i = 0; i < n; i++) {
    perm[i] = i;
  }
  for (long c = 0; c < nlist; c++) {
    long pick = c + (long)(rng.next() % (uint64_t)(n - c));
    std::swap(perm[c], perm[pick]);
    copy_vec(&C[c * d], &X[perm[c] * d], d);
  }
}

double assign_all(const float* X, long n, long d, const float* C, long nlist, long* assign) {
  double total_dist = 0.0;
  for (long i = 0; i < n; i++) {
    float min_dist;
    assign[i] = nearest(&X[i * d], C, nlist, d, &min_dist);
    total_dist += min_dist;
  }
  return total_dist / n;
}

void accumulate(const float* X, long n, long d, const long* assign, double* sum, long* count) {
  for (long i = 0; i < n; i++) {
    long c = assign[i];
    count[c]++;
    for (long j = 0; j < d; j++) {
      sum[c * d + j] += X[i * d + j];
    }
  }
}

long update_centroids(const float* X, long n, long d, long nlist, const double* sum, const long* count, float* C, Rng& rng) {
  long empty = 0;
  for (long c = 0; c < nlist; c++) {
    if (count[c] == 0) {
      empty++;
      copy_vec(&C[c * d], &X[(long)(rng.next() % (uint64_t)n) * d], d);
      continue;
    }
    for (long j = 0; j < d; j++) {
      C[c * d + j] = (float)(sum[c * d + j] / count[c]);
    }
  }
  return empty;
}

void select_probes(const Index& index, const float* x, long nprobe, float* probe_dist, long* probe_list) {
  TopK probes(nprobe, probe_dist, probe_list);
  for (long c = 0; c < index.nlist; c++) {
    probes.push(dist2(x, &index.centroids[c * index.d], index.d), c);
  }
}

long scan_list(const Index& index, const float* x, long c, TopK& result) {
  for (long entry = index.offsets[c]; entry < index.offsets[c + 1]; entry++) {
    result.push(dist2(x, &index.vecs[entry * index.d], index.d), index.ids[entry]);
  }
  return index.list_size(c);
}

bool contains(const long* ids, long k, long id) {
  for (long j = 0; j < k; j++) {
    if (ids[j] == id) return true;
  }
  return false;
}

}  // namespace

Rng::Rng(uint64_t seed) : state_(seed) {}

uint64_t Rng::next() {
  uint64_t z = (state_ += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

double Rng::uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }

double Rng::normal() {
  double u1 = std::max(uniform(), 1e-300);
  double u2 = uniform();
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

void generate_blobs(float* X, long n, long d, const float* centers, long ncenters, double sigma, Rng& rng) {
  for (long i = 0; i < n; i++) {
    const float* center = &centers[(long)(rng.next() % (uint64_t)ncenters) * d];
    for (long j = 0; j < d; j++) {
      X[i * d + j] = (float)(center[j] + sigma * rng.normal());
    }
  }
}

std::vector<KmeansIter> train_kmeans(const float* X, long n, long d, long nlist, long iters, float* C, Rng& rng) {
  init_centroids(X, n, d, nlist, C, rng);
  std::vector<long> assign(n);
  std::vector<double> sum(nlist * d);
  std::vector<long> count(nlist);
  std::vector<KmeansIter> history;
  for (long it = 0; it < iters; it++) {
    double objective = assign_all(X, n, d, C, nlist, assign.data());
    std::fill(sum.begin(), sum.end(), 0.0);
    std::fill(count.begin(), count.end(), 0);
    accumulate(X, n, d, assign.data(), sum.data(), count.data());
    long empty = update_centroids(X, n, d, nlist, sum.data(), count.data(), C, rng);
    history.push_back({objective, empty});
  }
  return history;
}

void build_lists(Index& index, const float* X, long n) {
  long d = index.d;
  std::vector<long> assign(n);
  assign_all(X, n, d, index.centroids.data(), index.nlist, assign.data());

  index.offsets.assign(index.nlist + 1, 0);
  for (long i = 0; i < n; i++) {
    index.offsets[assign[i] + 1]++;
  }
  for (long c = 0; c < index.nlist; c++) {
    index.offsets[c + 1] += index.offsets[c];
  }

  index.ids.resize(n);
  index.vecs.resize(n * d);
  std::vector<long> next_slot(index.offsets.begin(), index.offsets.end() - 1);
  for (long i = 0; i < n; i++) {
    long entry = next_slot[assign[i]]++;
    index.ids[entry] = i;
    copy_vec(&index.vecs[entry * d], &X[i * d], d);
  }
}

long search(const Index& index, const float* Q, long nq, long nprobe, long k, long* out_ids, float* out_dists) {
  std::vector<float> probe_dist(nprobe);
  std::vector<long> probe_list(nprobe);
  long scanned = 0;
  for (long q = 0; q < nq; q++) {
    const float* x = &Q[q * index.d];
    select_probes(index, x, nprobe, probe_dist.data(), probe_list.data());
    TopK result(k, &out_dists[q * k], &out_ids[q * k]);
    for (long p = 0; p < nprobe; p++) {
      scanned += scan_list(index, x, probe_list[p], result);
    }
    result.pad();
  }
  return scanned;
}

void brute_force(const float* X, long n, long d, const float* Q, long nq, long k, long* out_ids, float* out_dists) {
  for (long q = 0; q < nq; q++) {
    TopK result(k, &out_dists[q * k], &out_ids[q * k]);
    for (long i = 0; i < n; i++) {
      result.push(dist2(&Q[q * d], &X[i * d], d), i);
    }
    result.pad();
  }
}

double recall_at_k(const long* found, const long* truth, long nq, long k) {
  if (nq == 0) return 0.0;
  long hits = 0;
  for (long q = 0; q < nq; q++) {
    for (long j = 0; j < k; j++) {
      hits += truth[q * k + j] >= 0 && contains(&found[q * k], k, truth[q * k + j]);
    }
  }
  return (double)hits / (nq * k);
}

}  // namespace ivf
