#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ivf.h"

namespace {

struct Params {
  long n = 1000000;
  long d = 128;
  long nq = 10000;
  long nlist = 1024;
  long nprobe = 16;
  long k = 10;
  long ntrain = 50000;
  long iters = 10;
  long ngt = 100;
  long nblob = 2000;
  double sigma = 0.25;
  uint64_t seed = 12345;
};

[[noreturn]] void usage(const char* prog) {
  fprintf(
    stderr,
    "usage: %s [options]\n"
    "  -n N       database vectors      (default 1000000)\n"
    "  -d D       dimension             (default 128)\n"
    "  -q Q       queries               (default 10000)\n"
    "  -l NLIST   clusters / lists      (default 1024)\n"
    "  -p NPROBE  lists probed / query  (default 16)\n"
    "  -k K       top-k                 (default 10)\n"
    "  -t NTRAIN  k-means sample size   (default 50000)\n"
    "  -i ITERS   k-means iterations    (default 10)\n"
    "  -g NGT     queries for recall    (default 100)\n"
    "  -b NBLOB   blobs in the data     (default 2000)\n"
    "  -r SIGMA   noise around a blob   (default 0.25; larger = harder)\n"
    "  -s SEED    random seed           (default 12345)\n",
    prog
  );
  exit(1);
}

void set_option(Params& params, char opt, const char* val, const char* prog) {
  long v = atol(val);
  switch (opt) {
    case 'n': params.n = v; break;
    case 'd': params.d = v; break;
    case 'q': params.nq = v; break;
    case 'l': params.nlist = v; break;
    case 'p': params.nprobe = v; break;
    case 'k': params.k = v; break;
    case 't': params.ntrain = v; break;
    case 'i': params.iters = v; break;
    case 'g': params.ngt = v; break;
    case 'b': params.nblob = v; break;
    case 'r': params.sigma = atof(val); break;
    case 's': params.seed = strtoull(val, nullptr, 10); break;
    default: usage(prog);
  }
}

Params parse_args(int argc, char** argv) {
  Params params;
  for (int i = 1; i < argc; i += 2) {
    if (argv[i][0] != '-' || strlen(argv[i]) != 2 || i + 1 >= argc) usage(argv[0]);
    set_option(params, argv[i][1], argv[i + 1], argv[0]);
  }
  if (params.n < 1 || params.d < 1 || params.nq < 1 || params.k < 1 || params.nlist < 1 || params.nblob < 1 || params.sigma < 0) usage(argv[0]);
  params.ntrain = std::min(std::max(params.ntrain, 1L), params.n);
  params.nlist = std::min(params.nlist, params.ntrain);
  params.nprobe = std::min(std::max(params.nprobe, 1L), params.nlist);
  params.ngt = std::min(std::max(params.ngt, 0L), params.nq);
  params.k = std::min(params.k, params.n);
  return params;
}

double now() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

void print_lists(const ivf::Index& index, long n) {
  long min_len = n, max_len = 0;
  for (long c = 0; c < index.nlist; c++) {
    min_len = std::min(min_len, index.list_size(c));
    max_len = std::max(max_len, index.list_size(c));
  }
  printf("  list length: min %ld, avg %.1f, max %ld\n", min_len, (double)n / index.nlist, max_len);
}

}  // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  Params params = parse_args(argc, argv);
  const long n = params.n, d = params.d, nq = params.nq, k = params.k;
  printf(
    "n=%ld d=%ld nq=%ld nlist=%ld nprobe=%ld k=%ld ntrain=%ld iters=%ld ngt=%ld "
    "nblob=%ld sigma=%g seed=%llu\n",
    n, d, nq, params.nlist, params.nprobe, k, params.ntrain, params.iters, params.ngt, params.nblob,
    params.sigma, (unsigned long long)params.seed
  );

  double t_start = now();
  ivf::Rng rng(params.seed);
  std::vector<float> blobs(params.nblob * d), X(n * d), Q(nq * d);
  for (auto& v : blobs) {
    v = (float)rng.uniform();
  }
  ivf::generate_blobs(X.data(), n, d, blobs.data(), params.nblob, params.sigma, rng);
  ivf::generate_blobs(Q.data(), nq, d, blobs.data(), params.nblob, params.sigma, rng);
  double t_data = now();
  printf("data generation : %8.3f s\n", t_data - t_start);

  ivf::Index index;
  index.d = d;
  index.nlist = params.nlist;
  index.centroids.resize(params.nlist * d);
  auto history = ivf::train_kmeans(X.data(), params.ntrain, d, params.nlist, params.iters, index.centroids.data(), rng);
  double t_train = now();
  for (size_t it = 0; it < history.size(); it++) {
    printf("  kmeans iter %2zu: objective %.6f (empty clusters %ld)\n", it, history[it].objective, history[it].empty);
  }
  printf("kmeans train    : %8.3f s\n", t_train - t_data);

  ivf::build_lists(index, X.data(), n);
  double t_build = now();
  printf("build lists     : %8.3f s\n", t_build - t_train);
  print_lists(index, n);

  std::vector<long> ids(nq * k);
  std::vector<float> dists(nq * k);
  long scanned = ivf::search(index, Q.data(), nq, params.nprobe, k, ids.data(), dists.data());
  double t_search = now();
  double flops = 3.0 * d * ((double)nq * params.nlist + (double)scanned);
  double search_time = t_search - t_build;
  printf(
    "search          : %8.3f s  (%.1f queries/s, %.1f vectors/query, %.2f GFLOP/s)\n",
    search_time, nq / search_time, (double)scanned / nq, flops / search_time * 1e-9
  );
  printf("main computation (kmeans + build + search): %.3f s\n", t_search - t_data);

  std::vector<long> gt_ids(params.ngt * k);
  std::vector<float> gt_dists(params.ngt * k);
  ivf::brute_force(X.data(), n, d, Q.data(), params.ngt, k, gt_ids.data(), gt_dists.data());
  double t_check = now();
  printf("brute force     : %8.3f s  (%ld queries)\n", t_check - t_search, params.ngt);

  double checksum = 0.0;
  for (long q = 0; q < nq; q++) {
    checksum += dists[q * k];
  }
  printf("recall@%ld = %.4f\n", k, ivf::recall_at_k(ids.data(), gt_ids.data(), params.ngt, k));
  printf("checksum (sum of nearest distances) = %.6f\n", checksum);
  return 0;
}
