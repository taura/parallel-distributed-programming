// mqar.cc -- serial baseline: train a small decoder-only Transformer from scratch
// on Multi-Query Associative Recall (MQAR).
//
// Self-contained C++17, standard library only, single thread, plain loops.
// Build: clang++ -O3 -Wall -std=c++17 mqar.cc -o mqar
// Run:   ./mqar --help

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
struct Config {
  int L = 2;          // Transformer blocks
  int d = 64;         // model width
  int H = 4;          // attention heads
  int N = 8;          // key-value pairs in the context
  int Q = 8;          // queries per sequence
  int K = 64;         // number of key tokens
  int Vv = 64;        // number of value tokens
  int B = 32;         // sequences per training step
  int steps = 300;    // training steps
  double lr = 3e-3;   // peak learning rate (Adam)
  int warmup = -1;    // warmup steps (-1: steps/10, at most 100)
  double clip = 1.0;  // global gradient-norm clip (0: off)
  uint64_t seed = 1;
  int eval_every = 25;
  int eval_batches = 8;
  int gradcheck = 0;  // 1: run a finite-difference gradient check first
  int overfit = 0;    // 1: train on one fixed batch (overfit test)
  int shift = 1;      // 1: add an embedding of the previous token (token shift)

  int T() const { return 2 * N + 2 * Q; }
  int V() const { return K + Vv; }
};

static void usage() {
  std::printf(
      "usage: mqar [options]\n"
      "  task : --pairs N --queries Q --keys K --values Vv\n"
      "  model: --layers L --dim d --heads H --shift 0|1 (previous-token embedding)\n"
      "  train: --batch B --steps S --lr LR --warmup W --clip C --seed S\n"
      "  eval : --eval-every E --eval-batches E\n"
      "  check: --gradcheck 1 (double-precision finite differences)\n"
      "         --overfit 1   (train repeatedly on one fixed batch)\n"
      "  sequence length T = 2N + 2Q, vocabulary V = K + Vv\n");
}

static bool parse_args(int argc, char** argv, Config& c) {
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") { usage(); std::exit(0); }
    if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); return false; }
    const char* v = argv[++i];
    if (a == "--layers") c.L = std::atoi(v);
    else if (a == "--dim") c.d = std::atoi(v);
    else if (a == "--heads") c.H = std::atoi(v);
    else if (a == "--pairs") c.N = std::atoi(v);
    else if (a == "--queries") c.Q = std::atoi(v);
    else if (a == "--keys") c.K = std::atoi(v);
    else if (a == "--values") c.Vv = std::atoi(v);
    else if (a == "--batch") c.B = std::atoi(v);
    else if (a == "--steps") c.steps = std::atoi(v);
    else if (a == "--lr") c.lr = std::atof(v);
    else if (a == "--warmup") c.warmup = std::atoi(v);
    else if (a == "--clip") c.clip = std::atof(v);
    else if (a == "--seed") c.seed = std::strtoull(v, nullptr, 10);
    else if (a == "--eval-every") c.eval_every = std::atoi(v);
    else if (a == "--eval-batches") c.eval_batches = std::atoi(v);
    else if (a == "--gradcheck") c.gradcheck = std::atoi(v);
    else if (a == "--overfit") c.overfit = std::atoi(v);
    else if (a == "--shift") c.shift = std::atoi(v);
    else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return false; }
  }
  if (c.L < 1 || c.d < 1 || c.H < 1 || c.d % c.H != 0) { std::fprintf(stderr, "need d %% H == 0\n"); return false; }
  if (c.N < 1 || c.Q < 1 || c.N > c.K || c.Vv < 1 || c.B < 1 || c.steps < 0) {
    std::fprintf(stderr, "invalid task sizes (need 1 <= N <= K)\n"); return false;
  }
  if (c.eval_every < 1) c.eval_every = 1;
  if (c.warmup < 0) c.warmup = std::min(100, c.steps / 10);
  return true;
}

// ---------------------------------------------------------------------------
// Deterministic random numbers (identical on every compiler / library)
// ---------------------------------------------------------------------------
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed) {}
  uint64_t next() {  // splitmix64
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  uint32_t below(uint32_t n) { return (uint32_t)(((next() >> 32) * (uint64_t)n) >> 32); }
  double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
  double normal() {
    double u1 = uniform(), u2 = uniform();
    if (u1 < 1e-300) u1 = 1e-300;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
  }
};

// ---------------------------------------------------------------------------
// MQAR data
//   tokens 0..K-1 are keys, K..K+Vv-1 are values
//   sequence: k v k v ... (N pairs) | k v k v ... (Q queries with answers)
//   the model must predict each answer value from the position of its query key
// ---------------------------------------------------------------------------
struct Batch {
  std::vector<int> tok;     // B*T input tokens
  std::vector<int> pos;     // B*Q row indices (into B*T) where a prediction is scored
  std::vector<int> target;  // B*Q correct value tokens
};

static void make_batch(const Config& c, Rng& r, Batch& b) {
  const int T = c.T();
  b.tok.assign((size_t)c.B * T, 0);
  b.pos.assign((size_t)c.B * c.Q, 0);
  b.target.assign((size_t)c.B * c.Q, 0);
  std::vector<int> perm(c.K), keys(c.N), vals(c.N);
  for (int s = 0; s < c.B; s++) {
    for (int i = 0; i < c.K; i++) perm[i] = i;
    for (int i = 0; i < c.N; i++) {  // N distinct keys (partial Fisher-Yates)
      int j = i + (int)r.below((uint32_t)(c.K - i));
      std::swap(perm[i], perm[j]);
      keys[i] = perm[i];
      vals[i] = c.K + (int)r.below((uint32_t)c.Vv);
    }
    int* t = &b.tok[(size_t)s * T];
    for (int i = 0; i < c.N; i++) { t[2 * i] = keys[i]; t[2 * i + 1] = vals[i]; }
    for (int q = 0; q < c.Q; q++) {
      int i = (int)r.below((uint32_t)c.N);
      int p = 2 * c.N + 2 * q;
      t[p] = keys[i];
      t[p + 1] = vals[i];
      b.pos[(size_t)s * c.Q + q] = s * T + p;
      b.target[(size_t)s * c.Q + q] = vals[i];
    }
  }
}

// ---------------------------------------------------------------------------
// Parameter layout: all parameters live in one flat array
// ---------------------------------------------------------------------------
struct LayerOff { size_t ln1g, ln1b, wq, wk, wv, wo, ln2g, ln2b, w1, b1, w2, b2; };
struct Tensor { std::string name; size_t off, n; };
struct Layout {
  size_t tok, posemb, prev, lnfg, lnfb, wout, total;
  std::vector<LayerOff> lay;
  std::vector<Tensor> tensors;
};

static Layout make_layout(const Config& c) {
  Layout lo;
  size_t o = 0;
  const size_t d = c.d, V = c.V(), T = c.T();
  auto take = [&](const std::string& name, size_t n) {
    size_t r = o; o += n; lo.tensors.push_back({name, r, n}); return r;
  };
  lo.tok = take("tok_emb", V * d);
  lo.posemb = take("pos_emb", T * d);
  lo.prev = c.shift ? take("prev_emb", V * d) : 0;
  for (int l = 0; l < c.L; l++) {
    std::string p = "L" + std::to_string(l) + ".";
    LayerOff f;
    f.ln1g = take(p + "ln1_g", d);  f.ln1b = take(p + "ln1_b", d);
    f.wq = take(p + "Wq", d * d);   f.wk = take(p + "Wk", d * d);
    f.wv = take(p + "Wv", d * d);   f.wo = take(p + "Wo", d * d);
    f.ln2g = take(p + "ln2_g", d);  f.ln2b = take(p + "ln2_b", d);
    f.w1 = take(p + "W1", d * 4 * d); f.b1 = take(p + "b1", 4 * d);
    f.w2 = take(p + "W2", 4 * d * d); f.b2 = take(p + "b2", d);
    lo.lay.push_back(f);
  }
  lo.lnfg = take("lnf_g", d);
  lo.lnfb = take("lnf_b", d);
  lo.wout = take("W_out", d * V);
  lo.total = o;
  return lo;
}

// ---------------------------------------------------------------------------
// Basic kernels (plain loops; these are the targets for later optimization)
// ---------------------------------------------------------------------------
// Y[M][N] = X[M][K] * W[K][N] (+ bias[N])
template <class F>
static void matmul(const F* X, const F* W, const F* bias, F* Y, int M, int K, int N) {
  for (int i = 0; i < M; i++) {
    F* y = Y + (size_t)i * N;
    for (int j = 0; j < N; j++) y[j] = bias ? bias[j] : F(0);
    for (int k = 0; k < K; k++) {
      const F xv = X[(size_t)i * K + k];
      const F* w = W + (size_t)k * N;
      for (int j = 0; j < N; j++) y[j] += xv * w[j];
    }
  }
}

// backward of matmul: dX += dY * W^T (if dX), dW += X^T * dY, dbias += colsum(dY) (if dbias)
template <class F>
static void matmul_back(const F* X, const F* W, const F* dY, F* dX, F* dW, F* dbias, int M, int K, int N) {
  for (int i = 0; i < M; i++) {
    const F* dy = dY + (size_t)i * N;
    if (dX) {
      for (int k = 0; k < K; k++) {
        const F* w = W + (size_t)k * N;
        F s = 0;
        for (int j = 0; j < N; j++) s += dy[j] * w[j];
        dX[(size_t)i * K + k] += s;
      }
    }
    for (int k = 0; k < K; k++) {
      const F xv = X[(size_t)i * K + k];
      F* dw = dW + (size_t)k * N;
      for (int j = 0; j < N; j++) dw[j] += xv * dy[j];
    }
    if (dbias)
      for (int j = 0; j < N; j++) dbias[j] += dy[j];
  }
}

// LayerNorm over rows of length d; saves normalized input xh and 1/std rs
template <class F>
static void layernorm(const F* x, const F* g, const F* b, F* xh, F* rs, F* y, int M, int d) {
  for (int i = 0; i < M; i++) {
    const F* xr = x + (size_t)i * d;
    F mean = 0;
    for (int j = 0; j < d; j++) mean += xr[j];
    mean /= d;
    F var = 0;
    for (int j = 0; j < d; j++) { F t = xr[j] - mean; var += t * t; }
    var /= d;
    F r = F(1) / std::sqrt(var + F(1e-5));
    rs[i] = r;
    for (int j = 0; j < d; j++) {
      F h = (xr[j] - mean) * r;
      xh[(size_t)i * d + j] = h;
      y[(size_t)i * d + j] = h * g[j] + b[j];
    }
  }
}

// dx += LayerNorm backward; dg, db accumulate
template <class F>
static void layernorm_back(const F* dy, const F* xh, const F* rs, const F* g, F* dx, F* dg, F* db, int M, int d) {
  for (int i = 0; i < M; i++) {
    const F* dyr = dy + (size_t)i * d;
    const F* h = xh + (size_t)i * d;
    F m1 = 0, m2 = 0;
    for (int j = 0; j < d; j++) {
      F dh = dyr[j] * g[j];
      dg[j] += dyr[j] * h[j];
      db[j] += dyr[j];
      m1 += dh;
      m2 += dh * h[j];
    }
    m1 /= d; m2 /= d;
    for (int j = 0; j < d; j++) {
      F dh = dyr[j] * g[j];
      dx[(size_t)i * d + j] += rs[i] * (dh - m1 - h[j] * m2);
    }
  }
}

// GELU (tanh approximation) and its derivative
template <class F> static inline F gelu(F x) {
  const F c = F(0.7978845608028654);
  return F(0.5) * x * (F(1) + std::tanh(c * (x + F(0.044715) * x * x * x)));
}
template <class F> static inline F gelu_grad(F x) {
  const F c = F(0.7978845608028654);
  F t = std::tanh(c * (x + F(0.044715) * x * x * x));
  return F(0.5) * (F(1) + t) + F(0.5) * x * (F(1) - t * t) * c * (F(1) + F(3 * 0.044715) * x * x);
}

// ---------------------------------------------------------------------------
// Transformer (template on F so the gradient check can run in double)
// ---------------------------------------------------------------------------
template <class F>
struct Net {
  Config c;
  Layout lo;
  int T, V, M, P;  // P = number of scored predictions per batch (B*Q)
  std::vector<F> w, g;

  struct Acts {  // values saved for the backward pass, one set per block
    std::vector<F> x, xh1, rs1, a1, q, k, v, P, att, xh2, rs2, a2, u, gu;
  };
  std::vector<Acts> acts;
  std::vector<F> xf, xhf, rsf, af, hsel, logits;   // final layer
  std::vector<F> dx, dh, dtmp, da, dq, dk, dv, du, dsel, dlog, tmp, attrow;  // scratch

  explicit Net(const Config& cfg) : c(cfg), lo(make_layout(cfg)) {
    T = c.T(); V = c.V(); M = c.B * T; P = c.B * c.Q;
    const size_t d = c.d, Md = (size_t)M * d;
    w.assign(lo.total, F(0));
    g.assign(lo.total, F(0));
    acts.resize(c.L);
    for (auto& A : acts) {
      A.x.resize(Md); A.xh1.resize(Md); A.rs1.resize(M); A.a1.resize(Md);
      A.q.resize(Md); A.k.resize(Md); A.v.resize(Md);
      A.P.resize((size_t)c.B * c.H * T * T); A.att.resize(Md);
      A.xh2.resize(Md); A.rs2.resize(M); A.a2.resize(Md);
      A.u.resize(Md * 4); A.gu.resize(Md * 4);
    }
    xf.resize(Md); xhf.resize(Md); rsf.resize(M); af.resize(Md);
    hsel.resize((size_t)P * d); logits.resize((size_t)P * V);
    dx.resize(Md); dh.resize(Md); dtmp.resize(Md); da.resize(Md);
    dq.resize(Md); dk.resize(Md); dv.resize(Md); du.resize(Md * 4);
    dsel.resize((size_t)P * d); dlog.resize((size_t)P * V);
    tmp.resize(Md * 4); attrow.resize(T);
  }

  size_t activation_bytes() const {
    size_t n = 0;
    for (auto& A : acts)
      n += A.x.size() + A.xh1.size() + A.rs1.size() + A.a1.size() + A.q.size() + A.k.size() + A.v.size() +
           A.P.size() + A.att.size() + A.xh2.size() + A.rs2.size() + A.a2.size() + A.u.size() + A.gu.size();
    return n * sizeof(F);
  }

  void init(uint64_t seed) {
    Rng r(seed);
    const F sd = F(0.02), sd_out = F(0.02 / std::sqrt(2.0 * c.L));
    auto fill = [&](size_t off, size_t n, F s) { for (size_t i = 0; i < n; i++) w[off + i] = F(r.normal()) * s; };
    auto ones = [&](size_t off, size_t n) { for (size_t i = 0; i < n; i++) w[off + i] = F(1); };
    const size_t d = c.d;
    fill(lo.tok, (size_t)V * d, sd);
    fill(lo.posemb, (size_t)T * d, sd);
    if (c.shift) fill(lo.prev, (size_t)V * d, sd);
    for (auto& f : lo.lay) {
      ones(f.ln1g, d); ones(f.ln2g, d);
      fill(f.wq, d * d, sd); fill(f.wk, d * d, sd); fill(f.wv, d * d, sd);
      fill(f.wo, d * d, sd_out);
      fill(f.w1, d * 4 * d, sd); fill(f.w2, 4 * d * d, sd_out);
    }
    ones(lo.lnfg, d);
    fill(lo.wout, d * (size_t)V, sd);
  }

  // causal multi-head self-attention: out = softmax(q k^T / sqrt(hd)) v
  void attention(const F* q, const F* k, const F* v, F* Pm, F* out) {
    const int d = c.d, H = c.H, hd = d / H;
    const F scale = F(1) / std::sqrt(F(hd));
    for (int b = 0; b < c.B; b++)
      for (int h = 0; h < H; h++)
        for (int i = 0; i < T; i++) {
          const F* qi = q + (size_t)(b * T + i) * d + h * hd;
          F* pr = Pm + ((size_t)(b * H + h) * T + i) * T;
          F mx = -1e30f;
          for (int j = 0; j <= i; j++) {
            const F* kj = k + (size_t)(b * T + j) * d + h * hd;
            F s = 0;
            for (int e = 0; e < hd; e++) s += qi[e] * kj[e];
            s *= scale;
            pr[j] = s;
            mx = std::max(mx, s);
          }
          F sum = 0;
          for (int j = 0; j <= i; j++) { pr[j] = std::exp(pr[j] - mx); sum += pr[j]; }
          for (int j = 0; j <= i; j++) pr[j] /= sum;
          for (int j = i + 1; j < T; j++) pr[j] = 0;
          F* o = out + (size_t)(b * T + i) * d + h * hd;
          for (int e = 0; e < hd; e++) o[e] = 0;
          for (int j = 0; j <= i; j++) {
            const F* vj = v + (size_t)(b * T + j) * d + h * hd;
            for (int e = 0; e < hd; e++) o[e] += pr[j] * vj[e];
          }
        }
  }

  // attention backward: given dout, accumulate dq, dk, dv (all zeroed by caller)
  void attention_back(const F* q, const F* k, const F* v, const F* Pm, const F* dout, F* dq_, F* dk_, F* dv_) {
    const int d = c.d, H = c.H, hd = d / H;
    const F scale = F(1) / std::sqrt(F(hd));
    F* dp = attrow.data();
    for (int b = 0; b < c.B; b++)
      for (int h = 0; h < H; h++)
        for (int i = 0; i < T; i++) {
          const F* pr = Pm + ((size_t)(b * H + h) * T + i) * T;
          const F* doi = dout + (size_t)(b * T + i) * d + h * hd;
          F dot = 0;
          for (int j = 0; j <= i; j++) {
            const F* vj = v + (size_t)(b * T + j) * d + h * hd;
            F* dvj = dv_ + (size_t)(b * T + j) * d + h * hd;
            F s = 0;
            for (int e = 0; e < hd; e++) { s += doi[e] * vj[e]; dvj[e] += pr[j] * doi[e]; }
            dp[j] = s;
            dot += pr[j] * s;
          }
          const F* qi = q + (size_t)(b * T + i) * d + h * hd;
          F* dqi = dq_ + (size_t)(b * T + i) * d + h * hd;
          for (int j = 0; j <= i; j++) {
            F ds = pr[j] * (dp[j] - dot) * scale;
            const F* kj = k + (size_t)(b * T + j) * d + h * hd;
            F* dkj = dk_ + (size_t)(b * T + j) * d + h * hd;
            for (int e = 0; e < hd; e++) { dqi[e] += ds * kj[e]; dkj[e] += ds * qi[e]; }
          }
        }
  }

  // forward pass; returns mean cross-entropy over the B*Q scored positions.
  // If correct != nullptr, also counts argmax hits.
  double forward(const Batch& bt, int* correct = nullptr) {
    const int d = c.d, d4 = 4 * d;
    const F* W = w.data();
    // embeddings
    F* x = acts[0].x.data();
    for (int m = 0; m < M; m++) {
      const F* te = W + lo.tok + (size_t)bt.tok[m] * d;
      const F* pe = W + lo.posemb + (size_t)(m % T) * d;
      for (int j = 0; j < d; j++) x[(size_t)m * d + j] = te[j] + pe[j];
      if (c.shift && m % T > 0) {  // token shift: add embedding of the previous token
        const F* pv = W + lo.prev + (size_t)bt.tok[m - 1] * d;
        for (int j = 0; j < d; j++) x[(size_t)m * d + j] += pv[j];
      }
    }
    for (int l = 0; l < c.L; l++) {
      Acts& A = acts[l];
      const LayerOff& f = lo.lay[l];
      F* xn = (l + 1 < c.L) ? acts[l + 1].x.data() : xf.data();
      layernorm(A.x.data(), W + f.ln1g, W + f.ln1b, A.xh1.data(), A.rs1.data(), A.a1.data(), M, d);
      matmul(A.a1.data(), W + f.wq, (const F*)nullptr, A.q.data(), M, d, d);
      matmul(A.a1.data(), W + f.wk, (const F*)nullptr, A.k.data(), M, d, d);
      matmul(A.a1.data(), W + f.wv, (const F*)nullptr, A.v.data(), M, d, d);
      attention(A.q.data(), A.k.data(), A.v.data(), A.P.data(), A.att.data());
      matmul(A.att.data(), W + f.wo, (const F*)nullptr, tmp.data(), M, d, d);
      for (size_t i = 0; i < (size_t)M * d; i++) xn[i] = A.x[i] + tmp[i];  // h = x + attn
      layernorm(xn, W + f.ln2g, W + f.ln2b, A.xh2.data(), A.rs2.data(), A.a2.data(), M, d);
      matmul(A.a2.data(), W + f.w1, W + f.b1, A.u.data(), M, d, d4);
      for (size_t i = 0; i < (size_t)M * d4; i++) A.gu[i] = gelu(A.u[i]);
      matmul(A.gu.data(), W + f.w2, W + f.b2, tmp.data(), M, d4, d);
      for (size_t i = 0; i < (size_t)M * d; i++) xn[i] += tmp[i];  // x_next = h + mlp
    }
    layernorm(xf.data(), W + lo.lnfg, W + lo.lnfb, xhf.data(), rsf.data(), af.data(), M, d);
    // output projection only at the scored (query) positions
    for (int p = 0; p < P; p++)
      std::memcpy(&hsel[(size_t)p * d], &af[(size_t)bt.pos[p] * d], sizeof(F) * d);
    matmul(hsel.data(), W + lo.wout, (const F*)nullptr, logits.data(), P, d, V);
    double loss = 0;
    int hits = 0;
    for (int p = 0; p < P; p++) {
      F* lg = &logits[(size_t)p * V];
      F mx = lg[0];
      int arg = 0;
      for (int j = 1; j < V; j++) if (lg[j] > mx) { mx = lg[j]; arg = j; }
      F sum = 0;
      for (int j = 0; j < V; j++) { lg[j] = std::exp(lg[j] - mx); sum += lg[j]; }
      for (int j = 0; j < V; j++) lg[j] /= sum;  // logits now hold probabilities
      loss -= std::log(std::max((double)lg[bt.target[p]], 1e-30));
      if (arg == bt.target[p]) hits++;
    }
    if (correct) *correct = hits;
    return loss / P;
  }

  // backward pass for the batch used in the last forward(); fills g
  void backward(const Batch& bt) {
    const int d = c.d, d4 = 4 * d;
    const size_t Md = (size_t)M * d;
    const F* W = w.data();
    F* G = g.data();
    std::fill(g.begin(), g.end(), F(0));
    // softmax cross-entropy gradient
    for (int p = 0; p < P; p++) {
      for (int j = 0; j < V; j++) dlog[(size_t)p * V + j] = logits[(size_t)p * V + j] / P;
      dlog[(size_t)p * V + bt.target[p]] -= F(1) / P;
    }
    std::fill(dsel.begin(), dsel.end(), F(0));
    matmul_back(hsel.data(), W + lo.wout, dlog.data(), dsel.data(), G + lo.wout, (F*)nullptr, P, d, V);
    std::fill(da.begin(), da.end(), F(0));
    for (int p = 0; p < P; p++)
      for (int j = 0; j < d; j++) da[(size_t)bt.pos[p] * d + j] += dsel[(size_t)p * d + j];
    std::fill(dx.begin(), dx.end(), F(0));
    layernorm_back(da.data(), xhf.data(), rsf.data(), W + lo.lnfg, dx.data(), G + lo.lnfg, G + lo.lnfb, M, d);

    for (int l = c.L - 1; l >= 0; l--) {
      Acts& A = acts[l];
      const LayerOff& f = lo.lay[l];
      // MLP: x_next = h + W2 * gelu(W1 * LN2(h) + b1) + b2
      std::fill(tmp.begin(), tmp.begin() + Md * 4, F(0));
      matmul_back(A.gu.data(), W + f.w2, dx.data(), tmp.data(), G + f.w2, G + f.b2, M, d4, d);
      for (size_t i = 0; i < Md * 4; i++) du[i] = tmp[i] * gelu_grad(A.u[i]);
      std::fill(da.begin(), da.end(), F(0));
      matmul_back(A.a2.data(), W + f.w1, du.data(), da.data(), G + f.w1, G + f.b1, M, d, d4);
      dh = dx;  // residual path
      layernorm_back(da.data(), A.xh2.data(), A.rs2.data(), W + f.ln2g, dh.data(), G + f.ln2g, G + f.ln2b, M, d);
      // attention: h = x + Wo * attn(LN1(x))
      std::fill(dtmp.begin(), dtmp.end(), F(0));
      matmul_back(A.att.data(), W + f.wo, dh.data(), dtmp.data(), G + f.wo, (F*)nullptr, M, d, d);
      std::fill(dq.begin(), dq.end(), F(0));
      std::fill(dk.begin(), dk.end(), F(0));
      std::fill(dv.begin(), dv.end(), F(0));
      attention_back(A.q.data(), A.k.data(), A.v.data(), A.P.data(), dtmp.data(), dq.data(), dk.data(), dv.data());
      std::fill(da.begin(), da.end(), F(0));
      matmul_back(A.a1.data(), W + f.wq, dq.data(), da.data(), G + f.wq, (F*)nullptr, M, d, d);
      matmul_back(A.a1.data(), W + f.wk, dk.data(), da.data(), G + f.wk, (F*)nullptr, M, d, d);
      matmul_back(A.a1.data(), W + f.wv, dv.data(), da.data(), G + f.wv, (F*)nullptr, M, d, d);
      dx = dh;  // residual path
      layernorm_back(da.data(), A.xh1.data(), A.rs1.data(), W + f.ln1g, dx.data(), G + f.ln1g, G + f.ln1b, M, d);
    }
    // embeddings
    for (int m = 0; m < M; m++) {
      F* gt = G + lo.tok + (size_t)bt.tok[m] * d;
      F* gp = G + lo.posemb + (size_t)(m % T) * d;
      for (int j = 0; j < d; j++) { gt[j] += dx[(size_t)m * d + j]; gp[j] += dx[(size_t)m * d + j]; }
      if (c.shift && m % T > 0) {
        F* gv = G + lo.prev + (size_t)bt.tok[m - 1] * d;
        for (int j = 0; j < d; j++) gv[j] += dx[(size_t)m * d + j];
      }
    }
  }
};

// ---------------------------------------------------------------------------
// Adam optimizer
// ---------------------------------------------------------------------------
struct Adam {
  std::vector<float> m, v;
  int t = 0;
  explicit Adam(size_t n) : m(n, 0.f), v(n, 0.f) {}
  void step(std::vector<float>& w, const std::vector<float>& g, float lr) {
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    t++;
    const float c1 = 1.f - std::pow(b1, (float)t), c2 = 1.f - std::pow(b2, (float)t);
    for (size_t i = 0; i < w.size(); i++) {
      m[i] = b1 * m[i] + (1 - b1) * g[i];
      v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
      w[i] -= lr * (m[i] / c1) / (std::sqrt(v[i] / c2) + eps);
    }
  }
};

// ---------------------------------------------------------------------------
// Gradient check: analytic vs central finite differences, in double precision
// ---------------------------------------------------------------------------
static bool gradient_check(const Config& c) {
  Net<double> net(c);
  net.init(c.seed);
  Rng r(c.seed ^ 0x5EEDULL);
  // larger random weights than training init, so every term has a sizable gradient
  for (size_t i = 0; i < net.w.size(); i++) net.w[i] += 0.3 * r.normal();
  Batch b;
  make_batch(c, r, b);
  net.forward(b);
  net.backward(b);
  const double eps = 1e-5;
  double worst = 0;
  std::string worst_name;
  int checked = 0;
  for (const Tensor& t : net.lo.tensors) {
    for (int s = 0; s < 3; s++) {
      size_t i = t.off + r.below((uint32_t)t.n);
      double a = net.g[i], keep = net.w[i];
      net.w[i] = keep + eps; double lp = net.forward(b);
      net.w[i] = keep - eps; double lm = net.forward(b);
      net.w[i] = keep;
      double num = (lp - lm) / (2 * eps);
      double rel = std::fabs(a - num) / std::max(1e-7, std::fabs(a) + std::fabs(num));
      if (rel > worst) { worst = rel; worst_name = t.name; }
      checked++;
    }
  }
  bool ok = worst < 1e-4;
  std::printf("gradcheck: %d parameters checked, max relative error %.3e (%s) -> %s\n",
              checked, worst, worst_name.c_str(), ok ? "PASS" : "FAIL");
  return ok;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

int main(int argc, char** argv) {
  Config c;
  if (!parse_args(argc, argv, c)) { usage(); return 1; }
  auto t_start = Clock::now();

  const int T = c.T(), V = c.V();
  std::printf("== MQAR Transformer training (serial baseline) ==\n");
  std::printf("task : pairs N=%d queries Q=%d keys K=%d values Vv=%d -> T=%d V=%d\n", c.N, c.Q, c.K, c.Vv, T, V);
  std::printf("model: layers L=%d dim d=%d heads H=%d (head dim %d), MLP width %d, token shift %s\n", c.L, c.d, c.H,
              c.d / c.H, 4 * c.d, c.shift ? "on" : "off");
  std::printf("train: batch B=%d steps=%d lr=%g warmup=%d clip=%g seed=%llu%s\n", c.B, c.steps, c.lr, c.warmup,
              c.clip, (unsigned long long)c.seed, c.overfit ? " [OVERFIT: one fixed batch]" : "");

  if (c.gradcheck) {
    auto t0 = Clock::now();
    bool ok = gradient_check(c);
    std::printf("gradcheck time: %.3f s\n", secs(t0, Clock::now()));
    if (!ok) return 2;
    if (c.steps == 0) return 0;
  }

  Net<float> net(c);
  net.init(c.seed);
  Adam opt(net.w.size());
  const size_t nparam = net.w.size();
  std::printf("params: %zu (%.3f M); weights+grads+Adam: %.1f MiB; activations: %.1f MiB\n", nparam, nparam / 1e6,
              4.0 * nparam * sizeof(float) / 1048576.0, net.activation_bytes() / 1048576.0);

  // counted FLOPs per step (matrix products + attention; forward x3 for fwd+bwd)
  const double d = c.d, Bd = c.B;
  const double fwd = Bd * (c.L * (2.0 * 12.0 * d * d * T + 2.0 * d * T * (T + 1.0)) + 2.0 * d * V * c.Q);
  const double flops_step = 3.0 * fwd;
  std::printf("work : %.3f GFLOP per step (estimate), %d tokens per step\n", flops_step / 1e9, c.B * T);

  Rng train_rng(c.seed * 0x9E3779B97F4A7C15ULL + 1);
  Rng eval_rng(c.seed * 0x9E3779B97F4A7C15ULL + 2);
  std::vector<Batch> evalset(c.eval_batches);
  for (auto& b : evalset) make_batch(c, eval_rng, b);
  Batch batch;
  make_batch(c, train_rng, batch);

  auto evaluate = [&](double& loss, double& acc) {
    double l = 0;
    long hits = 0;
    for (auto& b : evalset) { int h = 0; l += net.forward(b, &h); hits += h; }
    loss = l / std::max(1, (int)evalset.size());
    acc = (double)hits / std::max<long>(1, (long)evalset.size() * c.B * c.Q);
  };

  const double chance = 1.0 / c.Vv;
  std::printf("chance accuracy: %.4f\n", chance);
  std::printf("%8s %12s %12s %12s %12s %10s\n", "step", "train_loss", "train_acc", "eval_loss", "eval_acc", "time_s");

  double t_gen = 0, t_train = 0, t_eval = 0;
  double loss_acc = 0, acc_acc = 0;
  int n_acc = 0;
  double last_eval_loss = 0, last_eval_acc = 0, last_train_loss = 0;
  if (c.eval_batches > 0) {
    auto e0 = Clock::now();
    evaluate(last_eval_loss, last_eval_acc);
    t_eval += secs(e0, Clock::now());
    std::printf("%8d %12s %12s %12.4f %12.4f %10.2f\n", 0, "-", "-", last_eval_loss, last_eval_acc, 0.0);
  }

  for (int step = 1; step <= c.steps; step++) {
    auto g0 = Clock::now();
    if (!c.overfit) make_batch(c, train_rng, batch);
    auto t0 = Clock::now();
    t_gen += secs(g0, t0);

    // ---- main computation: forward, backward, optimizer ----
    int hits = 0;
    double loss = net.forward(batch, &hits);
    net.backward(batch);
    if (c.clip > 0) {
      double nrm = 0;
      for (float x : net.g) nrm += (double)x * x;
      nrm = std::sqrt(nrm);
      if (nrm > c.clip) { float s = (float)(c.clip / nrm); for (auto& x : net.g) x *= s; }
    }
    double lr = c.lr;
    if (step <= c.warmup) lr = c.lr * step / std::max(1, c.warmup);
    else if (c.steps > c.warmup) {  // cosine decay to 10% of peak
      double p = (double)(step - c.warmup) / (c.steps - c.warmup);
      lr = c.lr * (0.1 + 0.9 * 0.5 * (1 + std::cos(3.141592653589793 * p)));
    }
    opt.step(net.w, net.g, (float)lr);
    t_train += secs(t0, Clock::now());
    // --------------------------------------------------------

    loss_acc += loss; acc_acc += (double)hits / (c.B * c.Q); n_acc++;
    last_train_loss = loss;
    if (step % c.eval_every == 0 || step == c.steps) {
      auto e0 = Clock::now();
      if (c.eval_batches > 0) evaluate(last_eval_loss, last_eval_acc);
      t_eval += secs(e0, Clock::now());
      std::printf("%8d %12.4f %12.4f %12.4f %12.4f %10.2f\n", step, loss_acc / n_acc, acc_acc / n_acc,
                  last_eval_loss, last_eval_acc, t_train);
      std::fflush(stdout);
      loss_acc = acc_acc = 0; n_acc = 0;
    }
  }

  double wsum = 0;
  for (float x : net.w) wsum += std::fabs((double)x);
  const double total = secs(t_start, Clock::now());
  const double tokens = (double)c.steps * c.B * T;
  std::printf("== summary ==\n");
  std::printf("final_train_loss %.6f\n", last_train_loss);
  std::printf("final_eval_loss  %.6f\n", last_eval_loss);
  std::printf("final_eval_acc   %.4f (chance %.4f)\n", last_eval_acc, chance);
  std::printf("weight_checksum  %.6f\n", wsum);
  std::printf("train_time_s     %.3f   (forward+backward+Adam, main timer)\n", t_train);
  std::printf("datagen_time_s   %.3f\n", t_gen);
  std::printf("eval_time_s      %.3f\n", t_eval);
  std::printf("total_time_s     %.3f\n", total);
  if (c.steps > 0 && t_train > 0) {
    std::printf("time_per_step_s  %.4f\n", t_train / c.steps);
    std::printf("tokens_per_s     %.1f\n", tokens / t_train);
    std::printf("gflops_est       %.3f\n", flops_step * c.steps / t_train / 1e9);
  }
  return 0;
}