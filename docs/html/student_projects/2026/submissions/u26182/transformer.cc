// Serial decoder-only Transformer benchmark; fixed random weights, no training.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
struct Config {
    int dim = 256, heads = 4, ff = 1024, layers = 2, vocab = 64;
    int prompt = 128, generate = 8, repeats = 3;
    uint32_t seed = 42;
    bool check = false;
};
void usage() {
    std::cout << "Usage: ./transformer [options]\n"
              << "  --dim N       model width (256)\n"
              << "  --heads N     attention heads; must divide width (4)\n"
              << "  --ff N        feed-forward width (1024)\n"
              << "  --layers N    decoder blocks (2)\n"
              << "  --vocab N     token vocabulary size (64)\n"
              << "  --prompt N    input token count (128)\n"
              << "  --generate N  generated tokens; 0 means prefill only (8)\n"
              << "  --repeats N   timed requests after one warm-up (3)\n"
              << "  --seed N      deterministic uint32 seed (42)\n"
              << "  --check       run untimed correctness checks\n";
}
Config parse(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--check") { c.check = true; continue; }
        int* target = nullptr;
        int limit = 8192;
        if (key == "--dim") { target = &c.dim; limit = 2048; }
        else if (key == "--heads") { target = &c.heads; limit = 2048; }
        else if (key == "--ff") target = &c.ff;
        else if (key == "--layers") { target = &c.layers; limit = 32; }
        else if (key == "--vocab") { target = &c.vocab; limit = 4096; }
        else if (key == "--prompt") target = &c.prompt;
        else if (key == "--generate") target = &c.generate;
        else if (key == "--repeats") { target = &c.repeats; limit = 1000000; }
        else if (key != "--seed") throw std::runtime_error("unknown option: " + key);
        if (++i == argc) throw std::runtime_error("missing value for " + key);
        const std::string text = argv[i];
        if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
            throw std::runtime_error("expected nonnegative integer for " + key);
        const auto value = std::stoull(text);
        if (key == "--seed") {
            if (value > UINT32_MAX) throw std::runtime_error("seed exceeds uint32 range");
            c.seed = static_cast<uint32_t>(value);
        } else {
            if (value > static_cast<unsigned>(limit) || (value == 0 && key != "--generate"))
                throw std::runtime_error("out-of-range value for " + key);
            *target = static_cast<int>(value);
        }
    }
    if (c.dim % c.heads != 0) throw std::runtime_error("heads must divide dim");
    if (c.prompt + c.generate > 8192) throw std::runtime_error("prompt + generate must be <= 8192");
    return c;
}

template<class T> using Array = std::vector<T>;
template<class T> struct Block {
    Array<T> q, k, v, out, up, down;
};
template<class T> struct Model {
    Array<T> embedding, position, head;
    std::vector<Block<T>> blocks;
};

Array<float> weights(size_t count, float scale, std::mt19937& rng) {
    Array<float> a(count);
    for (float& x : a) {
        const float u = static_cast<float>(rng() >> 8) / 16777216.0f;
        x = (2.0f * u - 1.0f) * scale;
    }
    return a;
}
Model<float> make_model(const Config& c) {
    std::mt19937 rng(c.seed);
    Model<float> m;
    m.embedding = weights(size_t(c.vocab) * c.dim, 0.5f, rng);
    // Fixed sinusoidal positions: extending context does not change the weights.
    m.position.resize(size_t(c.prompt + c.generate) * c.dim);
    for (int t = 0; t < c.prompt + c.generate; ++t)
        for (int j = 0; j < c.dim; ++j) {
            const double angle = t / std::pow(10000.0, 2.0 * (j / 2) / c.dim);
            m.position[size_t(t) * c.dim + j] = static_cast<float>(
                0.1 * (j % 2 == 0 ? std::sin(angle) : std::cos(angle)));
        }
    const float scale = std::sqrt(3.0f / c.dim);
    for (int l = 0; l < c.layers; ++l) {
        Block<float> b;
        b.q = weights(size_t(c.dim) * c.dim, scale, rng);
        b.k = weights(size_t(c.dim) * c.dim, scale, rng);
        b.v = weights(size_t(c.dim) * c.dim, scale, rng);
        b.out = weights(size_t(c.dim) * c.dim, scale, rng);
        b.up = weights(size_t(c.ff) * c.dim, scale, rng);
        b.down = weights(size_t(c.dim) * c.ff, std::sqrt(3.0f / c.ff), rng);
        m.blocks.push_back(std::move(b));
    }
    m.head = weights(size_t(c.vocab) * c.dim, scale, rng);
    return m;
}

Model<double> double_model(const Model<float>& m) {
    auto copy = [](const Array<float>& a) { return Array<double>(a.begin(), a.end()); };
    Model<double> d;
    d.embedding = copy(m.embedding); d.position = copy(m.position); d.head = copy(m.head);
    for (const auto& b : m.blocks)
        d.blocks.push_back({copy(b.q), copy(b.k), copy(b.v), copy(b.out), copy(b.up), copy(b.down)});
    return d;
}

// All matrices are row-major; weight rows correspond to output features.
template<class T>
void linear(const T* x, const Array<T>& w, T* y, int rows, int in, int out) {
    for (int t = 0; t < rows; ++t)
        for (int j = 0; j < out; ++j) {
            T sum = 0;
            for (int i = 0; i < in; ++i) sum += x[size_t(t) * in + i] * w[size_t(j) * in + i];
            y[size_t(t) * out + j] = sum;
        }
}
template<class T> void normalize(const T* x, T* y, int rows, int dim) {
    // LayerNorm with fixed gain=1 and bias=0.
    for (int t = 0; t < rows; ++t) {
        const T* row = x + size_t(t) * dim;
        T mean = 0, variance = 0;
        for (int j = 0; j < dim; ++j) mean += row[j];
        mean /= dim;
        for (int j = 0; j < dim; ++j) variance += (row[j] - mean) * (row[j] - mean);
        const T inv = T(1) / std::sqrt(variance / dim + T(1e-5));
        for (int j = 0; j < dim; ++j) y[size_t(t) * dim + j] = (row[j] - mean) * inv;
    }
}
template<class T> void softmax(T* a, int n) {
    const T largest = *std::max_element(a, a + n);
    T sum = 0;
    for (int i = 0; i < n; ++i) { a[i] = std::exp(a[i] - largest); sum += a[i]; }
    for (int i = 0; i < n; ++i) a[i] /= sum;
}
template<class T>
void attention(const T* q, const T* k, const T* v, T* out, T* scores,
               int rows, int dim, int heads) {
    const int hd = dim / heads;
    const T scale = T(1) / std::sqrt(T(hd));
    for (int t = 0; t < rows; ++t)
        for (int h = 0; h < heads; ++h) {
            // Causal mask: position t can see exactly positions 0..t.
            for (int s = 0; s <= t; ++s) {
                T dot = 0;
                for (int j = 0; j < hd; ++j)
                    dot += q[size_t(t) * dim + h * hd + j] * k[size_t(s) * dim + h * hd + j];
                scores[s] = dot * scale;
            }
            softmax(scores, t + 1);
            for (int j = 0; j < hd; ++j) {
                T sum = 0;
                for (int s = 0; s <= t; ++s) sum += scores[s] * v[size_t(s) * dim + h * hd + j];
                out[size_t(t) * dim + h * hd + j] = sum;
            }
        }
}
template<class T> struct Scratch {
    Array<T> x, norm, q, k, v, attn, tmp, ff, scores, probabilities;
    Scratch(const Config& c, int rows)
        : x(size_t(rows) * c.dim), norm(x.size()), q(x.size()), k(x.size()),
          v(x.size()), attn(x.size()), tmp(x.size()), ff(size_t(rows) * c.ff),
          scores(rows), probabilities(c.vocab) {}
};
template<class T>
void forward(const Model<T>& m, const Config& c, const std::vector<int>& tokens, Scratch<T>& s) {
    const int n = static_cast<int>(tokens.size()), d = c.dim;
    for (int t = 0; t < n; ++t)
        for (int j = 0; j < d; ++j)
            s.x[size_t(t) * d + j] = m.embedding[size_t(tokens[t]) * d + j] + m.position[size_t(t) * d + j];
    for (const auto& b : m.blocks) {
        normalize(s.x.data(), s.norm.data(), n, d);
        linear(s.norm.data(), b.q, s.q.data(), n, d, d);
        linear(s.norm.data(), b.k, s.k.data(), n, d, d);
        linear(s.norm.data(), b.v, s.v.data(), n, d, d);
        attention(s.q.data(), s.k.data(), s.v.data(), s.attn.data(), s.scores.data(), n, d, c.heads);
        linear(s.attn.data(), b.out, s.tmp.data(), n, d, d);
        for (size_t i = 0; i < size_t(n) * d; ++i) s.x[i] += s.tmp[i];
        normalize(s.x.data(), s.norm.data(), n, d);
        linear(s.norm.data(), b.up, s.ff.data(), n, d, c.ff);
        // Exact GELU, rather than a tanh approximation.
        for (size_t i = 0; i < size_t(n) * c.ff; ++i) {
            const T a = s.ff[i];
            s.ff[i] = T(0.5) * a * (T(1) + std::erf(a / std::sqrt(T(2))));
        }
        linear(s.ff.data(), b.down, s.tmp.data(), n, c.ff, d);
        for (size_t i = 0; i < size_t(n) * d; ++i) s.x[i] += s.tmp[i];
    }
    normalize(s.x.data(), s.norm.data(), n, d);
    // Generation only needs the next-token distribution at the last position.
    linear(s.norm.data() + size_t(n - 1) * d, m.head, s.probabilities.data(), 1, d, c.vocab);
    softmax(s.probabilities.data(), c.vocab);
}

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
double max_error(const float* a, const double* b, size_t n) {
    double error = 0;
    for (size_t i = 0; i < n; ++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite check output");
        error = std::max(error, std::abs(double(a[i]) - b[i]));
    }
    return error;
}

void check_attention() {
    // Independent, explicitly materialized double attention matrices.
    constexpr int n = 5, d = 6, heads = 2, hd = d / heads;
    Array<float> q(n*d), k(n*d), v(n*d), out(n*d), scores(n);
    for (int i = 0; i < n*d; ++i) {
        q[i] = float(i % 7 - 3) / 7; k[i] = float(i % 11 - 5) / 5;
        v[i] = float(i % 13 - 6) / 3;
    }
    attention(q.data(), k.data(), v.data(), out.data(), scores.data(), n, d, heads);
    Array<double> expected(n*d, 0), matrix(heads*n*n, 0);
    for (int h = 0; h < heads; ++h)
        for (int t = 0; t < n; ++t) {
            double sum = 0;
            for (int s = 0; s < n; ++s) {
                if (s > t) continue;
                double dot = 0;
                for (int j = h*hd; j < (h+1)*hd; ++j) dot += double(q[t*d+j]) * k[s*d+j];
                // Small bounded test values make direct exp safe here.
                matrix[(h*n+t)*n+s] = std::exp(dot / std::sqrt(double(hd)));
                sum += matrix[(h*n+t)*n+s];
            }
            for (int s = 0; s < n; ++s) matrix[(h*n+t)*n+s] /= sum;
        }
    for (int t = 0; t < n; ++t)
        for (int j = 0; j < d; ++j)
            for (int s = 0; s < n; ++s)
                expected[t*d+j] += matrix[((j/hd)*n+t)*n+s] * v[s*d+j];
    const double error = max_error(out.data(), expected.data(), out.size());
    require(error < 1e-6, "independent attention reference failed");
    // Q=K=0 gives exactly uniform weights over the visible prefix.
    std::fill(q.begin(), q.end(), 0); std::fill(k.begin(), k.end(), 0);
    attention(q.data(), k.data(), v.data(), out.data(), scores.data(), n, d, heads);
    for (int t = 0; t < n; ++t)
        for (int j = 0; j < d; ++j) {
            double sum = 0;
            for (int s = 0; s <= t; ++s) sum += v[s*d+j];
            expected[t*d+j] = sum / (t+1);
        }
    const double uniform_error = max_error(out.data(), expected.data(), out.size());
    require(uniform_error < 1e-6, "uniform attention check failed");
    std::cout << "attention_reference_max_abs_error=" << error
              << " uniform_attention_max_abs_error=" << uniform_error << '\n';
}

void checks(const Model<float>& m, const Config& c, const std::vector<int>& prompt) {
    check_attention();
    // Keep full-model checks short even for long benchmark contexts.
    const int n = std::min(c.prompt, 8);
    std::vector<int> tokens(prompt.begin(), prompt.begin() + n);
    Scratch<float> f(c, n);
    Scratch<double> d(c, n);
    const auto reference = double_model(m);
    forward(m, c, tokens, f); forward(reference, c, tokens, d);
    const double hidden_error = max_error(f.norm.data(), d.norm.data(), size_t(n)*c.dim);
    const double probability_error = max_error(f.probabilities.data(), d.probabilities.data(), c.vocab);
    require(hidden_error < 2e-3 && probability_error < 2e-4, "float/double forward check failed");
    const auto full = f.norm;
    const int prefix = std::max(1, n / 2);
    for (int i = prefix; i < n; ++i) tokens[i] = (tokens[i] + 1) % c.vocab;
    forward(m, c, tokens, f);
    for (size_t i = 0; i < size_t(prefix)*c.dim; ++i)
        require(std::isfinite(f.norm[i]) && std::abs(f.norm[i]-full[i]) < 1e-6f,
                "future tokens changed an earlier output");
    tokens.resize(prefix);
    forward(m, c, tokens, f);
    for (size_t i = 0; i < size_t(prefix)*c.dim; ++i)
        require(std::isfinite(f.norm[i]) && std::abs(f.norm[i]-full[i]) < 1e-6f,
                "prefix invariance check failed");
    std::cout << "forward_hidden_max_abs_error=" << hidden_error
              << " forward_probability_max_abs_error=" << probability_error << '\n'
              << "checks=PASS checked_prefix_tokens=" << n << '\n';
}

struct Result {
    double prefill = 0, decode = 0, checksum = 0, entropy = 0;
};
Result request(const Model<float>& m, const Config& c, const std::vector<int>& prompt,
               std::vector<int>& tokens, Scratch<float>& scratch, Array<float>& outputs) {
    tokens.assign(prompt.begin(), prompt.end());
    Result r;
    const int calls = std::max(1, c.generate);
    for (int step = 0; step < calls; ++step) {
        const auto start = Clock::now();
        forward(m, c, tokens, scratch); // Deliberately recompute the entire prefix.
        if (c.generate > 0)
            tokens.push_back(static_cast<int>(std::max_element(scratch.probabilities.begin(),
                scratch.probabilities.end()) - scratch.probabilities.begin()));
        const double elapsed = seconds(start);
        if (step == 0) r.prefill += elapsed; else r.decode += elapsed;
        std::copy(scratch.probabilities.begin(), scratch.probabilities.end(),
                  outputs.begin() + size_t(step)*c.vocab);
    }
    // Validation is outside the timers and observes every step's probabilities.
    for (int step = 0; step < calls; ++step) {
        double sum = 0;
        for (int j = 0; j < c.vocab; ++j) {
            const double p = outputs[size_t(step)*c.vocab+j];
            require(std::isfinite(p) && p >= 0 && p <= 1, "invalid probability");
            sum += p;
            r.checksum += p * (j+1) * (step+1);
            if (p > 0) r.entropy -= p * std::log(p) / calls;
        }
        require(std::abs(sum-1) < 1e-5, "probabilities do not sum to one");
    }
    return r;
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") { usage(); return 0; }
        const Config c = parse(argc, argv);
        const auto model = make_model(c);
        std::mt19937 rng(c.seed ^ 0x9e3779b9u);
        std::vector<int> prompt(c.prompt), tokens;
        for (int& t : prompt) t = static_cast<int>(rng() % c.vocab);
        tokens.reserve(c.prompt + c.generate);
        Scratch<float> scratch(c, c.prompt + c.generate);
        Array<float> outputs(size_t(std::max(1, c.generate))*c.vocab);
        std::cout << std::scientific << std::setprecision(6);
        if (c.check) checks(model, c, prompt);
        const auto warm = request(model, c, prompt, tokens, scratch, outputs);
        double prefill = 0, decode = 0, best = 1e300;
        volatile double observable = warm.checksum;
        Result last;
        for (int rep = 0; rep < c.repeats; ++rep) {
            last = request(model, c, prompt, tokens, scratch, outputs);
            require(std::abs(last.checksum-warm.checksum) < 1e-9, "repeat checksum changed");
            observable = last.checksum;
            prefill += last.prefill; decode += last.decode;
            best = std::min(best, last.prefill + last.decode);
        }
        (void)observable;
        const size_t parameters = size_t(2)*c.vocab*c.dim + size_t(c.layers)*
            (size_t(4)*c.dim*c.dim + size_t(2)*c.dim*c.ff);
        std::cout << "weights=fixed_random_untrained execution=serial_full_prefix\n"
                  << "dim=" << c.dim << " heads=" << c.heads << " ff=" << c.ff
                  << " layers=" << c.layers << " vocab=" << c.vocab << " prompt=" << c.prompt
                  << " generate=" << c.generate << " repeats=" << c.repeats << " seed=" << c.seed << '\n'
                  << "parameters=" << parameters << '\n'
                  << "inference_total_seconds=" << prefill+decode << '\n'
                  << "mean_request_seconds=" << (prefill+decode)/c.repeats << '\n'
                  << "best_request_seconds=" << best << '\n'
                  << "mean_prefill_seconds=" << prefill/c.repeats << '\n'
                  << "prefill_tokens_per_second=" << double(c.prompt)*c.repeats/prefill << '\n'
                  << "mean_decode_seconds=" << decode/c.repeats << '\n';
        if (c.generate > 1)
            std::cout << "decode_tokens_per_second=" << double(c.generate-1)*c.repeats/decode << '\n';
        if (c.generate > 0)
            std::cout << "generated_tokens_per_second=" << double(c.generate)*c.repeats/(prefill+decode) << '\n';
        std::cout << "mean_output_entropy_nats=" << last.entropy << " checksum=" << last.checksum << '\n'
                  << "generated_token_ids=";
        for (size_t i = prompt.size(); i < tokens.size(); ++i) std::cout << ' ' << tokens[i];
        std::cout << '\n';
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
