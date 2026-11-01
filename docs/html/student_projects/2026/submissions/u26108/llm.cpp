// Serial decoder-only Transformer inference; standard C++17 only.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using Vec = std::vector<float>;
using Clock = std::chrono::steady_clock;

struct Config {
    int dim = 256, heads = 4, layers = 4, vocab = 256;
    int prompt = 128, generate = 64, seed = 1;
    std::string mode = "cached";
    bool check = false;
};

int positive(const std::string& s) {
    std::size_t used = 0;
    long long n = std::stoll(s, &used);
    if (used != s.size() || n < 1 || n > 1000000)
        throw std::runtime_error("integer arguments must be in [1, 1000000]");
    return static_cast<int>(n);
}

Config parse(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help") {
            std::cout << "./llm [--dim 256] [--heads 4] [--layers 4] [--vocab 256]\n"
                         "      [--prompt 128] [--generate 64] [--seed 1]\n"
                         "      [--mode cached|recompute|both] [--check]\n"
                         "--check compares all logits on fixed prefixes of length 1..8.\n";
            std::exit(0);
        }
        if (key == "--check") { c.check = true; continue; }
        if (i + 1 == argc) throw std::runtime_error("missing value for " + key);
        std::string value = argv[++i];
        if (key == "--mode") c.mode = value;
        else if (key == "--dim") c.dim = positive(value);
        else if (key == "--heads") c.heads = positive(value);
        else if (key == "--layers") c.layers = positive(value);
        else if (key == "--vocab") c.vocab = positive(value);
        else if (key == "--prompt") c.prompt = positive(value);
        else if (key == "--generate") c.generate = positive(value);
        else if (key == "--seed") c.seed = positive(value);
        else throw std::runtime_error("unknown option: " + key);
    }
    if (c.dim % c.heads != 0 || c.dim % 2 != 0)
        throw std::runtime_error("dim must be even and divisible by heads");
    if (c.mode != "cached" && c.mode != "recompute" && c.mode != "both")
        throw std::runtime_error("mode must be cached, recompute, or both");
    return c;
}

// Matrices are row-major [output dimension][input dimension].
struct Matrix {
    int rows, cols;
    Vec w;
    Matrix(int r, int c, std::mt19937& rng) : rows(r), cols(c), w(std::size_t(r)*c) {
        float scale = std::sqrt(3.0f / c);
        for (float& v : w)
            v = (2.0f * (double(rng()) / 4294967296.0) - 1.0f) * scale;
    }
    Vec operator()(const Vec& x) const {
        Vec y(rows, 0.0f);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                y[i] += w[std::size_t(i)*cols+j] * x[j];
        return y;
    }
};

Vec norm(const Vec& x) {
    double mean = 0, variance = 0;
    for (float v : x) mean += v;
    mean /= x.size();
    for (float v : x) variance += (v-mean)*(v-mean);
    float inv = 1.0 / std::sqrt(variance/x.size() + 1e-5);
    Vec y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = (x[i]-mean)*inv;
    return y;  // LayerNorm with gamma=1, beta=0.
}

void add(Vec& x, const Vec& y) {
    for (std::size_t i = 0; i < x.size(); ++i) x[i] += y[i];
}

struct Layer {
    Matrix q, k, v, out, up, down;
    Layer(int d, std::mt19937& rng)
        : q(d,d,rng), k(d,d,rng), v(d,d,rng), out(d,d,rng),
          up(4*d,d,rng), down(d,4*d,rng) {}
    void finish(Vec& x, const Vec& attention) const {
        add(x, out(attention));
        Vec h = up(norm(x));
        for (float& z : h)
            z = 0.5f*z*(1.0f + std::tanh(0.7978845608f*(z+0.044715f*z*z*z)));
        add(x, down(h));
    }
};

// K and V are flat arrays [position][head][head dimension].
// Only positions 0..position are visible: causal attention, stable softmax.
Vec attend(const Vec& q, const Vec& k, const Vec& v, int position, int heads) {
    int d = static_cast<int>(q.size()), hd = d/heads;
    Vec result(d, 0.0f), score(position+1);
    float scale = 1.0f/std::sqrt(float(hd));
    for (int h = 0; h < heads; ++h) {
        for (int t = 0; t <= position; ++t) {
            float dot = 0;
            for (int j = 0; j < hd; ++j)
                dot += q[h*hd+j] * k[std::size_t(t)*d+h*hd+j];
            score[t] = dot*scale;
        }
        float max_score = *std::max_element(score.begin(), score.end());
        float sum = 0;
        for (float& s : score) { s = std::exp(s-max_score); sum += s; }
        for (int t = 0; t <= position; ++t)
            for (int j = 0; j < hd; ++j)
                result[h*hd+j] += (score[t]/sum)*v[std::size_t(t)*d+h*hd+j];
    }
    return result;
}

struct Model {
    Config c;
    std::mt19937 rng;
    Matrix embedding, output;
    std::vector<Layer> layers;
    explicit Model(Config config)
        : c(config), rng(c.seed), embedding(c.vocab,c.dim,rng),
          output(c.vocab,c.dim,rng) {
        for (int i = 0; i < c.layers; ++i) layers.emplace_back(c.dim,rng);
    }
    Vec embed(int token, int position) const {
        Vec x(c.dim);
        for (int j = 0; j < c.dim; j += 2) {
            double angle = position/std::pow(10000.0, double(j)/c.dim);
            x[j] = embedding.w[std::size_t(token)*c.dim+j] + std::sin(angle);
            x[j+1] = embedding.w[std::size_t(token)*c.dim+j+1] + std::cos(angle);
        }
        return x;
    }
    // Independent full-sequence schedule: recompute every token at every layer.
    // No persistent KV cache; output logits for the last position only.
    Vec full(const std::vector<int>& tokens) const {
        int n = static_cast<int>(tokens.size());
        std::vector<Vec> x;
        for (int t = 0; t < n; ++t) x.push_back(embed(tokens[t], t));
        for (const Layer& layer : layers) {
            std::vector<Vec> queries;
            Vec keys(std::size_t(n)*c.dim), values(keys.size());
            for (int t = 0; t < n; ++t) {
                Vec z = norm(x[t]), k = layer.k(z), v = layer.v(z);
                queries.push_back(layer.q(z));
                std::copy(k.begin(), k.end(), keys.begin()+std::size_t(t)*c.dim);
                std::copy(v.begin(), v.end(), values.begin()+std::size_t(t)*c.dim);
            }
            for (int t = 0; t < n; ++t)
                layer.finish(x[t], attend(queries[t], keys, values, t, c.heads));
        }
        return output(norm(x.back()));
    }
    std::size_t parameters() const {
        std::size_t n = embedding.w.size()+output.w.size();
        for (const Layer& l : layers)
            n += l.q.w.size()+l.k.w.size()+l.v.w.size()+l.out.w.size()
                 +l.up.w.size()+l.down.w.size();
        return n;
    }
};

struct Cached {
    const Model& model;
    int position = 0, capacity;
    std::vector<Vec> keys, values;
    Cached(const Model& m, int cap) : model(m), capacity(cap),
        keys(m.c.layers, Vec(std::size_t(cap)*m.c.dim)), values(keys) {}
    Vec step(int token, bool logits = true) {
        if (position >= capacity) throw std::runtime_error("KV cache capacity exceeded");
        Vec x = model.embed(token, position);
        for (int l = 0; l < model.c.layers; ++l) {
            const Layer& layer = model.layers[l];
            Vec z = norm(x), q = layer.q(z), k = layer.k(z), v = layer.v(z);
            std::size_t offset = std::size_t(position)*model.c.dim;
            std::copy(k.begin(), k.end(), keys[l].begin()+offset);
            std::copy(v.begin(), v.end(), values[l].begin()+offset);
            layer.finish(x, attend(q, keys[l], values[l], position, model.c.heads));
        }
        ++position;
        return logits ? model.output(norm(x)) : Vec{};
    }
};

int argmax(const Vec& x) {
    for (float v : x)
        if (!std::isfinite(v)) throw std::runtime_error("non-finite logits");
    return static_cast<int>(std::max_element(x.begin(), x.end())-x.begin());
}

void check(const Model& m) {
    Cached cache(m, 8);
    std::vector<int> tokens;
    double max_error = 0, max_scaled = 0;
    bool same_argmax = true;
    for (int t = 0; t < 8; ++t) {
        tokens.push_back((t*7+3)%m.c.vocab);
        Vec a = cache.step(tokens.back()), b = m.full(tokens);
        same_argmax = same_argmax && (argmax(a) == argmax(b));
        for (int i = 0; i < m.c.vocab; ++i) {
            double error = std::abs(double(a[i])-b[i]);
            max_error = std::max(max_error,error);
            max_scaled = std::max(max_scaled,error/(1.0+std::abs(b[i])));
        }
    }
    bool ok = max_scaled <= 1e-4 && same_argmax;
    std::cout << "check_prefixes=8 max_abs_logit_error=" << max_error
              << " max_scaled_error=" << max_scaled
              << " argmax_match=" << same_argmax << " check=" << (ok ? "PASS" : "FAIL") << '\n';
    if (!ok) throw std::runtime_error("cache/reference comparison failed");
}

double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b-a).count();
}

std::vector<int> run(const Model& m, bool use_cache) {
    const Config& c = m.c;
    std::vector<int> tokens;
    std::mt19937 input_rng(static_cast<unsigned>(c.seed)^0x9e3779b9u);
    for (int t = 0; t < c.prompt; ++t) tokens.push_back(input_rng()%c.vocab);
    // Cache allocation is outside the computation timer. No cache in reference mode.
    std::vector<Cached> caches;
    if (use_cache) caches.emplace_back(m,c.prompt+c.generate-1);
    auto begin = Clock::now();
    Vec logits;
    if (use_cache) {
        // Plain token-at-a-time prefill; later work can use matrix-matrix products.
        for (int t = 0; t < c.prompt; ++t)
            logits = caches[0].step(tokens[t],t == c.prompt-1);
    } else logits = m.full(tokens);
    auto prefilled = Clock::now();
    std::vector<int> generated;
    generated.push_back(argmax(logits));
    tokens.push_back(generated.back());
    auto decode_begin = Clock::now();
    for (int t = 1; t < c.generate; ++t) {
        logits = use_cache ? caches[0].step(tokens.back()) : m.full(tokens);
        generated.push_back(argmax(logits));
        tokens.push_back(generated.back());
    }
    auto end = Clock::now();
    double prefill = seconds(begin,prefilled), decode = seconds(decode_begin,end);
    double checksum = 0;
    for (int i = 0; i < c.vocab; ++i) checksum += (i+1)*double(logits[i]);
    double kv_bytes = use_cache ? 2.0*c.layers*(c.prompt+c.generate-1)*c.dim*sizeof(float) : 0;
    std::cout << "mode=" << (use_cache ? "cached" : "recompute")
              << " prefill_s=" << prefill << " decode_s=" << decode
              << " total_s=" << seconds(begin,end)
              << " decode_steps=" << c.generate-1 << '\n'
              << "prefill_tokens_per_s=" << c.prompt/prefill;
    if (c.generate > 1) std::cout << " decode_tokens_per_s=" << (c.generate-1)/decode;
    std::cout << " kv_cache_MiB=" << kv_bytes/(1024*1024)
              << " last_logits_checksum=" << checksum << '\n'
              << "generated_token_ids (first 32):";
    for (int i = 0; i < std::min(c.generate,32); ++i) std::cout << ' ' << generated[i];
    std::cout << '\n';
    return generated;
}

int main(int argc, char** argv) {
    try {
        Config c = parse(argc,argv);
        Model model(c);
        std::cout << std::setprecision(9)
                  << "Untrained synthetic Transformer; generated IDs are not natural language.\n"
                  << "dim=" << c.dim << " heads=" << c.heads << " layers=" << c.layers
                  << " vocab=" << c.vocab << " prompt=" << c.prompt
                  << " generate=" << c.generate << " seed=" << c.seed
                  << " parameters=" << model.parameters() << '\n';
        if (c.check) check(model);
        if (c.mode == "both") {
            auto a = run(model,true), b = run(model,false);
            bool equal = a == b;
            std::cout << "generated_sequence_match=" << equal << '\n';
            if (!equal) throw std::runtime_error("generated sequences differ");
        } else run(model,c.mode == "cached");
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
