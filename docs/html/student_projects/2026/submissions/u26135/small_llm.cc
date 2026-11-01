#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using Vec = std::vector<double>;
using Tokens = std::vector<int>;

struct Corpus {
    Tokens tokens;
    std::vector<unsigned char> alphabet;
    std::array<int, 256> char_to_id{};
};

Corpus read_corpus(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open " + filename + " (run from code/)");
    std::string text(std::istreambuf_iterator<char>{file}, {});
    if (text.size() < 100) throw std::runtime_error("corpus needs at least 100 bytes");
    Corpus corpus;
    corpus.char_to_id.fill(-1);
    std::array<bool,256> present{};
    for (unsigned char c : text) present[c] = true;
    for (int c = 0; c < 256; ++c) {
        if (!present[c]) continue;
        corpus.char_to_id[c] = static_cast<int>(corpus.alphabet.size());
        corpus.alphabet.push_back(static_cast<unsigned char>(c));
    }
    corpus.tokens.reserve(text.size());
    for (unsigned char c : text) corpus.tokens.push_back(corpus.char_to_id[c]);
    return corpus;
}

struct Param {
    int rows, cols;
    Vec w, g, m, v;
    Param(int r, int c) : rows(r), cols(c), w(r*c), g(r*c), m(r*c), v(r*c) {}
    void init(std::mt19937& rng, double deviation) {
        std::normal_distribution<double> normal(0.0, deviation);
        for (double& x : w) x = normal(rng);
    }
    void step(int iteration, double rate) {
        const double c1 = 1.0 - std::pow(0.9, iteration);
        const double c2 = 1.0 - std::pow(0.999, iteration);
        for (size_t i = 0; i < w.size(); ++i) {
            m[i] = 0.9*m[i] + 0.1*g[i];
            v[i] = 0.999*v[i] + 0.001*g[i]*g[i];
            w[i] -= rate*(m[i]/c1)/(std::sqrt(v[i]/c2) + 1e-8);
            g[i] = 0.0;
        }
    }
};

// Each row of W maps one input vector to one output coordinate
void linear(const Vec& input, Vec& output, const Param& W, const Param& bias,
            int timesteps) {
    for (int t = 0; t < timesteps; ++t)
        for (int o = 0; o < W.rows; ++o) {
            double s = bias.w[o];
            for (int i = 0; i < W.cols; ++i)
                s += W.w[o*W.cols+i] * input[t*W.cols+i];
            output[t*W.rows+o] = s;
        }
}

// Accumulate gradients for a linear map and for its input
void linear_backward(const Vec& input, Vec& dinput, const Vec& doutput,
                     Param& W, Param& bias, int timesteps) {
    for (int t = 0; t < timesteps; ++t)
        for (int o = 0; o < W.rows; ++o) {
            const double d = doutput[t*W.rows+o];
            bias.g[o] += d;
            for (int i = 0; i < W.cols; ++i) {
                W.g[o*W.cols+i] += d * input[t*W.cols+i];
                dinput[t*W.cols+i] += W.w[o*W.cols+i] * d;
            }
        }
}

struct Cache {
    int T;
    Vec x, q, k, v, attention, context, projected, h, pre, activated,
        ff, y, probabilities;
    Cache(int t, int d, int vocab)
        : T(t), x(t*d), q(t*d), k(t*d), v(t*d), attention(t*t),
          context(t*d), projected(t*d), h(t*d), pre(t*2*d),
          activated(t*2*d), ff(t*d), y(t*d), probabilities(t*vocab) {}
};

struct Model {
    int d, vocab;
    Param token, position, Wq, bq, Wk, bk, Wv, bv, Wo, bo,
          W1, b1, W2, b2, Wout, bout;
    std::vector<Param*> params;

    Model(int width, int vocabulary, int context, std::mt19937& rng)
        : d(width), vocab(vocabulary), token(vocabulary, width),
          position(context, width), Wq(width,width), bq(1,width),
          Wk(width,width), bk(1,width), Wv(width,width), bv(1,width),
          Wo(width,width), bo(1,width), W1(2*width,width), b1(1,2*width),
          W2(width,2*width), b2(1,width), Wout(vocabulary,width),
          bout(1,vocabulary) {
        params = {&token,&position,&Wq,&bq,&Wk,&bk,&Wv,&bv,&Wo,&bo,
                  &W1,&b1,&W2,&b2,&Wout,&bout};
        token.init(rng, 0.12);
        position.init(rng, 0.12);
        for (Param* p : {&Wq,&Wk,&Wv,&Wo,&W1,&W2,&Wout})
            p->init(rng, 1.0/std::sqrt(p->cols));
    }

    Cache forward(const Tokens& sentence, int T) const {
        Cache c(T, d, vocab);
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < d; ++i)
                c.x[t*d+i] = token.w[sentence[t]*d+i] + position.w[t*d+i];
        linear(c.x,c.q,Wq,bq,T);
        linear(c.x,c.k,Wk,bk,T);
        linear(c.x,c.v,Wv,bv,T);

        // Causal softmax: position t sees only positions 0,...,t
        for (int t = 0; t < T; ++t) {
            double maximum = -1e300;
            for (int j = 0; j <= t; ++j) {
                double score = 0.0;
                for (int i = 0; i < d; ++i)
                    score += c.q[t*d+i]*c.k[j*d+i];
                c.attention[t*T+j] = score/std::sqrt(double(d));
                maximum = std::max(maximum, c.attention[t*T+j]);
            }
            double sum = 0.0;
            for (int j = 0; j <= t; ++j) {
                c.attention[t*T+j] = std::exp(c.attention[t*T+j]-maximum);
                sum += c.attention[t*T+j];
            }
            for (int j = 0; j <= t; ++j) {
                c.attention[t*T+j] /= sum;
                for (int i = 0; i < d; ++i)
                    c.context[t*d+i] += c.attention[t*T+j]*c.v[j*d+i];
            }
        }
        linear(c.context,c.projected,Wo,bo,T);
        for (int i = 0; i < T*d; ++i) c.h[i] = c.x[i] + c.projected[i];
        linear(c.h,c.pre,W1,b1,T);
        for (int i = 0; i < T*2*d; ++i) c.activated[i] = std::tanh(c.pre[i]);
        linear(c.activated,c.ff,W2,b2,T);
        for (int i = 0; i < T*d; ++i) c.y[i] = c.h[i] + c.ff[i];
        linear(c.y,c.probabilities,Wout,bout,T);
        for (int t = 0; t < T; ++t) {
            double maximum = *std::max_element(c.probabilities.begin()+t*vocab,
                                                c.probabilities.begin()+(t+1)*vocab);
            double sum = 0.0;
            for (int i = 0; i < vocab; ++i) {
                double& p = c.probabilities[t*vocab+i];
                p = std::exp(p-maximum);
                sum += p;
            }
            for (int i = 0; i < vocab; ++i) c.probabilities[t*vocab+i] /= sum;
        }
        return c;
    }

    double loss(const Tokens& sentence, const Cache& c) const {
        double total = 0.0;
        for (int t = 0; t < c.T; ++t)
            total -= std::log(std::max(1e-300, c.probabilities[t*vocab+sentence[t+1]]));
        return total/c.T;
    }

    // Reverse-mode derivatives through every layer, including causal attention
    void backward(const Tokens& sentence, const Cache& c, double scale) {
        const int T = c.T;
        Vec dout = c.probabilities, dy(T*d), dh(T*d), dff(T*d);
        for (int t = 0; t < T; ++t) {
            dout[t*vocab+sentence[t+1]] -= 1.0;
            for (int i = 0; i < vocab; ++i) dout[t*vocab+i] *= scale/T;
        }
        linear_backward(c.y,dy,dout,Wout,bout,T);
        dh = dy;
        dff = dy;
        Vec dactivated(T*2*d), dpre(T*2*d);
        linear_backward(c.activated,dactivated,dff,W2,b2,T);
        for (int i = 0; i < T*2*d; ++i)
            dpre[i] = dactivated[i]*(1.0-c.activated[i]*c.activated[i]);
        linear_backward(c.h,dh,dpre,W1,b1,T);
        Vec dx = dh, dprojected = dh, dcontext(T*d);
        linear_backward(c.context,dcontext,dprojected,Wo,bo,T);

        Vec dq(T*d), dk(T*d), dv(T*d);
        for (int t = 0; t < T; ++t) {
            Vec dprob(t+1);
            double weighted = 0.0;
            for (int j = 0; j <= t; ++j) {
                for (int i = 0; i < d; ++i) {
                    dprob[j] += dcontext[t*d+i]*c.v[j*d+i];
                    dv[j*d+i] += c.attention[t*T+j]*dcontext[t*d+i];
                }
                weighted += c.attention[t*T+j]*dprob[j];
            }
            for (int j = 0; j <= t; ++j) {
                const double ds = c.attention[t*T+j]*(dprob[j]-weighted)
                                  /std::sqrt(double(d));
                for (int i = 0; i < d; ++i) {
                    dq[t*d+i] += ds*c.k[j*d+i];
                    dk[j*d+i] += ds*c.q[t*d+i];
                }
            }
        }
        linear_backward(c.x,dx,dq,Wq,bq,T);
        linear_backward(c.x,dx,dk,Wk,bk,T);
        linear_backward(c.x,dx,dv,Wv,bv,T);
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < d; ++i) {
                token.g[sentence[t]*d+i] += dx[t*d+i];
                position.g[t*d+i] += dx[t*d+i];
            }
    }

    void update(int iteration, double rate) {
        for (Param* p : params) p->step(iteration, rate);
    }
};

double estimate_loss(const Model& model, const Tokens& text, size_t start,
                     size_t end, int context) {
    constexpr int windows = 32;
    double sum = 0.0;
    for (int i = 0; i < windows; ++i) {
        const size_t offset = start + (end-start-context-1)*i/(windows-1);
        Tokens window(text.begin()+offset,text.begin()+offset+context+1);
        sum += model.loss(window,model.forward(window,context));
    }
    return sum/windows;
}

void generate(const Model& model, const Corpus& corpus, int context,
              std::mt19937& rng) {
    const std::string prompt = "ROMEO:\n";
    Tokens text;
    for (unsigned char c : prompt) {
        if (corpus.char_to_id[c] < 0) throw std::runtime_error("prompt character absent from corpus");
        text.push_back(corpus.char_to_id[c]);
    }
    std::cout << prompt;
    for (int n = 0; n < 240; ++n) {
        Tokens window(text.end()-std::min<size_t>(context,text.size()),text.end());
        Cache c = model.forward(window,static_cast<int>(window.size()));
        const int last = c.T-1;
        std::discrete_distribution<int> choice(c.probabilities.begin()+last*model.vocab,
                                               c.probabilities.begin()+(last+1)*model.vocab);
        const int next = choice(rng);
        text.push_back(next);
        std::cout << static_cast<char>(corpus.alphabet[next]);
    }
    std::cout << '\n';
}

int main(int argc, char** argv) {
    try {
        if (argc > 7) throw std::invalid_argument("too many arguments");
        const int steps = argc > 1 ? std::stoi(argv[1]) : 1200;
        const int batch = argc > 2 ? std::stoi(argv[2]) : 8;
        const int width = argc > 3 ? std::stoi(argv[3]) : 32;
        const int context = argc > 4 ? std::stoi(argv[4]) : 32;
        const unsigned seed = argc > 5 ? static_cast<unsigned>(std::stoul(argv[5])) : 42;
        const std::string filename = argc > 6 ? argv[6] : "tinyshakespeare.txt";
        if (steps < 1 || batch < 1 || batch > 1024 || width < 4 || width > 256 ||
            context < 4 || context > 128)
            throw std::invalid_argument("steps/batch positive; batch <= 1024, width 4..256, context 4..128");
        std::mt19937 rng(seed);
        Corpus corpus = read_corpus(filename);
        const size_t train_end = corpus.tokens.size()*9/10;
        if (train_end <= static_cast<size_t>(context) ||
            corpus.tokens.size()-train_end <= static_cast<size_t>(context))
            throw std::invalid_argument("corpus too short for this context length");
        Model model(width,static_cast<int>(corpus.alphabet.size()),context,rng);
        std::cout << "Characters: " << corpus.tokens.size()
                  << ", training: " << train_end
                  << ", held out: " << corpus.tokens.size()-train_end
                  << ", alphabet: " << corpus.alphabet.size() << '\n';
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Training loss before: "
                  << estimate_loss(model,corpus.tokens,0,train_end,context) << '\n';
        std::cout << "Held-out loss before: "
                  << estimate_loss(model,corpus.tokens,train_end,corpus.tokens.size(),context) << '\n';
        std::cout << "Sample before:\n";
        std::mt19937 sampler(seed+1);
        generate(model,corpus,context,sampler);

        std::uniform_int_distribution<size_t> choose_start(0,train_end-context-1);
        auto start = std::chrono::steady_clock::now();
        for (int step = 1; step <= steps; ++step) {
            for (int b = 0; b < batch; ++b) {
                const size_t offset = choose_start(rng);
                Tokens window(corpus.tokens.begin()+offset,
                              corpus.tokens.begin()+offset+context+1);
                Cache c = model.forward(window,context);
                model.backward(window,c,1.0/batch);
            }
            model.update(step,0.003);
        }
        auto end = std::chrono::steady_clock::now();
        std::cout << "Training time (s): "
                  << std::chrono::duration<double>(end-start).count() << '\n';
        std::cout << "Training loss after: "
                  << estimate_loss(model,corpus.tokens,0,train_end,context) << '\n';
        std::cout << "Held-out loss after: "
                  << estimate_loss(model,corpus.tokens,train_end,corpus.tokens.size(),context) << '\n';
        std::cout << "Sample after:\n";
        generate(model,corpus,context,sampler);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what()
                  << "\nUsage: ./small_llm [steps=1200] [batch=8] [width=32]"
                     " [context=32] [seed=42] [corpus=tinyshakespeare.txt]\n";
        return 1;
    }
}
