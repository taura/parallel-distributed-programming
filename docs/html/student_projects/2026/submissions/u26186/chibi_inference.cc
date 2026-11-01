#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

void read_exact(std::istream& in, void* data, std::size_t bytes) {
  if (!in.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes))) throw std::runtime_error("unexpected end of input file");
}

struct Config {
    int dim, hidden, layers, heads, kv_heads, vocab, context;
    int head_size() const { return dim / heads; }
    int kv_dim() const { return head_size() * kv_heads; }
};

class Model {
  public:
    Config c{};
    std::vector<float> storage;
    const float *embedding{}, *attention_norm{}, *query{}, *key{}, *value{};
    const float *attention_out{}, *feed_norm{}, *gate{}, *down{}, *up{}, *final_norm{};

    explicit Model(const std::string& path) {
      const std::uint32_t endian = 1;
      if (*reinterpret_cast<const unsigned char*>(&endian) != 1 || sizeof(float) != 4 || !std::numeric_limits<float>::is_iec559)
        throw std::runtime_error("model requires little-endian IEEE 754 float32");
      std::ifstream f(path, std::ios::binary | std::ios::ate);
      if (!f) throw std::runtime_error("cannot open model: " + path);
      const std::streamoff bytes = f.tellg();
      if (bytes < 28 || bytes > 256 * 1024 * 1024 || (bytes - 28) % 4 != 0) throw std::runtime_error("unexpected model size");
      f.seekg(0);
      std::array<std::int32_t, 7> header{};
      read_exact(f, header.data(), sizeof(header));
      c = {header[0], header[1], header[2], header[3], header[4], header[5], header[6]};
      if (c.dim <= 0 || c.dim > 8192 || c.hidden <= 0 || c.hidden > 32768 || c.layers <= 0 || c.layers > 128 || c.heads <= 0 || c.heads > 128 ||
          c.kv_heads <= 0 || c.kv_heads > c.heads || c.vocab <= 0 || c.vocab > 262144 || c.context <= 0 || c.context > 32768 || c.dim % c.heads ||
          c.heads % c.kv_heads || c.head_size() % 2)
        throw std::runtime_error("unsupported model dimensions or untied classifier");
      storage.resize(static_cast<std::size_t>(bytes - 28) / sizeof(float));
      read_exact(f, storage.data(), storage.size() * sizeof(float));
      std::size_t at = 0;
      auto take = [&](std::size_t count) {
        if (count > storage.size() - at) throw std::runtime_error("truncated model tensors");
        const float* result = storage.data() + at;
        at += count;
        return result;
      };
      const std::size_t d = c.dim, h = c.hidden, l = c.layers, k = c.kv_dim();
      embedding = take(static_cast<std::size_t>(c.vocab) * d);
      attention_norm = take(l * d);
      query = take(l * d * d);
      key = take(l * k * d);
      value = take(l * k * d);
      attention_out = take(l * d * d);
      feed_norm = take(l * d);
      gate = take(l * h * d);
      down = take(l * d * h);
      up = take(l * h * d);
      final_norm = take(d);
      take(static_cast<std::size_t>(c.context) * c.head_size());
      if (at != storage.size()) throw std::runtime_error("unexpected trailing model tensors");
    }
};

class Tokenizer {
    std::vector<std::string> pieces;
    std::vector<float> scores;
    std::unordered_map<std::string, int> ids;

    int find(const std::string& piece) const {
      auto it = ids.find(piece);
      return it == ids.end() ? -1 : it->second;
    }

  public:
    Tokenizer(const std::string& path, int count) : pieces(count), scores(count) {
      std::ifstream f(path, std::ios::binary);
      if (!f) throw std::runtime_error("cannot open tokenizer: " + path);
      std::uint32_t max_length;
      read_exact(f, &max_length, sizeof(max_length));
      if (max_length == 0 || max_length > 4096) throw std::runtime_error("invalid tokenizer header");
      for (int i = 0; i < count; ++i) {
        std::int32_t length;
        read_exact(f, &scores[i], sizeof(float));
        read_exact(f, &length, sizeof(length));
        if (length <= 0 || static_cast<std::uint32_t>(length) > max_length) throw std::runtime_error("invalid tokenizer entry");
        pieces[i].resize(static_cast<std::size_t>(length));
        read_exact(f, &pieces[i][0], pieces[i].size());
        if (!ids.emplace(pieces[i], i).second) throw std::runtime_error("duplicate tokenizer entry");
      }
      if (f.peek() != std::char_traits<char>::eof()) throw std::runtime_error("extra tokenizer data");
      if (count < 259 || find(" ") < 0) throw std::runtime_error("tokenizer lacks byte or space tokens");
      for (int byte = 0; byte < 256; ++byte) {
        static const char hex[] = "0123456789ABCDEF";
        std::string label = "<0x00>";
        label[3] = hex[byte / 16];
        label[4] = hex[byte % 16];
        if (pieces[byte + 3] != label) throw std::runtime_error("unexpected byte-token layout");
      }
    }

    std::vector<int> encode(const std::string& text) const {
      std::vector<int> result{1};
      if (!text.empty()) result.push_back(find(" "));
      for (std::size_t i = 0; i < text.size();) {
        const unsigned char first = static_cast<unsigned char>(text[i]);
        std::size_t width =
            first < 0x80 ? 1 : (first >= 0xC2 && first <= 0xDF ? 2 : (first >= 0xE0 && first <= 0xEF ? 3 : (first >= 0xF0 && first <= 0xF4 ? 4 : 0)));
        if (!width || i + width > text.size()) throw std::runtime_error("prompt is not valid UTF-8");
        for (std::size_t j = 1; j < width; ++j)
          if ((static_cast<unsigned char>(text[i + j]) & 0xC0) != 0x80) throw std::runtime_error("prompt is not valid UTF-8");
        if (width >= 3) {
          const unsigned char second = static_cast<unsigned char>(text[i + 1]);
          if ((first == 0xE0 && second < 0xA0) || (first == 0xED && second > 0x9F) || (first == 0xF0 && second < 0x90) || (first == 0xF4 && second > 0x8F))
            throw std::runtime_error("prompt is not valid UTF-8");
        }
        const std::string symbol = text.substr(i, width);
        int id = find(symbol);
        if (id >= 0)
          result.push_back(id);
        else
          for (unsigned char b : symbol) result.push_back(static_cast<int>(b) + 3);
        i += width;
      }
      while (result.size() > 1) {
        float best = -std::numeric_limits<float>::infinity();
        std::size_t place = result.size();
        int merged = -1;
        for (std::size_t i = 0; i + 1 < result.size(); ++i) {
          int id = find(pieces[result[i]] + pieces[result[i + 1]]);
          if (id >= 0 && scores[id] > best) {
            best = scores[id];
            place = i;
            merged = id;
          }
        }
        if (merged < 0) break;
        result[place] = merged;
        result.erase(result.begin() + static_cast<std::ptrdiff_t>(place + 1));
      }
      return result;
    }

    std::string decode(int previous, int token) const {
      if (token < 0 || token >= static_cast<int>(pieces.size())) throw std::runtime_error("token outside vocabulary");
      std::string s = pieces[token];
      if (previous == 1 && !s.empty() && s[0] == ' ') s.erase(0, 1);
      if (token >= 3 && token < 259) {
        const unsigned char b = static_cast<unsigned char>(token - 3);
        if (b < 32 && b != '\n' && b != '\r' && b != '\t') return {};
        return std::string(1, static_cast<char>(b));
      }
      return s;
    }
};

void multiply(const float* matrix, const float* input, float* output, int rows, int columns) {
  for (int row = 0; row < rows; ++row) {
    float sum = 0.0f;
    for (int col = 0; col < columns; ++col) sum += matrix[static_cast<std::size_t>(row) * columns + col] * input[col];
    output[row] = sum;
  }
}

void normalize(const float* input, const float* weight, float* output, int count) {
  float squares = 0.0f;
  for (int i = 0; i < count; ++i) squares += input[i] * input[i];
  const float factor = 1.0f / std::sqrt(squares / count + 1e-5f);
  for (int i = 0; i < count; ++i) output[i] = weight[i] * (input[i] * factor);
}

struct Intervention {
    int layer = -1, neuron = 0;
    float delta = 0.0f;
};

class Inference {
    const Model& m;
    std::vector<float> x, norm, scratch, q, attention, scores, f1, f2, keys, values, logits;

  public:
    explicit Inference(const Model& model)
        : m(model),
          x(m.c.dim),
          norm(m.c.dim),
          scratch(m.c.dim),
          q(m.c.dim),
          attention(m.c.dim),
          scores(static_cast<std::size_t>(m.c.heads) * m.c.context),
          f1(m.c.hidden),
          f2(m.c.hidden),
          keys(static_cast<std::size_t>(m.c.layers) * m.c.context * m.c.kv_dim()),
          values(keys.size()),
          logits(m.c.vocab) {}

    const std::vector<float>& step(int token, int position, Intervention intervention = {}) {
      const Config& c = m.c;
      if (token < 0 || token >= c.vocab || position < 0 || position >= c.context) throw std::runtime_error("token or position outside model range");
      const std::size_t d = c.dim, hidden = c.hidden, kv = c.kv_dim(), head = c.head_size();
      std::copy_n(m.embedding + static_cast<std::size_t>(token) * d, d, x.begin());
      for (int layer = 0; layer < c.layers; ++layer) {
        const std::size_t l = static_cast<std::size_t>(layer);
        normalize(x.data(), m.attention_norm + l * d, norm.data(), c.dim);
        const std::size_t offset = (l * c.context + position) * kv;
        float* k = keys.data() + offset;
        float* v = values.data() + offset;
        multiply(m.query + l * d * d, norm.data(), q.data(), c.dim, c.dim);
        multiply(m.key + l * kv * d, norm.data(), k, c.kv_dim(), c.dim);
        multiply(m.value + l * kv * d, norm.data(), v, c.kv_dim(), c.dim);
        for (std::size_t i = 0; i < d; i += 2) {
          const float theta = position / std::pow(10000.0f, (i % head) / static_cast<float>(head));
          const float cs = std::cos(theta), sn = std::sin(theta);
          const float q0 = q[i], q1 = q[i + 1];
          q[i] = q0 * cs - q1 * sn;
          q[i + 1] = q0 * sn + q1 * cs;
          if (i < kv) {
            const float k0 = k[i], k1 = k[i + 1];
            k[i] = k0 * cs - k1 * sn;
            k[i + 1] = k0 * sn + k1 * cs;
          }
        }
        for (int h = 0; h < c.heads; ++h) {
          float* s = scores.data() + static_cast<std::size_t>(h) * c.context;
          const float* query_head = q.data() + static_cast<std::size_t>(h) * head;
          const std::size_t kv_head = static_cast<std::size_t>(h / (c.heads / c.kv_heads)) * head;
          float max_score = -std::numeric_limits<float>::infinity();
          for (int t = 0; t <= position; ++t) {
            const float* kt = keys.data() + (l * c.context + t) * kv + kv_head;
            float dot = 0.0f;
            for (std::size_t i = 0; i < head; ++i) dot += query_head[i] * kt[i];
            s[t] = dot / std::sqrt(static_cast<float>(head));
            max_score = std::max(max_score, s[t]);
          }
          float sum = 0.0f;
          for (int t = 0; t <= position; ++t) {
            s[t] = std::exp(s[t] - max_score);
            sum += s[t];
          }
          float* out = attention.data() + static_cast<std::size_t>(h) * head;
          std::fill_n(out, head, 0.0f);
          for (int t = 0; t <= position; ++t) {
            const float* vt = values.data() + (l * c.context + t) * kv + kv_head;
            const float weight = s[t] / sum;
            for (std::size_t i = 0; i < head; ++i) out[i] += weight * vt[i];
          }
        }
        multiply(m.attention_out + l * d * d, attention.data(), scratch.data(), c.dim, c.dim);
        for (std::size_t i = 0; i < d; ++i) x[i] += scratch[i];
        normalize(x.data(), m.feed_norm + l * d, norm.data(), c.dim);
        multiply(m.gate + l * hidden * d, norm.data(), f1.data(), c.hidden, c.dim);
        multiply(m.up + l * hidden * d, norm.data(), f2.data(), c.hidden, c.dim);
        for (std::size_t i = 0; i < hidden; ++i) f1[i] = (f1[i] / (1.0f + std::exp(-f1[i]))) * f2[i];
        if (layer == intervention.layer && intervention.delta != 0.0f) f1[intervention.neuron] += intervention.delta;
        multiply(m.down + l * d * hidden, f1.data(), scratch.data(), c.dim, c.hidden);
        for (std::size_t i = 0; i < d; ++i) x[i] += scratch[i];
      }
      normalize(x.data(), m.final_norm, norm.data(), c.dim);
      multiply(m.embedding, norm.data(), logits.data(), c.vocab, c.dim);
      return logits;
    }
};

struct Options {
    std::string model = "stories260K.bin", tokenizer = "tok512.bin", prompt = "Once upon a time";
    int tokens = 128, repeat = 1, layer = 0, first_neuron = 0, neurons = 4;
    float dose = 1.0f;
    bool trace = false, sweep = false, sweep_options = false;
};

int number(const std::string& text, int maximum, int minimum = 1) {
  std::size_t used = 0;
  long value = std::stol(text, &used);
  if (used != text.size() || value < minimum || value > maximum) throw std::runtime_error("numeric argument outside allowed range");
  return static_cast<int>(value);
}

float dose_number(const std::string& text) {
  std::size_t used = 0;
  const float value = std::stof(text, &used);
  if (used != text.size() || !std::isfinite(value) || value < 0.0f || value > 10000.0f) throw std::runtime_error("dose must be finite and between 0 and 10000");
  return value;
}

Options arguments(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help") {
      std::cout << "Usage: ./tiny_inference [--model FILE] [--tokenizer FILE] "
                   "[--prompt TEXT] [--tokens N] [--repeat N] [--trace]\n"
                   "       [--sweep --layer L --start N --neurons N --dose D]\n";
      std::exit(0);
    }
    if (key == "--sweep") {
      o.sweep = true;
      continue;
    }
    if (key == "--trace") {
      o.trace = true;
      continue;
    }
    if (++i >= argc) throw std::runtime_error("missing option value: " + key);
    const std::string value = argv[i];
    if (key == "--model")
      o.model = value;
    else if (key == "--tokenizer")
      o.tokenizer = value;
    else if (key == "--prompt")
      o.prompt = value;
    else if (key == "--tokens")
      o.tokens = number(value, 32768);
    else if (key == "--repeat")
      o.repeat = number(value, 1000000);
    else if (key == "--layer") {
      o.layer = number(value, 127, 0);
      o.sweep_options = true;
    } else if (key == "--start") {
      o.first_neuron = number(value, 32767, 0);
      o.sweep_options = true;
    } else if (key == "--neurons") {
      o.neurons = number(value, 32768);
      o.sweep_options = true;
    } else if (key == "--dose") {
      o.dose = dose_number(value);
      o.sweep_options = true;
    } else
      throw std::runtime_error("unknown option: " + key);
  }
  if (!o.sweep && o.sweep_options) throw std::runtime_error("layer, start, neurons, and dose require --sweep");
  if (o.sweep && o.repeat != 1) throw std::runtime_error("--repeat cannot be combined with --sweep");
  return o;
}

std::vector<double> log_probabilities(const std::vector<float>& logits) {
  const double maximum = *std::max_element(logits.begin(), logits.end());
  double sum = 0.0;
  for (float value : logits) sum += std::exp(static_cast<double>(value) - maximum);
  const double log_sum = maximum + std::log(sum);
  std::vector<double> result;
  result.reserve(logits.size());
  for (float value : logits) result.push_back(value - log_sum);
  return result;
}

struct Generation {
    std::string text;
    std::vector<double> first_logp;
    int first_token = -1, generated = 0, prefill_steps = 0, decode_steps = 0;
    double prefill_ms = 0.0, decode_ms = 0.0;
};

void trace_input(const std::vector<int>& prompt, bool trace) {
  if (!trace) return;
  std::cerr << "INPUT";
  for (int id : prompt) std::cerr << ' ' << id;
  std::cerr << '\n' << std::setprecision(9);
}

Generation generate(Inference& inference, const Tokenizer& tokenizer, const std::vector<int>& prompt, const Options& o, int index,
                    Intervention intervention = {}) {
  Generation result;
  if (o.trace) std::cerr << "RUN " << index << '\n';
  const int prompt_size = static_cast<int>(prompt.size());
  const int total_steps = prompt_size + o.tokens - 1;
  int current = prompt[0];
  for (int pos = 0; pos < total_steps; ++pos) {
    const auto start = Clock::now();
    const std::vector<float>& logits = inference.step(current, pos, intervention);
    const double elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    if (pos < prompt_size) {
      result.prefill_ms += elapsed;
      ++result.prefill_steps;
    } else {
      result.decode_ms += elapsed;
      ++result.decode_steps;
    }
    const int next = pos < prompt_size - 1 ? prompt[pos + 1] : static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
    if (o.sweep && pos == prompt_size - 1) {
      result.first_token = next;
      result.first_logp = log_probabilities(logits);
    }
    if (o.trace) {
      std::cerr << "TRACE " << pos << ' ' << next;
      for (float value : logits) std::cerr << ' ' << value;
      std::cerr << '\n';
    }
    if (next == 1) break;
    if (pos >= prompt_size - 1) ++result.generated;
    if (o.sweep || index == 0) result.text += tokenizer.decode(current, next);
    current = next;
  }
  return result;
}

int sweep(const Options& o, const Model& model, const Tokenizer& tokenizer, const std::vector<int>& prompt, double load_ms) {
  if (o.layer >= model.c.layers || o.first_neuron >= model.c.hidden || o.neurons > model.c.hidden - o.first_neuron)
    throw std::runtime_error("sweep layer or neuron range is outside the model");
  trace_input(prompt, o.trace);
  std::cout << std::setprecision(9) << "SWEEP model=" << std::quoted(o.model) << " tokenizer=" << std::quoted(o.tokenizer)
            << " prompt=" << std::quoted(o.prompt) << " tokens=" << o.tokens << " layer=" << o.layer << " start=" << o.first_neuron << " neurons=" << o.neurons
            << " dose=" << o.dose << " prompt_tokens=" << prompt.size() << '\n';
  const auto sweep_start = Clock::now();
  std::vector<double> control_logp;
  int control_first_token = -1;
  for (int case_index = 0; case_index < 1 + 2 * o.neurons; ++case_index) {
    const int neuron = case_index ? o.first_neuron + (case_index - 1) / 2 : -1;
    const float delta = case_index ? ((case_index % 2) ? o.dose : -o.dose) : 0.0f;
    const Intervention intervention{case_index ? o.layer : -1, neuron, delta};
    Inference inference(model);
    const Generation result = generate(inference, tokenizer, prompt, o, case_index, intervention);
    double kl = 0.0, control_token_logp_delta = 0.0;
    if (case_index == 0) {
      control_logp = result.first_logp;
      control_first_token = result.first_token;
    } else {
      for (std::size_t i = 0; i < result.first_logp.size(); ++i) kl += std::exp(control_logp[i]) * (control_logp[i] - result.first_logp[i]);
      control_token_logp_delta = result.first_logp[control_first_token] - control_logp[control_first_token];
    }
    std::cout << "CASE index=" << case_index << " neuron=" << neuron << " delta=" << delta << " first_token=" << result.first_token << " kl_from_control=" << kl
              << " control_token_logp_delta=" << control_token_logp_delta << " generated_tokens=" << result.generated
              << " forward_steps=" << result.prefill_steps + result.decode_steps << " forward_ms=" << result.prefill_ms + result.decode_ms << '\n';
    std::cout << "TEXT " << std::quoted(result.text) << '\n';
  }
  const double total_ms = std::chrono::duration<double, std::milli>(Clock::now() - sweep_start).count();
  std::cerr << std::fixed << std::setprecision(3) << "model_load_and_setup_ms=" << load_ms << " sweep_wall_ms=" << total_ms << " cases=" << 1 + 2 * o.neurons
            << '\n';
  return 0;
}

int run(const Options& o) {
  const auto load_start = Clock::now();
  Model model(o.model);
  Tokenizer tokenizer(o.tokenizer, model.c.vocab);
  const std::vector<int> prompt = tokenizer.encode(o.prompt);
  if (prompt.size() + static_cast<std::size_t>(o.tokens) - 1 > static_cast<std::size_t>(model.c.context))
    throw std::runtime_error("prompt and requested generation exceed model context");
  if (o.sweep) {
    const double load_ms = std::chrono::duration<double, std::milli>(Clock::now() - load_start).count();
    return sweep(o, model, tokenizer, prompt, load_ms);
  }
  Inference inference(model);
  const double load_ms = std::chrono::duration<double, std::milli>(Clock::now() - load_start).count();
  trace_input(prompt, o.trace);
  double prefill_ms = 0.0, decode_ms = 0.0;
  std::uint64_t prefill_steps = 0, decode_steps = 0, produced = 0;
  std::string first_result;
  for (int run_index = 0; run_index < o.repeat; ++run_index) {
    const Generation result = generate(inference, tokenizer, prompt, o, run_index);
    prefill_ms += result.prefill_ms;
    decode_ms += result.decode_ms;
    prefill_steps += result.prefill_steps;
    decode_steps += result.decode_steps;
    produced += result.generated;
    if (run_index == 0) first_result = result.text;
  }
  std::cout << first_result << '\n';
  std::cerr << std::fixed << std::setprecision(3) << "model_load_and_setup_ms=" << load_ms << " repeats=" << o.repeat << " prompt_tokens=" << prompt.size()
            << " generated_tokens=" << produced << '\n'
            << "prefill_forward_ms=" << prefill_ms << " prefill_forward_steps=" << prefill_steps << " decode_forward_ms=" << decode_ms
            << " decode_forward_steps=" << decode_steps << '\n';
  if (decode_steps && decode_ms > 0.0) std::cerr << "decode_forward_tokens_per_second=" << 1000.0 * decode_steps / decode_ms << '\n';
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    return run(arguments(argc, argv));
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
