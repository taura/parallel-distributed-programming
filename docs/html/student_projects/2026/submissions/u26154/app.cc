#include <algorithm>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "inference/model_loader.h"
#include "inference/orchestrator.h"
#include "inference/ops.h"
#include "inference/tokenizer.h"
#include "inference/types.h"
#include "utils/file_utils.h"

namespace {

std::string TrimTrailingNewlines(std::string s) {
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
    s.pop_back();
  }
  return s;
}

std::vector<int> ParseCsvIds(const std::string& csv) {
  std::vector<int> ids;
  if (csv.empty()) {
    return ids;
  }
  std::stringstream ss(csv);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) {
      continue;
    }
    ids.push_back(std::stoi(item));
  }
  return ids;
}

Args ParseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s(argv[i]);
    auto need = [&](const std::string& flag) {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + flag);
      }
      return std::string(argv[++i]);
    };
    if (s == "--model-dir") {
      a.model_dir = need(s);
    } else if (s == "--prompt-file") {
      a.prompt_file = need(s);
    } else if (s == "--prompt-text") {
      a.prompt_text = need(s);
    } else if (s == "--tokenizer-json") {
      a.tokenizer_json = need(s);
    } else if (s == "--gen-len") {
      a.gen_len = std::stoi(need(s));
    } else if (s == "--layers") {
      a.layers = std::stoi(need(s));
    } else if (s == "--warmup") {
      a.warmup = std::stoi(need(s));
    } else if (s == "--repeat") {
      a.repeat = std::stoi(need(s));
    } else if (s == "--seed") {
      a.seed = std::stoi(need(s));
    } else if (s == "--temperature") {
      a.temperature = std::stof(need(s));
    } else if (s == "--top-k") {
      a.top_k = std::stoi(need(s));
    } else if (s == "--repetition-penalty") {
      a.repetition_penalty = std::stof(need(s));
    } else if (s == "--dump-last-logits") {
      a.dump_last_logits_file = need(s);
    } else if (s == "--inspect-logit-ids") {
      a.inspect_logit_ids_csv = need(s);
    } else if (s == "--help") {
      std::cout << "Usage: ./app [--model-dir DIR] [--prompt-file FILE | --prompt-text TEXT] "
                << "[--tokenizer-json FILE] [--gen-len N] [--layers N|-1] [--warmup N] "
                << "[--repeat N] [--seed N] [--temperature X] [--top-k N] "
                << "[--repetition-penalty X] [--dump-last-logits FILE] "
                << "[--inspect-logit-ids CSV]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + s);
    }
  }
  if (a.gen_len < 0 || a.warmup < 0 || a.repeat <= 0) {
    throw std::runtime_error("Invalid non-negative arguments");
  }
  if (a.temperature < 0.0f || a.top_k == 0 || a.repetition_penalty < 1.0f) {
    throw std::runtime_error("Invalid sampling args: temperature>=0, top-k!=0, repetition-penalty>=1");
  }
  return a;
}

int MainImpl(int argc, char** argv) {
  const Args args = ParseArgs(argc, argv);
  const ModelWeights model = LoadModel(args.model_dir);
  std::vector<int> prompt;
  ByteLevelBpeTokenizer tokenizer;
  bool has_tokenizer = false;
  std::string prompt_text_for_display;
  if (!args.prompt_text.empty()) {
    tokenizer.LoadFromTokenizerJson(args.tokenizer_json);
    has_tokenizer = true;
    prompt_text_for_display = args.prompt_text;
    prompt = tokenizer.Encode(args.prompt_text);
  } else {
    const std::string prompt_file_content = ReadFileToString(args.prompt_file);
    std::istringstream iss(prompt_file_content);
    std::vector<int> maybe_ids;
    bool parse_failed = false;
    while (true) {
      int x = 0;
      if (iss >> x) {
        if (x < 0 || x >= model.cfg.vocab_size) {
          throw std::runtime_error("Prompt token out of range in file: " + std::to_string(x));
        }
        maybe_ids.push_back(x);
      } else {
        if (iss.eof()) {
          break;
        }
        parse_failed = true;
        break;
      }
    }

    if (!parse_failed && !maybe_ids.empty()) {
      prompt = std::move(maybe_ids);
    } else {
      tokenizer.LoadFromTokenizerJson(args.tokenizer_json);
      has_tokenizer = true;
      prompt_text_for_display = TrimTrailingNewlines(prompt_file_content);
      prompt = tokenizer.Encode(prompt_text_for_display);
    }
  }

  if (prompt.empty()) {
    throw std::runtime_error("Prompt is empty");
  }

  const int layers = (args.layers < 0) ? model.cfg.num_hidden_layers : args.layers;
  if (layers <= 0 || layers > model.cfg.num_hidden_layers) {
    throw std::runtime_error("--layers must be in [1, num_hidden_layers] or -1");
  }

  std::cout << "model_dir: " << args.model_dir << "\n";
  std::cout << "hidden_size: " << model.cfg.hidden_size << ", layers: " << model.cfg.num_hidden_layers
            << ", heads: " << model.cfg.num_attention_heads << ", vocab: " << model.cfg.vocab_size
            << "\n";
  std::cout << "prompt_len: " << prompt.size() << ", gen_len: " << args.gen_len
            << ", active_layers: " << layers << "\n";
  std::cout << "sampling: temperature=" << args.temperature << ", top_k=" << args.top_k
            << ", repetition_penalty=" << args.repetition_penalty << "\n";
  if (has_tokenizer) {
    std::cout << "prompt_text: " << prompt_text_for_display << "\n";
  }

  for (int i = 0; i < args.warmup; ++i) {
    (void)RunOne(model, prompt, args.gen_len, layers, args.seed, args.temperature, args.top_k,
                 args.repetition_penalty);
  }

  Timers agg;
  VerifyStats verify;
  RunResult last_run;

  for (int i = 0; i < args.repeat; ++i) {
    RunResult rr = RunOne(model, prompt, args.gen_len, layers, args.seed, args.temperature,
                          args.top_k, args.repetition_penalty);
    AddTimers(agg, rr.timers);
    verify.max_softmax_row_sum_error =
        std::max(verify.max_softmax_row_sum_error, rr.verify.max_softmax_row_sum_error);
    verify.has_nan = verify.has_nan || rr.verify.has_nan;
    last_run = std::move(rr);
  }

  const double repeat_d = static_cast<double>(args.repeat);
  const double total_s = agg.total_ms / 1000.0;
  const double tokps = (args.gen_len > 0 && total_s > 0.0)
                           ? (static_cast<double>(args.gen_len * args.repeat) / total_s)
                           : 0.0;

  std::cout << "\n[Timing average over repeat]\n";
  std::cout << "total_ms: " << agg.total_ms / repeat_d << "\n";
  std::cout << "embed_ms: " << agg.embed_ms / repeat_d << "\n";
  std::cout << "ln1_ms: " << agg.ln1_ms / repeat_d << "\n";
  std::cout << "qkv_ms: " << agg.qkv_ms / repeat_d << "\n";
  std::cout << "attn_ms: " << agg.attn_ms / repeat_d << "\n";
  std::cout << "attn_proj_ms: " << agg.attn_proj_ms / repeat_d << "\n";
  std::cout << "ln2_ms: " << agg.ln2_ms / repeat_d << "\n";
  std::cout << "mlp_ms: " << agg.mlp_ms / repeat_d << "\n";
  std::cout << "final_ln_ms: " << agg.final_ln_ms / repeat_d << "\n";
  std::cout << "lm_head_ms: " << agg.lm_head_ms / repeat_d << "\n";
  std::cout << "generated_tokens_per_sec: " << tokps << "\n";

  std::cout << "\n[Verification]\n";
  std::cout << "has_nan_or_inf: " << (verify.has_nan ? "true" : "false") << "\n";
  std::cout << "max_softmax_row_sum_error: " << verify.max_softmax_row_sum_error << "\n";
  std::cout << "last_logits_checksum100: " << ChecksumPrefix(last_run.last_logits, 100) << "\n";

  const std::vector<int> inspect_ids = ParseCsvIds(args.inspect_logit_ids_csv);
  if (!inspect_ids.empty()) {
    std::cout << "inspected_logits:";
    for (int id : inspect_ids) {
      if (id < 0 || id >= static_cast<int>(last_run.last_logits.size())) {
        std::cout << " [" << id << ":out_of_range]";
      } else {
        std::cout << " [" << id << ":" << last_run.last_logits[id] << "]";
      }
    }
    std::cout << "\n";
  }

  if (!args.dump_last_logits_file.empty()) {
    std::ofstream fout(args.dump_last_logits_file);
    if (!fout) {
      throw std::runtime_error("Cannot open dump file: " + args.dump_last_logits_file);
    }
    for (size_t i = 0; i < last_run.last_logits.size(); ++i) {
      fout << i << " " << last_run.last_logits[i] << "\n";
    }
  }
  std::cout << "generated_token_ids:";
  for (int id : last_run.generated_tokens) {
    std::cout << " " << id;
  }
  std::cout << "\n";
  if (has_tokenizer) {
    std::cout << "generated_text: " << tokenizer.Decode(last_run.generated_tokens) << "\n";
  }

  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return MainImpl(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
