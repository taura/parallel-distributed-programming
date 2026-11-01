#pragma once

#include <string>
#include <vector>

struct ModelConfig {
  int hidden_size = 0;
  int intermediate_size = 0;
  int num_attention_heads = 0;
  int num_hidden_layers = 0;
  int max_position_embeddings = 0;
  int vocab_size = 0;
  double layer_norm_eps = 1e-5;
  double rotary_emb_base = 10000.0;
  double rotary_pct = 0.25;
  bool use_parallel_residual = true;
};

struct LayerWeights {
  std::vector<float> input_ln_w;
  std::vector<float> input_ln_b;
  std::vector<float> post_ln_w;
  std::vector<float> post_ln_b;
  std::vector<float> qkv_w;
  std::vector<float> qkv_b;
  std::vector<float> attn_out_w;
  std::vector<float> attn_out_b;
  std::vector<float> mlp_in_w;
  std::vector<float> mlp_in_b;
  std::vector<float> mlp_out_w;
  std::vector<float> mlp_out_b;
};

struct ModelWeights {
  ModelConfig cfg;
  std::vector<float> embed_in;
  std::vector<float> final_ln_w;
  std::vector<float> final_ln_b;
  std::vector<float> embed_out;
  std::vector<LayerWeights> layers;
};

struct Timers {
  double total_ms = 0.0;
  double embed_ms = 0.0;
  double ln1_ms = 0.0;
  double qkv_ms = 0.0;
  double attn_ms = 0.0;
  double attn_proj_ms = 0.0;
  double ln2_ms = 0.0;
  double mlp_ms = 0.0;
  double final_ln_ms = 0.0;
  double lm_head_ms = 0.0;
};

struct VerifyStats {
  double max_softmax_row_sum_error = 0.0;
  bool has_nan = false;
};

struct RunResult {
  Timers timers;
  VerifyStats verify;
  std::vector<int> generated_tokens;
  std::vector<float> last_logits;
};

struct Args {
  std::string model_dir = "models/EleutherAI_pythia_14m";
  std::string prompt_file = "prompt_text.txt";
  std::string prompt_text;
  std::string tokenizer_json = "models/EleutherAI_pythia_14m/tokenizer.json";
  int gen_len = 16;
  int layers = -1;
  int warmup = 1;
  int repeat = 3;
  int seed = 1234;
  float temperature = 0.8f;
  int top_k = 40;
  float repetition_penalty = 1.1f;
  std::string dump_last_logits_file;
  std::string inspect_logit_ids_csv;
};
