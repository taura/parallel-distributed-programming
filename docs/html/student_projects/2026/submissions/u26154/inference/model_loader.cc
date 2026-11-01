#include "model_loader.h"

#include <fstream>
#include <stdexcept>

#include "../utils/file_utils.h"
#include "../utils/json_parser.h"
#include "../utils/safetensors.h"

ModelConfig LoadConfig(const std::string& config_path) {
  const JsonValue root = JsonParser(ReadFileToString(config_path)).Parse();
  ModelConfig c;
  c.hidden_size = static_cast<int>(root.Get("hidden_size").number_value);
  c.intermediate_size = static_cast<int>(root.Get("intermediate_size").number_value);
  c.num_attention_heads = static_cast<int>(root.Get("num_attention_heads").number_value);
  c.num_hidden_layers = static_cast<int>(root.Get("num_hidden_layers").number_value);
  c.max_position_embeddings = static_cast<int>(root.Get("max_position_embeddings").number_value);
  c.vocab_size = static_cast<int>(root.Get("vocab_size").number_value);
  c.layer_norm_eps = root.Get("layer_norm_eps").number_value;
  c.rotary_emb_base = root.Get("rotary_emb_base").number_value;
  c.rotary_pct = root.Get("rotary_pct").number_value;
  c.use_parallel_residual = root.Get("use_parallel_residual").bool_value;
  return c;
}

ModelWeights LoadModel(const std::string& model_dir) {
  ModelWeights m;
  m.cfg = LoadConfig(model_dir + "/config.json");
  SafeTensors st;
  st.Load(model_dir + "/model.safetensors");

  m.embed_in = st.LoadAsF32("gpt_neox.embed_in.weight");
  m.final_ln_w = st.LoadAsF32("gpt_neox.final_layer_norm.weight");
  m.final_ln_b = st.LoadAsF32("gpt_neox.final_layer_norm.bias");
  m.embed_out = st.LoadAsF32("embed_out.weight");

  m.layers.resize(m.cfg.num_hidden_layers);
  for (int l = 0; l < m.cfg.num_hidden_layers; ++l) {
    LayerWeights& w = m.layers[l];
    const std::string p = "gpt_neox.layers." + std::to_string(l) + ".";
    w.input_ln_w = st.LoadAsF32(p + "input_layernorm.weight");
    w.input_ln_b = st.LoadAsF32(p + "input_layernorm.bias");
    w.post_ln_w = st.LoadAsF32(p + "post_attention_layernorm.weight");
    w.post_ln_b = st.LoadAsF32(p + "post_attention_layernorm.bias");
    w.qkv_w = st.LoadAsF32(p + "attention.query_key_value.weight");
    w.qkv_b = st.LoadAsF32(p + "attention.query_key_value.bias");
    w.attn_out_w = st.LoadAsF32(p + "attention.dense.weight");
    w.attn_out_b = st.LoadAsF32(p + "attention.dense.bias");
    w.mlp_in_w = st.LoadAsF32(p + "mlp.dense_h_to_4h.weight");
    w.mlp_in_b = st.LoadAsF32(p + "mlp.dense_h_to_4h.bias");
    w.mlp_out_w = st.LoadAsF32(p + "mlp.dense_4h_to_h.weight");
    w.mlp_out_b = st.LoadAsF32(p + "mlp.dense_4h_to_h.bias");
  }

  return m;
}

std::vector<int> LoadPromptIds(const std::string& path, int vocab_size) {
  std::ifstream fin(path);
  if (!fin) {
    throw std::runtime_error("Cannot open prompt file: " + path);
  }
  std::vector<int> ids;
  int x;
  while (fin >> x) {
    if (x < 0 || x >= vocab_size) {
      throw std::runtime_error("Prompt token out of range: " + std::to_string(x));
    }
    ids.push_back(x);
  }
  if (ids.empty()) {
    throw std::runtime_error("Prompt file has no token ids");
  }
  return ids;
}
