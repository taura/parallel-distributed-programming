#include "mlp_block.h"

#include "ops.h"

void MlpForward(const std::vector<float>& ln_x, const LayerWeights& w, const ModelConfig& cfg,
                Timers& timers, std::vector<float>& out) {
  const int t_len = static_cast<int>(ln_x.size()) / cfg.hidden_size;
  const int hidden = cfg.hidden_size;
  std::vector<float> tmp;

  auto t0 = std::chrono::steady_clock::now();
  Linear(ln_x, t_len, hidden, cfg.intermediate_size, w.mlp_in_w, w.mlp_in_b, tmp);
  ApplyGelu(tmp);
  Linear(tmp, t_len, cfg.intermediate_size, hidden, w.mlp_out_w, w.mlp_out_b, out);
  auto t1 = std::chrono::steady_clock::now();
  timers.mlp_ms += ElapsedMs(t0, t1);
}
