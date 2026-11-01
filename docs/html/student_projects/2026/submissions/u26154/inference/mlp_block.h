#pragma once

#include <vector>

#include "types.h"

void MlpForward(const std::vector<float>& ln_x, const LayerWeights& w, const ModelConfig& cfg,
                Timers& timers, std::vector<float>& out);
