#pragma once

#include <vector>

#include "types.h"

void SelfAttention(const std::vector<float>& ln_x, const LayerWeights& w, const ModelConfig& cfg,
                   VerifyStats& verify, Timers& timers, std::vector<float>& out);
