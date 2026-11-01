#pragma once

#include <vector>

#include "types.h"

RunResult RunOne(const ModelWeights& model, const std::vector<int>& prompt, int gen_len,
                 int layers_to_run, int seed, float temperature, int top_k,
                 float repetition_penalty);
void AddTimers(Timers& a, const Timers& b);
