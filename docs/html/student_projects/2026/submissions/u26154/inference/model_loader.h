#pragma once

#include <string>
#include <vector>

#include "types.h"

ModelConfig LoadConfig(const std::string& config_path);
ModelWeights LoadModel(const std::string& model_dir);
std::vector<int> LoadPromptIds(const std::string& path, int vocab_size);
