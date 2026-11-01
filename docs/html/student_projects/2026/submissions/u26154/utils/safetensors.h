#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct TensorInfo {
  std::string dtype;
  std::vector<int64_t> shape;
  uint64_t begin = 0;
  uint64_t end = 0;
};

class SafeTensors {
 public:
  void Load(const std::string& path);
  std::vector<float> LoadAsF32(const std::string& name) const;

 private:
  std::string data_;
  std::unordered_map<std::string, TensorInfo> tensors_;
};
