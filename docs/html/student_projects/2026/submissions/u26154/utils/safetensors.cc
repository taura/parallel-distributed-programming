#include "safetensors.h"

#include <cstring>
#include <stdexcept>

#include "file_utils.h"
#include "json_parser.h"

namespace {

uint64_t ReadU64LE(const char* p) {
  uint64_t x = 0;
  for (int i = 0; i < 8; ++i) {
    x |= (static_cast<uint64_t>(static_cast<unsigned char>(p[i])) << (8 * i));
  }
  return x;
}

float HalfToFloat(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp = (h & 0x7C00u) >> 10;
  uint32_t mant = h & 0x03FFu;

  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;
    } else {
      exp = 1;
      while ((mant & 0x0400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x03FFu;
      const uint32_t exp_f = exp + (127 - 15);
      f = sign | (exp_f << 23) | (mant << 13);
    }
  } else if (exp == 0x1Fu) {
    f = sign | 0x7F800000u | (mant << 13);
  } else {
    const uint32_t exp_f = exp + (127 - 15);
    f = sign | (exp_f << 23) | (mant << 13);
  }

  float out;
  std::memcpy(&out, &f, sizeof(float));
  return out;
}

}  // namespace

void SafeTensors::Load(const std::string& path) {
  data_ = ReadFileToString(path);
  if (data_.size() < 8) {
    throw std::runtime_error("Invalid safetensors file: too small");
  }
  const uint64_t header_len = ReadU64LE(data_.data());
  if (8 + header_len > data_.size()) {
    throw std::runtime_error("Invalid safetensors file: header length");
  }
  const std::string header = data_.substr(8, header_len);
  const JsonValue root = JsonParser(header).Parse();
  if (root.type != JsonValue::Type::kObject) {
    throw std::runtime_error("Invalid safetensors header JSON");
  }
  const uint64_t data_base = 8 + header_len;
  tensors_.clear();
  for (const auto& kv : root.object_value) {
    if (kv.first == "__metadata__") {
      continue;
    }
    const JsonValue& tv = *kv.second;
    TensorInfo ti;
    ti.dtype = tv.Get("dtype").string_value;
    for (const JsonValue& d : tv.Get("shape").array_value) {
      ti.shape.push_back(static_cast<int64_t>(d.number_value));
    }
    const auto& offs = tv.Get("data_offsets").array_value;
    if (offs.size() != 2) {
      throw std::runtime_error("Invalid data_offsets");
    }
    ti.begin = data_base + static_cast<uint64_t>(offs[0].number_value);
    ti.end = data_base + static_cast<uint64_t>(offs[1].number_value);
    tensors_[kv.first] = std::move(ti);
  }
}

std::vector<float> SafeTensors::LoadAsF32(const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    throw std::runtime_error("Tensor not found: " + name);
  }
  const TensorInfo& t = it->second;
  size_t n = 1;
  for (int64_t d : t.shape) {
    n *= static_cast<size_t>(d);
  }
  std::vector<float> out(n);
  if (t.dtype == "F16") {
    const size_t need = n * 2;
    if (t.begin + need > data_.size() || t.begin + need != t.end) {
      throw std::runtime_error("Invalid F16 tensor length: " + name);
    }
    const unsigned char* p = reinterpret_cast<const unsigned char*>(data_.data() + t.begin);
    for (size_t i = 0; i < n; ++i) {
      const uint16_t h = static_cast<uint16_t>(p[2 * i]) |
                         (static_cast<uint16_t>(p[2 * i + 1]) << 8);
      out[i] = HalfToFloat(h);
    }
    return out;
  }
  if (t.dtype == "F32") {
    const size_t need = n * 4;
    if (t.begin + need > data_.size() || t.begin + need != t.end) {
      throw std::runtime_error("Invalid F32 tensor length: " + name);
    }
    std::memcpy(out.data(), data_.data() + t.begin, need);
    return out;
  }
  throw std::runtime_error("Unsupported dtype: " + t.dtype + " for " + name);
}
