#include "file_utils.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

std::string ReadFileToString(const std::string& path) {
  std::ifstream fin(path, std::ios::binary);
  if (!fin) {
    throw std::runtime_error("Cannot open file: " + path);
  }
  std::ostringstream ss;
  ss << fin.rdbuf();
  return ss.str();
}
