#pragma once

#include <string>
#include <unordered_map>
#include <vector>

class ByteLevelBpeTokenizer {
 public:
  void LoadFromTokenizerJson(const std::string& path);
  void LoadFromVocabAndMerges(const std::string& vocab_json_path,
                              const std::string& merges_txt_path);
  std::vector<int> Encode(const std::string& text) const;
  std::string Decode(const std::vector<int>& ids) const;

 private:
  std::vector<std::string> byte_to_unicode_;
  std::vector<int> unicode_to_byte_;
  std::unordered_map<std::string, int> vocab_;
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int> bpe_rank_;

  static std::string Utf8FromCodePoint(int cp);
  static std::vector<std::string> SplitUtf8CodePoints(const std::string& s);
  static std::vector<std::string> SplitTextChunks(const std::string& text);
  static std::string PairKey(const std::string& a, const std::string& b);
  std::vector<std::string> Bpe(const std::string& token) const;
};
