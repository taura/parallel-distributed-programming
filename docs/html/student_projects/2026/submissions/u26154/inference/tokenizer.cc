#include "tokenizer.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "../utils/file_utils.h"
#include "../utils/json_parser.h"

namespace {

void InitByteLevelMaps(std::vector<std::string>& byte_to_unicode,
                       std::vector<int>& unicode_to_byte) {
  byte_to_unicode.assign(256, "");
  unicode_to_byte.assign(65536, -1);

  std::vector<int> bs;
  for (int i = static_cast<int>('!'); i <= static_cast<int>('~'); ++i) {
    bs.push_back(i);
  }
  for (int i = 0xA1; i <= 0xAC; ++i) {
    bs.push_back(i);
  }
  for (int i = 0xAE; i <= 0xFF; ++i) {
    bs.push_back(i);
  }
  std::vector<int> cs = bs;
  std::unordered_set<int> in_bs(bs.begin(), bs.end());
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    if (in_bs.find(b) == in_bs.end()) {
      bs.push_back(b);
      cs.push_back(256 + n);
      ++n;
    }
  }
  for (size_t i = 0; i < bs.size(); ++i) {
    int cp = cs[i];
    std::string ch;
    if (cp <= 0x7F) {
      ch.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
      ch.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      ch.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
      ch.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      ch.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      ch.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      ch.push_back(static_cast<char>(0xF0 | (cp >> 18)));
      ch.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      ch.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      ch.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }

    byte_to_unicode[bs[i]] = ch;
    if (cp < static_cast<int>(unicode_to_byte.size())) {
      unicode_to_byte[cp] = bs[i];
    }
  }
}

void BuildIdToToken(const std::unordered_map<std::string, int>& vocab,
                    std::vector<std::string>& id_to_token) {
  int max_id = -1;
  for (const auto& kv : vocab) {
    max_id = std::max(max_id, kv.second);
  }
  id_to_token.assign(static_cast<size_t>(max_id + 1), "");
  for (const auto& kv : vocab) {
    id_to_token[static_cast<size_t>(kv.second)] = kv.first;
  }
}

}  // namespace

std::string ByteLevelBpeTokenizer::Utf8FromCodePoint(int cp) {
  std::string out;
  if (cp <= 0x7F) {
    out.push_back(static_cast<char>(cp));
  } else if (cp <= 0x7FF) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp <= 0xFFFF) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
  return out;
}

std::vector<std::string> ByteLevelBpeTokenizer::SplitUtf8CodePoints(const std::string& s) {
  std::vector<std::string> out;
  for (size_t i = 0; i < s.size();) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    size_t len = 1;
    if ((c & 0x80u) == 0x00u) {
      len = 1;
    } else if ((c & 0xE0u) == 0xC0u) {
      len = 2;
    } else if ((c & 0xF0u) == 0xE0u) {
      len = 3;
    } else if ((c & 0xF8u) == 0xF0u) {
      len = 4;
    }
    if (i + len > s.size()) {
      len = 1;
    }
    out.push_back(s.substr(i, len));
    i += len;
  }
  return out;
}

std::vector<std::string> ByteLevelBpeTokenizer::SplitTextChunks(const std::string& text) {
  std::vector<std::string> chunks;
  if (text.empty()) {
    return chunks;
  }
  size_t i = 0;
  while (i < text.size()) {
    const bool is_ws = std::isspace(static_cast<unsigned char>(text[i])) != 0;
    if (!is_ws) {
      size_t j = i;
      while (j < text.size() && std::isspace(static_cast<unsigned char>(text[j])) == 0) {
        ++j;
      }
      chunks.push_back(text.substr(i, j - i));
      i = j;
      continue;
    }

    size_t ws_begin = i;
    while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i])) != 0) {
      ++i;
    }
    if (i == text.size()) {
      chunks.push_back(text.substr(ws_begin, i - ws_begin));
      break;
    }

    size_t token_end = i;
    while (token_end < text.size() && std::isspace(static_cast<unsigned char>(text[token_end])) == 0) {
      ++token_end;
    }
    chunks.push_back(text.substr(ws_begin, token_end - ws_begin));
    i = token_end;
  }
  return chunks;
}

std::string ByteLevelBpeTokenizer::PairKey(const std::string& a, const std::string& b) {
  return a + "\n" + b;
}

std::vector<std::string> ByteLevelBpeTokenizer::Bpe(const std::string& token) const {
  std::vector<std::string> symbols = SplitUtf8CodePoints(token);
  if (symbols.size() <= 1) {
    return symbols;
  }

  while (true) {
    int best_rank = std::numeric_limits<int>::max();
    int best_i = -1;
    for (int i = 0; i + 1 < static_cast<int>(symbols.size()); ++i) {
      const auto it = bpe_rank_.find(PairKey(symbols[i], symbols[i + 1]));
      if (it != bpe_rank_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_i = i;
      }
    }
    if (best_i < 0) {
      break;
    }
    std::vector<std::string> merged;
    merged.reserve(symbols.size() - 1);
    for (int i = 0; i < static_cast<int>(symbols.size());) {
      if (i == best_i) {
        merged.push_back(symbols[i] + symbols[i + 1]);
        i += 2;
      } else {
        merged.push_back(symbols[i]);
        ++i;
      }
    }
    symbols.swap(merged);
    if (symbols.size() <= 1) {
      break;
    }
  }
  return symbols;
}

void ByteLevelBpeTokenizer::LoadFromTokenizerJson(const std::string& path) {
  const JsonValue root = JsonParser(ReadFileToString(path)).Parse();

  InitByteLevelMaps(byte_to_unicode_, unicode_to_byte_);
  vocab_.clear();
  bpe_rank_.clear();
  id_to_token_.clear();

  const JsonValue& model = root.Get("model");
  const JsonValue& vocab = model.Get("vocab");
  for (const auto& kv : vocab.object_value) {
    const int id = static_cast<int>(kv.second->number_value);
    vocab_[kv.first] = id;
  }

  if (root.object_value.find("added_tokens") != root.object_value.end()) {
    for (const JsonValue& tok : root.Get("added_tokens").array_value) {
      const std::string content = tok.Get("content").string_value;
      const int id = static_cast<int>(tok.Get("id").number_value);
      vocab_[content] = id;
    }
  }

  BuildIdToToken(vocab_, id_to_token_);

  const JsonValue& merges = model.Get("merges");
  for (int i = 0; i < static_cast<int>(merges.array_value.size()); ++i) {
    const std::string pair = merges.array_value[i].string_value;
    const size_t sp = pair.find(' ');
    if (sp == std::string::npos) {
      continue;
    }
    const std::string a = pair.substr(0, sp);
    const std::string b = pair.substr(sp + 1);
    bpe_rank_[PairKey(a, b)] = i;
  }
}

void ByteLevelBpeTokenizer::LoadFromVocabAndMerges(const std::string& vocab_json_path,
                                                   const std::string& merges_txt_path) {
  const JsonValue vocab_root = JsonParser(ReadFileToString(vocab_json_path)).Parse();
  if (vocab_root.type != JsonValue::Type::kObject) {
    throw std::runtime_error("Invalid vocab.json");
  }

  InitByteLevelMaps(byte_to_unicode_, unicode_to_byte_);
  vocab_.clear();
  bpe_rank_.clear();
  id_to_token_.clear();

  for (const auto& kv : vocab_root.object_value) {
    const int id = static_cast<int>(kv.second->number_value);
    vocab_[kv.first] = id;
  }
  BuildIdToToken(vocab_, id_to_token_);

  std::ifstream fin(merges_txt_path);
  if (!fin) {
    throw std::runtime_error("Cannot open merges file: " + merges_txt_path);
  }
  std::string line;
  int rank = 0;
  while (std::getline(fin, line)) {
    if (line.empty()) {
      continue;
    }
    if (line[0] == '#') {
      continue;
    }
    std::istringstream iss(line);
    std::string a;
    std::string b;
    if (!(iss >> a >> b)) {
      continue;
    }
    bpe_rank_[PairKey(a, b)] = rank;
    ++rank;
  }
}

std::vector<int> ByteLevelBpeTokenizer::Encode(const std::string& text) const {
  std::vector<int> ids;
  for (const std::string& chunk : SplitTextChunks(text)) {
    std::string mapped;
    mapped.reserve(chunk.size() * 2);
    for (unsigned char b : chunk) {
      mapped += byte_to_unicode_[b];
    }
    std::vector<std::string> pieces = Bpe(mapped);
    for (const std::string& p : pieces) {
      auto it = vocab_.find(p);
      if (it != vocab_.end()) {
        ids.push_back(it->second);
      } else {
        std::vector<std::string> chars = SplitUtf8CodePoints(p);
        for (const std::string& ch : chars) {
          auto it2 = vocab_.find(ch);
          if (it2 == vocab_.end()) {
            throw std::runtime_error("Tokenizer cannot encode piece");
          }
          ids.push_back(it2->second);
        }
      }
    }
  }
  return ids;
}

std::string ByteLevelBpeTokenizer::Decode(const std::vector<int>& ids) const {
  std::string merged;
  for (int id : ids) {
    if (id >= 0 && id < static_cast<int>(id_to_token_.size())) {
      merged += id_to_token_[id];
    }
  }

  std::string out;
  const std::vector<std::string> cps = SplitUtf8CodePoints(merged);
  for (const std::string& cp_s : cps) {
    const unsigned char c0 = static_cast<unsigned char>(cp_s[0]);
    int cp = -1;
    if ((c0 & 0x80u) == 0x00u) {
      cp = c0;
    } else if ((c0 & 0xE0u) == 0xC0u && cp_s.size() >= 2) {
      cp = ((c0 & 0x1Fu) << 6) | (static_cast<unsigned char>(cp_s[1]) & 0x3Fu);
    } else if ((c0 & 0xF0u) == 0xE0u && cp_s.size() >= 3) {
      cp = ((c0 & 0x0Fu) << 12) | ((static_cast<unsigned char>(cp_s[1]) & 0x3Fu) << 6) |
           (static_cast<unsigned char>(cp_s[2]) & 0x3Fu);
    } else if ((c0 & 0xF8u) == 0xF0u && cp_s.size() >= 4) {
      cp = ((c0 & 0x07u) << 18) | ((static_cast<unsigned char>(cp_s[1]) & 0x3Fu) << 12) |
           ((static_cast<unsigned char>(cp_s[2]) & 0x3Fu) << 6) |
           (static_cast<unsigned char>(cp_s[3]) & 0x3Fu);
    }
    if (cp >= 0 && cp < static_cast<int>(unicode_to_byte_.size()) && unicode_to_byte_[cp] >= 0) {
      out.push_back(static_cast<char>(unicode_to_byte_[cp]));
    }
  }
  return out;
}
