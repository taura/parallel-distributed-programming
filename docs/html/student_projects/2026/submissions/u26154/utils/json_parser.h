#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct JsonValue {
  enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };
  Type type = Type::kNull;
  bool bool_value = false;
  double number_value = 0.0;
  std::string string_value;
  std::vector<JsonValue> array_value;
  std::unordered_map<std::string, std::shared_ptr<JsonValue>> object_value;

  const JsonValue& Get(const std::string& key) const;
};

class JsonParser {
 public:
  explicit JsonParser(std::string input);
  JsonValue Parse();

 private:
  JsonValue ParseValue();
  JsonValue ParseObject();
  JsonValue ParseArray();
  std::string ParseString();
  double ParseNumber();
  void SkipWs();
  bool Peek(char c) const;
  void Consume(char c);
  void Expect(const char* word);

  std::string input_;
  size_t pos_ = 0;
};
