#include "json_parser.h"

#include <memory>
#include <stdexcept>

const JsonValue& JsonValue::Get(const std::string& key) const {
  auto it = object_value.find(key);
  if (it == object_value.end()) {
    throw std::runtime_error("Missing JSON key: " + key);
  }
  return *it->second;
}

JsonParser::JsonParser(std::string input) : input_(std::move(input)) {}

JsonValue JsonParser::Parse() {
  SkipWs();
  JsonValue v = ParseValue();
  SkipWs();
  if (pos_ != input_.size()) {
    throw std::runtime_error("Unexpected trailing JSON content");
  }
  return v;
}

JsonValue JsonParser::ParseValue() {
  SkipWs();
  if (pos_ >= input_.size()) {
    throw std::runtime_error("Unexpected EOF in JSON");
  }
  char c = input_[pos_];
  if (c == '{') {
    return ParseObject();
  }
  if (c == '[') {
    return ParseArray();
  }
  if (c == '"') {
    JsonValue v;
    v.type = JsonValue::Type::kString;
    v.string_value = ParseString();
    return v;
  }
  if (c == 't') {
    Expect("true");
    JsonValue v;
    v.type = JsonValue::Type::kBool;
    v.bool_value = true;
    return v;
  }
  if (c == 'f') {
    Expect("false");
    JsonValue v;
    v.type = JsonValue::Type::kBool;
    v.bool_value = false;
    return v;
  }
  if (c == 'n') {
    Expect("null");
    JsonValue v;
    v.type = JsonValue::Type::kNull;
    return v;
  }
  if (c == '-' || (c >= '0' && c <= '9')) {
    JsonValue v;
    v.type = JsonValue::Type::kNumber;
    v.number_value = ParseNumber();
    return v;
  }
  throw std::runtime_error("Invalid JSON value");
}

JsonValue JsonParser::ParseObject() {
  JsonValue v;
  v.type = JsonValue::Type::kObject;
  Consume('{');
  SkipWs();
  if (Peek('}')) {
    Consume('}');
    return v;
  }
  while (true) {
    SkipWs();
    std::string key = ParseString();
    SkipWs();
    Consume(':');
    SkipWs();
    v.object_value[key] = std::make_shared<JsonValue>(ParseValue());
    SkipWs();
    if (Peek('}')) {
      Consume('}');
      break;
    }
    Consume(',');
  }
  return v;
}

JsonValue JsonParser::ParseArray() {
  JsonValue v;
  v.type = JsonValue::Type::kArray;
  Consume('[');
  SkipWs();
  if (Peek(']')) {
    Consume(']');
    return v;
  }
  while (true) {
    v.array_value.push_back(ParseValue());
    SkipWs();
    if (Peek(']')) {
      Consume(']');
      break;
    }
    Consume(',');
  }
  return v;
}

std::string JsonParser::ParseString() {
  Consume('"');
  std::string out;
  while (pos_ < input_.size()) {
    char c = input_[pos_++];
    if (c == '"') {
      return out;
    }
    if (c == '\\') {
      if (pos_ >= input_.size()) {
        throw std::runtime_error("Invalid escape");
      }
      char e = input_[pos_++];
      switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u':
          if (pos_ + 4 > input_.size()) {
            throw std::runtime_error("Invalid unicode escape");
          }
          pos_ += 4;
          out.push_back('?');
          break;
        default:
          throw std::runtime_error("Unknown escape");
      }
    } else {
      out.push_back(c);
    }
  }
  throw std::runtime_error("Unterminated string");
}

double JsonParser::ParseNumber() {
  size_t start = pos_;
  if (input_[pos_] == '-') {
    ++pos_;
  }
  if (pos_ >= input_.size()) {
    throw std::runtime_error("Invalid number");
  }
  if (input_[pos_] == '0') {
    ++pos_;
  } else {
    while (pos_ < input_.size() && input_[pos_] >= '0' && input_[pos_] <= '9') {
      ++pos_;
    }
  }
  if (pos_ < input_.size() && input_[pos_] == '.') {
    ++pos_;
    while (pos_ < input_.size() && input_[pos_] >= '0' && input_[pos_] <= '9') {
      ++pos_;
    }
  }
  if (pos_ < input_.size() && (input_[pos_] == 'e' || input_[pos_] == 'E')) {
    ++pos_;
    if (pos_ < input_.size() && (input_[pos_] == '+' || input_[pos_] == '-')) {
      ++pos_;
    }
    while (pos_ < input_.size() && input_[pos_] >= '0' && input_[pos_] <= '9') {
      ++pos_;
    }
  }
  return std::stod(input_.substr(start, pos_ - start));
}

void JsonParser::SkipWs() {
  while (pos_ < input_.size()) {
    char c = input_[pos_];
    if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
      ++pos_;
    } else {
      break;
    }
  }
}

bool JsonParser::Peek(char c) const {
  return pos_ < input_.size() && input_[pos_] == c;
}

void JsonParser::Consume(char c) {
  if (pos_ >= input_.size() || input_[pos_] != c) {
    throw std::runtime_error(std::string("Expected '") + c + "'");
  }
  ++pos_;
}

void JsonParser::Expect(const char* word) {
  while (*word) {
    if (pos_ >= input_.size() || input_[pos_] != *word) {
      throw std::runtime_error("Invalid JSON token");
    }
    ++pos_;
    ++word;
  }
}
