#pragma once

#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace comm_playground {

inline std::size_t parse_size_arg(int argc, char** argv, std::size_t default_value) {
  if (argc < 2) {
    return default_value;
  }

  char* end = nullptr;
  const auto value = std::strtoull(argv[1], &end, 10);
  if (end == argv[1] || *end != '\0' || value == 0) {
    throw std::invalid_argument("expected a positive vector size");
  }

  return static_cast<std::size_t>(value);
}

inline std::vector<std::size_t> default_power_of_two_sizes(std::size_t max_value) {
  std::vector<std::size_t> sizes;
  for (std::size_t size = 1; size <= max_value; size *= 2U) {
    sizes.push_back(size);
    if (size > max_value / 2U) {
      break;
    }
  }
  return sizes;
}

inline std::size_t parse_size_token(const std::string& token, std::size_t max_value) {
  if (token.empty()) {
    throw std::invalid_argument("expected a comma-separated list of positive message sizes");
  }

  char* end = nullptr;
  const auto value = std::strtoull(token.c_str(), &end, 10);
  if (end == token.c_str() || *end != '\0' || value == 0) {
    throw std::invalid_argument("expected a comma-separated list of positive message sizes");
  }
  if (value > max_value) {
    throw std::invalid_argument("message size exceeds maximum message size");
  }

  return static_cast<std::size_t>(value);
}

inline std::vector<std::size_t> parse_size_list_arg(int argc, char** argv, int index,
                                                    std::size_t max_value) {
  if (argc <= index) {
    return default_power_of_two_sizes(max_value);
  }

  const std::string input(argv[index]);
  std::vector<std::size_t> sizes;
  std::size_t start = 0;
  while (start <= input.size()) {
    const auto comma = input.find(',', start);
    const auto end = comma == std::string::npos ? input.size() : comma;
    sizes.push_back(parse_size_token(input.substr(start, end - start), max_value));
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1U;
  }
  if (sizes.empty()) {
    throw std::invalid_argument("expected a comma-separated list of positive message sizes");
  }
  return sizes;
}

inline int parse_positive_int_arg(int argc, char** argv, int index, int default_value) {
  if (argc <= index) {
    return default_value;
  }

  char* end = nullptr;
  const auto value = std::strtol(argv[index], &end, 10);
  if (end == argv[index] || *end != '\0' || value <= 0) {
    throw std::invalid_argument("expected a positive integer argument");
  }

  return static_cast<int>(value);
}

}  // namespace comm_playground
