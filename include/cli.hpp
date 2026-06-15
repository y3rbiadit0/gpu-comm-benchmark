#pragma once

#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>

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

}  // namespace comm_playground
