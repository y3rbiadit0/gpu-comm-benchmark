#pragma once

#include <cmath>
#include <cstddef>

namespace comm_playground {

inline bool nearly_equal(float lhs, float rhs, float tolerance = 1.0e-5F) {
  return std::fabs(lhs - rhs) <= tolerance * std::fmax(1.0F, std::fmax(std::fabs(lhs), std::fabs(rhs)));
}

inline bool validate_vector_add(const float* values, std::size_t count, std::size_t global_offset) {
  for (std::size_t i = 0; i < count; ++i) {
    const auto index = static_cast<float>(global_offset + i);
    if (!nearly_equal(values[i], 3.0F * index)) {
      return false;
    }
  }
  return true;
}

}  // namespace comm_playground
