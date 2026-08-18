#pragma once

#include <cmath>
#include <cstddef>

namespace gpu_bench {

inline bool nearly_equal(float lhs, float rhs, float tolerance = 1.0e-5F) {
  return std::fabs(lhs - rhs) <= tolerance * std::fmax(1.0F, std::fmax(std::fabs(lhs), std::fabs(rhs)));
}

inline bool nearly_equal(double lhs, double rhs, double tolerance = 1.0e-9) {
  return std::fabs(lhs - rhs) <= tolerance * std::fmax(1.0, std::fmax(std::fabs(lhs), std::fabs(rhs)));
}

// All-to-all permutation check. On rank `r` the block destined for rank `s` is
// filled with r*ranks + s; after the exchange rank `r`'s block received from rank
// `s` must equal s*ranks + r. Buffers hold ranks * count elements.
inline void fill_alltoall_send(float* send, int rank, int ranks, std::size_t count) {
  for (int s = 0; s < ranks; ++s) {
    const auto value = static_cast<float>(rank * ranks + s);
    for (std::size_t i = 0; i < count; ++i) {
      send[static_cast<std::size_t>(s) * count + i] = value;
    }
  }
}

inline bool validate_alltoall(const float* recv, int rank, int ranks, std::size_t count) {
  for (int s = 0; s < ranks; ++s) {
    const auto expected = static_cast<float>(s * ranks + rank);
    for (std::size_t i = 0; i < count; ++i) {
      if (!nearly_equal(recv[static_cast<std::size_t>(s) * count + i], expected)) {
        return false;
      }
    }
  }
  return true;
}

inline bool validate_halo_1d(const float* values, std::size_t count, std::size_t global_offset,
                             std::size_t global_size) {
  for (std::size_t i = 0; i < count; ++i) {
    const auto global_i = global_offset + i;
    const auto left = global_i == 0 ? 0.0F : static_cast<float>(global_i - 1U);
    const auto center = static_cast<float>(global_i);
    const auto right = global_i + 1U == global_size ? 0.0F : static_cast<float>(global_i + 1U);
    const auto expected = 0.25F * left + 0.5F * center + 0.25F * right;
    if (!nearly_equal(values[i], expected)) {
      return false;
    }
  }
  return true;
}

}  // namespace gpu_bench
