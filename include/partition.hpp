#pragma once

#include <algorithm>
#include <cstddef>

namespace gpu_bench {

// Block decomposition of `total` items across `ranks`: the first `total % ranks`
// ranks get one extra item. Used to split vectors (or grid columns) evenly.
inline std::size_t local_count(std::size_t total, int rank, int ranks) {
  const auto base = total / static_cast<std::size_t>(ranks);
  const auto remainder = total % static_cast<std::size_t>(ranks);
  return base + (static_cast<std::size_t>(rank) < remainder ? 1U : 0U);
}

inline std::size_t local_offset(std::size_t total, int rank, int ranks) {
  const auto base = total / static_cast<std::size_t>(ranks);
  const auto remainder = total % static_cast<std::size_t>(ranks);
  const auto rank_value = static_cast<std::size_t>(rank);
  return rank_value * base + std::min(rank_value, remainder);
}

}  // namespace gpu_bench
