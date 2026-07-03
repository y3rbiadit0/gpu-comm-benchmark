#pragma once

#include <cstddef>

#include "validation.hpp"

namespace comm_playground {

// Host-side helpers shared by the column-slab 2D stencil benchmarks (cg_step). A square S x S grid is split by columns; each rank stores its slab
// in a padded row-major array of width (local_cols + 2) with ghost columns at
// j = 0 and j = local_cols + 1.

// Field value at global (row i, column jg), or 0 outside the [0, side) domain.
template <typename FieldFn>
inline float domain_value(long i, long jg, std::size_t side, FieldFn field) {
  if (i < 0 || jg < 0 || static_cast<std::size_t>(i) >= side || static_cast<std::size_t>(jg) >= side) {
    return 0.0F;
  }
  return field(static_cast<std::size_t>(i), static_cast<std::size_t>(jg));
}

// One 5-point averaging stencil step on `field` at global (i, jg): the value the
// distributed SpMV/stencil must produce once the halo is correct.
template <typename FieldFn>
inline float stencil5(std::size_t i, std::size_t jg, std::size_t side, FieldFn field) {
  const auto li = static_cast<long>(i);
  const auto lj = static_cast<long>(jg);
  return 0.25F * (domain_value(li - 1, lj, side, field) + domain_value(li + 1, lj, side, field) +
                  domain_value(li, lj - 1, side, field) + domain_value(li, lj + 1, side, field));
}

// Check a padded column-slab array's interior against expected(i, global_column).
template <typename ExpectedFn>
inline bool validate_columns(const float* padded, std::size_t side, std::size_t local_cols, std::size_t width,
                             std::size_t col_offset, ExpectedFn expected) {
  for (std::size_t i = 0; i < side; ++i) {
    for (std::size_t jj = 0; jj < local_cols; ++jj) {
      if (!nearly_equal(padded[i * width + (jj + 1U)], expected(i, col_offset + jj))) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace comm_playground
