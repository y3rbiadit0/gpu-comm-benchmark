#pragma once

#include <mpi.h>

#include <cstddef>
#include <limits>
#include <stdexcept>

#include "stats/collective.hpp"

namespace gpu_bench {

// MPI adapter for collective_stats(): one MPI_Allreduce over the whole sample
// series. Collective over `comm` - every rank must call it with the same number
// of samples, and every rank gets the same stats back.
inline bench_stats collective_stats(const bench_stats& local, MPI_Comm comm = MPI_COMM_WORLD) {
  return collective_stats(local, [comm](const double* in, double* out, std::size_t count) {
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::length_error("timing sample count exceeds MPI count limit");
    }
    MPI_Allreduce(in, out, static_cast<int>(count), MPI_DOUBLE, MPI_MAX, comm);
  });
}

}  // namespace gpu_bench
