#pragma once

#include <chrono>
#include <cstddef>
#include <utility>
#include <vector>

#include "stats/summary.hpp"

namespace gpu_bench {

class wall_timer {
 public:
  wall_timer() : start_(clock::now()) {}

  double seconds() const {
    return std::chrono::duration<double>(clock::now() - start_).count();
  }

 private:
  using clock = std::chrono::steady_clock;
  clock::time_point start_;
};

// Runs `body` for `warmup_iterations` untimed calls, then `timed_iterations`
// timed calls, returning the local timing distribution. `body` must perform one
// fully-completed communication step (including any device synchronization), so
// each recorded sample reflects a finished operation.
template <typename Body>
bench_stats run_benchmark(int warmup_iterations, int timed_iterations, Body&& body) {
  for (int i = 0; i < warmup_iterations; ++i) {
    body();
  }

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(timed_iterations > 0 ? timed_iterations : 0));
  for (int i = 0; i < timed_iterations; ++i) {
    wall_timer timer;
    body();
    samples.push_back(timer.seconds());
  }

  return summarize(std::move(samples));
}

}  // namespace gpu_bench
