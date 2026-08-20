#pragma once

#include <chrono>
#include <cstddef>
#include <stdexcept>
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

/* Samples to record for one batch case.
 *
 * The two cases need different sample counts to carry comparable confidence.
 * A `steady` sample already averages `batch_iters` operations inside itself, so
 * ten of them is a tight estimate. An `isolated` sample is one exchange: the
 * barrier-exit skew that `before_batch` leaves behind, and any one-off jitter,
 * land in it undiluted. Ten such samples is far too thin a series to fit the
 * latency intercept from - and the intercept is the number the crossover
 * analysis rests on.
 *
 * Rather than equalize total work (which would mean thousands of validated
 * single-exchange batches, each paying a device-to-host halo copy), give the
 * isolated case a larger fixed count and leave the steady case as it was. */
inline int batch_samples_for(int batch_iterations, int steady_samples, int isolated_samples) {
  if (steady_samples <= 0 || isolated_samples <= 0) {
    throw std::invalid_argument("batch sample counts must be positive");
  }
  return batch_iterations == 1 ? isolated_samples : steady_samples;
}

inline std::vector<int> batch_iteration_counts(int steady_iterations) {
  if (steady_iterations <= 0) {
    throw std::invalid_argument("steady iteration count must be positive");
  }
  return steady_iterations == 1 ? std::vector<int>{1} : std::vector<int>{1, steady_iterations};
}

// Measures completed batches and reports amortized time per logical operation.
// `before_batch` and `after_batch` run outside timing. They should align/reset
// participants and validate completion respectively. `body(count)` must
// complete all `count` ordered operations.
template <typename BeforeBatch, typename Body, typename AfterBatch>
bench_stats run_batched_benchmark(int warmup_iterations, int batch_iterations,
                                  int timed_batches, BeforeBatch&& before_batch, Body&& body,
                                  AfterBatch&& after_batch) {
  if (warmup_iterations < 0 || batch_iterations <= 0 || timed_batches <= 0) {
    throw std::invalid_argument("invalid batched benchmark iteration count");
  }

  if (warmup_iterations > 0) {
    before_batch();
    body(warmup_iterations);
  }

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(timed_batches));
  for (int batch = 0; batch < timed_batches; ++batch) {
    before_batch();
    wall_timer timer;
    body(batch_iterations);
    samples.push_back(timer.seconds() / static_cast<double>(batch_iterations));
    after_batch();
  }

  return summarize(std::move(samples));
}

}  // namespace gpu_bench
