#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "stats/collective.hpp"
#include "timing.hpp"

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool near(double lhs, double rhs, double tolerance = 1.0e-12) {
  return std::abs(lhs - rhs) <= tolerance;
}

void test_average_of_iteration_maxima() {
  const auto local = gpu_bench::summarize({10.0, 1.0, 8.0});
  const std::vector<double> peer = {1.0, 11.0, 2.0};

  const auto global = gpu_bench::collective_stats(
      local, [&](const double* input, double* output, std::size_t count) {
        require(count == peer.size(), "collective received the wrong sample count");
        for (std::size_t i = 0; i < count; ++i) {
          output[i] = std::max(input[i], peer[i]);
        }
      });

  const std::vector<double> expected = {10.0, 11.0, 8.0};
  require(global.samples == expected, "collective did not preserve iteration-wise maxima");
  require(near(global.total_s, 29.0), "global total is not based on iteration-wise maxima");
  require(near(global.avg_s, 29.0 / 3.0), "headline is not AVG(MAX per iteration)");
  require(near(global.min_s, 8.0), "minimum is not from the global series");
  require(near(global.max_s, 11.0), "maximum is not from the global series");
  require(near(global.median_s, 10.0), "median is not from the global series");
  require(near(global.p25_s, 8.0), "p25 is not from the global series");
  require(near(global.p75_s, 11.0), "p75 is not from the global series");
  require(near(global.stddev_s, std::sqrt(7.0 / 3.0)),
          "standard deviation is not from the global series");

  const double max_of_rank_averages = std::max(local.avg_s, gpu_bench::summarize(peer).avg_s);
  require(!near(global.avg_s, max_of_rank_averages),
          "test data does not distinguish AVG(MAX) from MAX(AVG)");
}

void test_wall_timing_samples_completed_bodies() {
  int calls = 0;
  const auto stats = gpu_bench::run_benchmark(2, 3, [&]() {
    ++calls;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  });

  require(calls == 5, "benchmark did not execute every warmup and timed body");
  require(stats.iterations == 3, "benchmark reported the wrong timed iteration count");
  require(stats.samples.size() == 3U, "benchmark did not retain one sample per timed body");
  require(std::all_of(stats.samples.begin(), stats.samples.end(),
                      [](double sample) { return sample > 0.0; }),
          "benchmark recorded a non-positive elapsed time");
  require(near(stats.total_s, stats.samples[0] + stats.samples[1] + stats.samples[2]),
          "benchmark total does not match its raw samples");
}

}  // namespace

int main() {
  test_average_of_iteration_maxima();
  test_wall_timing_samples_completed_bodies();
  return 0;
}
