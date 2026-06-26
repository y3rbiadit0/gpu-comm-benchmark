#pragma once

#include <cstddef>
#include <iostream>
#include <string>

namespace comm_playground {

struct bench_report {
  const char* name = "";
  std::size_t n = 0;
  int ranks = 0;
  std::size_t bytes_per_iter = 0;   // bytes communicated per timed iteration
  int iterations = 0;
  int warmup = 0;
  double time_per_iter_s = 0.0;     // headline: slowest-rank average per iteration
  double min_s = 0.0;
  double max_s = 0.0;
  bool valid = true;
  std::string extra;                // optional extra key=value pairs (e.g. device="...")
};

inline void print_report(const bench_report& report) {
  const double gbytes_per_s = (report.bytes_per_iter > 0 && report.time_per_iter_s > 0.0)
                                  ? static_cast<double>(report.bytes_per_iter) / report.time_per_iter_s / 1.0e9
                                  : 0.0;

  std::cout << report.name << " n=" << report.n << " ranks=" << report.ranks
            << " bytes=" << report.bytes_per_iter << " iters=" << report.iterations
            << " warmup=" << report.warmup << " time_per_iter_s=" << report.time_per_iter_s
            << " usec=" << (report.time_per_iter_s * 1.0e6) << " min_usec=" << (report.min_s * 1.0e6)
            << " max_usec=" << (report.max_s * 1.0e6) << " gbytes_per_s=" << gbytes_per_s;
  if (!report.extra.empty()) {
    std::cout << ' ' << report.extra;
  }
  std::cout << " validation=" << (report.valid ? "PASS" : "FAIL") << '\n';
}

}  // namespace comm_playground
