#pragma once

#include <cstddef>
#include <cstdlib>
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

/* The reported name carries the backend: benchscribe splits it into
 * <backend>_<benchmark>. A binary that can be driven by more than one backend -
 * oneCCL, which dispatches to NCCL or OSHMPI at runtime - therefore cannot use a
 * fixed name, or the two would be indistinguishable in the results. Runtime
 * scripts set CP_REPORT_BACKEND to say which one is actually in use.
 *
 * Any override must still be a name benchscribe knows, otherwise the record is
 * skipped. */
inline std::string report_name(const bench_report& report) {
  const char* backend = std::getenv("CP_REPORT_BACKEND");
  if (!backend || !*backend) {
    return report.name;
  }

  // Replace the leading <backend>_ of the compiled-in name, keeping the
  // benchmark suffix: sycl_oneccl_allreduce -> sycl_oneccl_oshmpi_allreduce.
  const std::string original = report.name;
  const std::size_t split = original.rfind('_');
  if (split == std::string::npos) {
    return original;
  }
  return std::string(backend) + original.substr(split);
}

inline void print_report(const bench_report& report) {
  const double gbytes_per_s = (report.bytes_per_iter > 0 && report.time_per_iter_s > 0.0)
                                  ? static_cast<double>(report.bytes_per_iter) / report.time_per_iter_s / 1.0e9
                                  : 0.0;

  std::cout << report_name(report) << " n=" << report.n << " ranks=" << report.ranks
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
