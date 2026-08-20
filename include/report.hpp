#pragma once

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>

#include "timing.hpp"

namespace gpu_bench {

struct bench_report {
  const char* name = "";
  std::size_t n = 0;
  int ranks = 0;
  std::size_t bytes_per_iter = 0;   // bytes communicated per timed iteration
  int iterations = 0;
  int warmup = 0;
  double time_per_iter_s = 0.0;     // headline: mean per-iteration time of the operation
  double min_s = 0.0;
  double max_s = 0.0;
  /* Within-run spread of the same series `time_per_iter_s` averages, for box
   * plots and for saying how noisy a single measurement was. Set it with
   * set_distribution(), passing the same stats the headline came from - never a
   * separately reduced quantity, or the quartiles would describe a different
   * series than the mean. */
  bool has_distribution = false;
  double median_s = 0.0;
  double p25_s = 0.0;
  double p75_s = 0.0;
  double stddev_s = 0.0;
  bool valid = true;
  std::string extra;                // optional extra key=value pairs (e.g. device="...")
};

/* The reported name carries the backend: benchscribe splits it into
 * <backend>_<benchmark>. A binary that can be driven by more than one backend -
 * oneCCL, which dispatches to NCCL or OSHMPI at runtime - therefore cannot use a
 * fixed name, or the two would be indistinguishable in the results. Runtime
 * scripts set GPU_BENCH_REPORT_BACKEND to say which one is actually in use.
 *
 * Any override must still be a name benchscribe knows, otherwise the record is
 * skipped. */
inline std::string report_name(const bench_report& report) {
  const char* backend = std::getenv("GPU_BENCH_REPORT_BACKEND");
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

/* Records the iteration-to-iteration spread of the reported series. benchscribe
 * treats the emitted fields as optional, so results produced before this existed
 * still parse. `scale` converts the measured quantity into the reported one -
 * pingpong reports half of each measured round trip. */
inline void set_distribution(bench_report& report, const bench_stats& stats,
                             double scale = 1.0) {
  report.median_s = scale * stats.median_s;
  report.p25_s = scale * stats.p25_s;
  report.p75_s = scale * stats.p75_s;
  report.stddev_s = scale * stats.stddev_s;
  report.has_distribution = true;
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
    if (report.has_distribution) {
      std::cout << " median_usec=" << (report.median_s * 1.0e6)
                << " p25_usec=" << (report.p25_s * 1.0e6)
                << " p75_usec=" << (report.p75_s * 1.0e6)
                << " stddev_usec=" << (report.stddev_s * 1.0e6);
    }
  if (!report.extra.empty()) {
    std::cout << ' ' << report.extra;
  }
  std::cout << " validation=" << (report.valid ? "PASS" : "FAIL") << '\n';
}

}  // namespace gpu_bench
