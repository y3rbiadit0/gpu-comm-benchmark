#include <cstddef>

#ifndef USE_CUDA
#define USE_CUDA 1
#endif
#include <shmem.h>

#include <algorithm>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

#include "cli.hpp"
#include "collective_stats_shmem.hpp"
#include "report.hpp"
#include "timing.hpp"
#include "validation.hpp"

namespace {

bool validate_result(const float* values, std::size_t count, float expected) {
  for (std::size_t i = 0; i < count; ++i) {
    if (!gpu_bench::nearly_equal(values[i], expected)) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  shmem_init();

  const int pe = shmem_my_pe();
  const int pes = shmem_n_pes();

  try {
    const auto max_elements = gpu_bench::parse_size_arg(argc, argv, 4194304U);
    const auto iterations = gpu_bench::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = gpu_bench::parse_positive_int_arg(argc, argv, 3, 20);
    const auto message_sizes = gpu_bench::parse_size_list_arg(argc, argv, 4, max_elements);
    const auto max_bytes =
        gpu_bench::checked_size_multiply(max_elements, sizeof(float), "allreduce allocation");
    const auto pwrk_elements = std::max<std::size_t>(max_elements / 2U + 1U,
                                                      SHMEM_REDUCE_MIN_WRKDATA_SIZE);
    const auto pwrk_bytes =
        gpu_bench::checked_size_multiply(pwrk_elements, sizeof(float), "allreduce work buffer");
    const auto psync_bytes = gpu_bench::checked_size_multiply(
        static_cast<std::size_t>(SHMEM_REDUCE_SYNC_SIZE), sizeof(long), "allreduce sync buffer");
    const auto ok_bytes = gpu_bench::checked_size_multiply(
        static_cast<std::size_t>(pes), sizeof(double), "allreduce validation buffer");
    const auto gather_bytes =
        gpu_bench::checked_size_multiply(gpu_bench::collective_gather_elements(pes, iterations),
                                         sizeof(double), "allreduce sample buffer");

    auto* sym_source = static_cast<float*>(shmem_malloc(max_bytes));
    auto* sym_result = static_cast<float*>(shmem_malloc(max_bytes));
    auto* pwrk = static_cast<float*>(shmem_malloc(pwrk_bytes));
    auto* psync = static_cast<long*>(shmem_malloc(psync_bytes));
    auto* ok_by_pe = static_cast<double*>(shmem_malloc(ok_bytes));
    auto* sample_gather = static_cast<double*>(shmem_malloc(gather_bytes));
    if (sym_source == nullptr || sym_result == nullptr || pwrk == nullptr || psync == nullptr ||
        ok_by_pe == nullptr || sample_gather == nullptr) {
      throw std::runtime_error("failed to allocate OSHMPI symmetric memory");
    }

    std::fill_n(sym_source, max_elements, static_cast<float>(pe + 1));
    std::fill_n(sym_result, max_elements, 0.0F);
    for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; ++i) {
      psync[i] = SHMEM_SYNC_VALUE;
    }
    shmem_barrier_all();

    const auto expected = static_cast<float>(static_cast<double>(pes) * (pes + 1) / 2.0);
    int all_sizes_ok = 1;
    for (const auto count : message_sizes) {
      const auto bytes = gpu_bench::checked_size_multiply(count, sizeof(float), "allreduce message");
      shmem_barrier_all();
      const auto stats = gpu_bench::run_benchmark(warmup, iterations, [&]() {
        shmem_float_sum_to_all(sym_result, sym_source, count, 0, 0, pes, pwrk, psync);
      });

      const auto global = gpu_bench::collective_stats(stats, sample_gather, pe, pes);

      const int local_ok = validate_result(sym_result, count, expected) ? 1 : 0;
      const double local_value = static_cast<double>(local_ok);
      shmem_putmem(ok_by_pe + pe, &local_value, sizeof(double), 0);
      shmem_quiet();
      shmem_barrier_all();

      int global_ok = local_ok;
      if (pe == 0) {
        global_ok = 1;
        for (int source_pe = 0; source_pe < pes; ++source_pe) {
          if (ok_by_pe[source_pe] < 0.5) {
            global_ok = 0;
          }
        }
        // Hand the verdict back to everyone so all PEs agree on the exit code.
        // Slot 0 is PE 0's own flag, already consumed and rewritten next size.
        const double global_value = static_cast<double>(global_ok);
        for (int target_pe = 0; target_pe < pes; ++target_pe) {
          shmem_putmem(ok_by_pe, &global_value, sizeof(global_value), target_pe);
        }
        shmem_quiet();
      }
      shmem_barrier_all();
      global_ok = ok_by_pe[0] >= 0.5 ? 1 : 0;
      all_sizes_ok = std::min(all_sizes_ok, global_ok);

      if (pe == 0) {
        const double algorithm_gbytes_per_s =
            global.avg_s > 0.0 ? static_cast<double>(bytes) / global.avg_s / 1.0e9 : 0.0;
        const double bus_gbytes_per_s =
            algorithm_gbytes_per_s * 2.0 * static_cast<double>(pes - 1) / static_cast<double>(pes);

        gpu_bench::bench_report report;
        report.name = "oshmpi_allreduce";
        report.n = count;
        report.ranks = pes;
        report.bytes_per_iter = bytes;
        report.iterations = iterations;
        report.warmup = warmup;
        report.time_per_iter_s = global.avg_s;
        report.min_s = global.min_s;
        report.max_s = global.max_s;
        gpu_bench::set_distribution(report, global);
        report.valid = global_ok != 0;
        report.extra = "datatype=float32 reduction=sum bus_gbytes_per_s=" +
                       std::to_string(bus_gbytes_per_s) + " memory=host_symmetric";
        gpu_bench::print_report(report);
      }
    }

    shmem_free(sample_gather);
    shmem_free(ok_by_pe);
    shmem_free(psync);
    shmem_free(pwrk);
    shmem_free(sym_result);
    shmem_free(sym_source);

    shmem_finalize();
    return all_sizes_ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "PE " << pe << ": " << error.what() << '\n';
    shmem_global_exit(1);
  }
}
