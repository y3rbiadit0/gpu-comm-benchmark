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
#include "report.hpp"
#include "timing.hpp"
#include "validation.hpp"

namespace {

bool validate_result(const float* values, std::size_t count, float expected) {
  for (std::size_t i = 0; i < count; ++i) {
    if (!comm_playground::nearly_equal(values[i], expected)) {
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
    const auto max_elements = comm_playground::parse_size_arg(argc, argv, 4194304U);
    const auto iterations = comm_playground::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = comm_playground::parse_positive_int_arg(argc, argv, 3, 20);
    const auto message_sizes = comm_playground::parse_size_list_arg(argc, argv, 4, max_elements);
    const auto max_bytes =
        comm_playground::checked_size_multiply(max_elements, sizeof(float), "allreduce allocation");
    const auto pwrk_elements = std::max<std::size_t>(max_elements / 2U + 1U,
                                                      SHMEM_REDUCE_MIN_WRKDATA_SIZE);
    const auto pwrk_bytes =
        comm_playground::checked_size_multiply(pwrk_elements, sizeof(float), "allreduce work buffer");
    const auto psync_bytes = comm_playground::checked_size_multiply(
        static_cast<std::size_t>(SHMEM_REDUCE_SYNC_SIZE), sizeof(long), "allreduce sync buffer");
    const auto stats_elements = comm_playground::checked_size_multiply(
        4U, static_cast<std::size_t>(pes), "allreduce statistics buffer");
    const auto stats_bytes =
        comm_playground::checked_size_multiply(stats_elements, sizeof(double), "allreduce statistics buffer");

    auto* sym_source = static_cast<float*>(shmem_malloc(max_bytes));
    auto* sym_result = static_cast<float*>(shmem_malloc(max_bytes));
    auto* pwrk = static_cast<float*>(shmem_malloc(pwrk_bytes));
    auto* psync = static_cast<long*>(shmem_malloc(psync_bytes));
    auto* stats_by_pe = static_cast<double*>(shmem_malloc(stats_bytes));
    if (sym_source == nullptr || sym_result == nullptr || pwrk == nullptr || psync == nullptr ||
        stats_by_pe == nullptr) {
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
      const auto bytes = comm_playground::checked_size_multiply(count, sizeof(float), "allreduce message");
      shmem_barrier_all();
      const auto stats = comm_playground::run_benchmark(warmup, iterations, [&]() {
        shmem_float_sum_to_all(sym_result, sym_source, count, 0, 0, pes, pwrk, psync);
      });

      const int local_ok = validate_result(sym_result, count, expected) ? 1 : 0;
      const double local_values[4] = {
          stats.avg_s, stats.min_s, stats.max_s, static_cast<double>(local_ok)};
      shmem_putmem(stats_by_pe + 4 * pe, local_values, 4U * sizeof(double), 0);
      shmem_quiet();
      shmem_barrier_all();

      double time_per_iter = stats.avg_s;
      double min_time = stats.min_s;
      double max_time = stats.max_s;
      int global_ok = local_ok;
      if (pe == 0) {
        global_ok = 1;
        for (int source_pe = 0; source_pe < pes; ++source_pe) {
          time_per_iter = std::max(time_per_iter, stats_by_pe[4 * source_pe + 0]);
          min_time = std::min(min_time, stats_by_pe[4 * source_pe + 1]);
          max_time = std::max(max_time, stats_by_pe[4 * source_pe + 2]);
          if (stats_by_pe[4 * source_pe + 3] < 0.5) {
            global_ok = 0;
          }
        }
        const double global_value = static_cast<double>(global_ok);
        for (int target_pe = 0; target_pe < pes; ++target_pe) {
          shmem_putmem(stats_by_pe, &global_value, sizeof(global_value), target_pe);
        }
        shmem_quiet();
      }
      shmem_barrier_all();
      global_ok = stats_by_pe[0] >= 0.5 ? 1 : 0;
      all_sizes_ok = std::min(all_sizes_ok, global_ok);

      if (pe == 0) {
        const double algorithm_gbytes_per_s =
            time_per_iter > 0.0 ? static_cast<double>(bytes) / time_per_iter / 1.0e9 : 0.0;
        const double bus_gbytes_per_s =
            algorithm_gbytes_per_s * 2.0 * static_cast<double>(pes - 1) / static_cast<double>(pes);

        comm_playground::bench_report report;
        report.name = "oshmpi_allreduce";
        report.n = count;
        report.ranks = pes;
        report.bytes_per_iter = bytes;
        report.iterations = iterations;
        report.warmup = warmup;
        report.time_per_iter_s = time_per_iter;
        report.min_s = min_time;
        report.max_s = max_time;
        report.valid = global_ok != 0;
        report.extra = "datatype=float32 reduction=sum bus_gbytes_per_s=" +
                       std::to_string(bus_gbytes_per_s) + " memory=host_symmetric";
        comm_playground::print_report(report);
      }
    }

    shmem_free(stats_by_pe);
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
