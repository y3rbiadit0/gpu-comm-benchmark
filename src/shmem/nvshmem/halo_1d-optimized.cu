#include <mpi.h>

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli.hpp"
#include "report.hpp"
#include "timing.hpp"
#include "validation.hpp"

namespace cg = cooperative_groups;

// Comm-only 1D halo exchange benchmark, NVSHMEM -- OPTIMIZED variant.
//
// Fixes two handicaps of the baseline (src/shmem/nvshmem/halo_1d.cu):
//
//  1. Bandwidth. The baseline issues the whole put from ONE block
//     (nvshmemx_float_put_signal_nbi_block <<<1, ...>>>), so copy throughput is
//     capped at a single SM (~33 GB/s bus on A100). Here MANY blocks each move a
//     chunk of the halo, so copy bandwidth scales with the grid.
//
//  2. Per-iteration launch. The baseline relaunches the exchange kernel every
//     iteration, charging kernel-launch latency to alpha. Here the whole
//     warmup+timed loop runs INSIDE one persistent cooperative kernel; grid.sync()
//     carries the ring dependency between iterations, so the launch is paid once.
//
// The payload is split across blocks as plain (signal-less) puts. After a
// grid.sync() confirms every block has issued its puts, block 0 drains them
// PE-wide with nvshmem_quiet() and raises a SINGLE signal per direction, which the
// waiter gates on. Earlier this used a per-block SIGNAL_ADD counted up to
// (iteration * active_blocks); on the IBGDA-less proxy path those many concurrent
// signal-adds could be dropped and deadlock the wait, so it was collapsed to one
// signal per direction per iteration (see the kernel comment for the full story).
//
// Launched with nvshmemx_collective_launch so that both grid.sync() and in-kernel
// NVSHMEM point-to-point sync are legal across PEs.

namespace {

constexpr int sig_left = 0;   // raised by my left neighbour when my left halo is ready
constexpr int sig_right = 1;  // raised by my right neighbour when my right halo is ready

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
  }
}

__global__ void fill_interior_kernel(float* interior, std::size_t n_local, std::size_t half,
                                     float left_marker, float right_marker) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n_local) {
    interior[i] = i < half ? left_marker : right_marker;
  }
}

// Persistent, multi-block ring exchange. Block b owns the contiguous chunk
// [b*chunk, min((b+1)*chunk, halo)) of each boundary and moves it with a plain
// (signal-less) block put. Blocks with an empty chunk still take part in the
// grid.sync() barriers but issue nothing.
//
// Robustness fix vs. the per-block SIGNAL_ADD design: without IBGDA every block's
// remote signal is a separate proxied op, and many concurrent signal-adds per
// iteration can be dropped on the proxy path, leaving signal_wait_until spinning
// forever (the intermittent hang). Here the data goes as N plain puts, then --
// after grid.sync() guarantees every block has *issued* its puts -- ONLY block 0
// completes them PE-wide with nvshmem_quiet() and raises ONE signal per direction.
// One signal per direction per iteration (not one per block) removes the race.
// The single writer per counter means SIGNAL_ADD(+1) is safe; the threshold is
// base + it. Data correctness is still checked by the host-side halo validation.
__global__ void halo_persistent_kernel(float* recv_left, float* recv_right,
                                        const float* send_left, const float* send_right,
                                        std::uint64_t* signals, std::size_t halo,
                                        std::size_t chunk, int left, int right,
                                        int iters, std::uint64_t base) {
  cg::grid_group grid = cg::this_grid();
  const std::size_t off = static_cast<std::size_t>(blockIdx.x) * chunk;
  const std::size_t len = off < halo ? (halo - off < chunk ? halo - off : chunk) : 0U;

  for (int it = 1; it <= iters; ++it) {
    const std::uint64_t threshold = base + static_cast<std::uint64_t>(it);
    if (len != 0U) {
      // My right boundary chunk -> right neighbour's left halo; my left chunk ->
      // left neighbour's right halo. Data only -- the signal is raised once, below.
      nvshmemx_float_put_nbi_block(recv_left + off, send_right + off, len, right);
      nvshmemx_float_put_nbi_block(recv_right + off, send_left + off, len, left);
    }
    grid.sync();  // every block has now issued its data puts for this iteration
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      nvshmem_quiet();  // PE-wide completion: all blocks' puts have landed remotely
      // One signal per direction, after the data is complete. sig_left on the right
      // neighbour = "its left halo is ready"; sig_right on the left neighbour = "its
      // right halo is ready".
      nvshmemx_signal_op(signals + sig_left, 1U, NVSHMEM_SIGNAL_ADD, right);
      nvshmemx_signal_op(signals + sig_right, 1U, NVSHMEM_SIGNAL_ADD, left);
      nvshmem_signal_wait_until(signals + sig_left, NVSHMEM_CMP_GE, threshold);
      nvshmem_signal_wait_until(signals + sig_right, NVSHMEM_CMP_GE, threshold);
    }
    grid.sync();  // reopen the next iteration only after this one's signal + wait
  }
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int mpi_rank = 0;
  int mpi_ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_ranks);
  bool nvshmem_initialized = false;

  try {
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
      throw std::runtime_error("no CUDA devices available");
    }
    check_cuda(cudaSetDevice(mpi_rank % device_count), "cudaSetDevice");

    int coop_supported = 0;
    check_cuda(cudaDeviceGetAttribute(&coop_supported, cudaDevAttrCooperativeLaunch,
                                      mpi_rank % device_count),
               "cudaDeviceGetAttribute(cooperative)");
    if (coop_supported == 0) {
      throw std::runtime_error("device does not support cooperative launch");
    }
    int sm_count = 0;
    check_cuda(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                      mpi_rank % device_count),
               "cudaDeviceGetAttribute(SM count)");

    nvshmemx_init_attr_t attr = {};
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    attr.mpi_comm = &mpi_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    nvshmem_initialized = true;

    const int pe = nvshmem_my_pe();
    const int pes = nvshmem_n_pes();
    if (pe != mpi_rank || pes != mpi_ranks) {
      throw std::runtime_error("NVSHMEM PE layout does not match MPI rank layout");
    }
    if (pes < 2) {
      throw std::runtime_error("ring halo exchange requires at least 2 PEs");
    }

    const auto max_halo = gpu_bench::parse_size_arg(argc, argv, 1U << 20U);
    const auto iterations = gpu_bench::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = gpu_bench::parse_positive_int_arg(argc, argv, 3, 20);
    const auto halo_sizes = gpu_bench::parse_size_list_arg(argc, argv, 4, max_halo);

    const int left = (pe - 1 + pes) % pes;
    const int right = (pe + 1) % pes;

    const std::size_t cap = max_halo;
    const std::size_t n_local = 2U * cap;
    const std::size_t total = n_local + 2U * cap;
    const float left_marker = static_cast<float>(2 * (pe + 1));
    const float right_marker = static_cast<float>(2 * (pe + 1) + 1);
    const float expect_left = static_cast<float>(2 * (left + 1) + 1);
    const float expect_right = static_cast<float>(2 * (right + 1));

    auto* buf = static_cast<float*>(nvshmem_malloc(total * sizeof(float)));
    auto* signals = static_cast<std::uint64_t*>(nvshmem_malloc(2U * sizeof(std::uint64_t)));
    if (buf == nullptr || signals == nullptr) {
      throw std::runtime_error("failed to allocate NVSHMEM symmetric memory");
    }
    check_cuda(cudaMemset(buf, 0, total * sizeof(float)), "cudaMemset(buf)");
    check_cuda(cudaMemset(signals, 0, 2U * sizeof(std::uint64_t)), "cudaMemset(signals)");

    float* interior = buf + cap;
    constexpr int block_size = 256;
    {
      const auto grid_size = static_cast<int>((n_local + block_size - 1U) / block_size);
      fill_interior_kernel<<<grid_size, block_size>>>(interior, n_local, cap, left_marker, right_marker);
      check_cuda(cudaGetLastError(), "fill_interior_kernel");
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(fill)");
    }

    // Largest grid that can run concurrently -- the cooperative-launch ceiling.
    int blocks_per_sm = 0;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &blocks_per_sm, halo_persistent_kernel, block_size, 0),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
    std::size_t max_grid =
        static_cast<std::size_t>(blocks_per_sm > 0 ? blocks_per_sm : 1) * static_cast<std::size_t>(sm_count);

    // Transport-aware block cap. Without IBGDA, every block's remote put_signal is
    // a separate proxied IB operation, so many blocks flood the host proxy and the
    // multi-block optimization backfires inter-node. Cap the grid when the job
    // spans nodes (8 by default); GPU_BENCH_NVSHMEM_MAX_BLOCKS overrides for sweeping the
    // optimum. Intra-node (IPC) the cap is unset, so bandwidth still scales.
    std::size_t block_cap = 0;
    if (const char* cap_env = std::getenv("GPU_BENCH_NVSHMEM_MAX_BLOCKS")) {
      block_cap = std::strtoull(cap_env, nullptr, 10);
    } else if (const char* nodes_env = std::getenv("GPU_BENCH_JOB_NODES")) {
      if (std::strtol(nodes_env, nullptr, 10) > 1) {
        block_cap = 8;
      }
    }
    if (block_cap > 0 && block_cap < max_grid) {
      max_grid = block_cap;
    }

    nvshmem_barrier_all();

    std::vector<float> host_left;
    std::vector<float> host_right;
    std::uint64_t base = 0;  // running signal total; never reset (see launch note)

    for (const std::size_t halo : halo_sizes) {
      float* send_left = interior;
      float* send_right = interior + n_local - halo;
      float* recv_left = interior - halo;
      float* recv_right = interior + n_local;

      // Use up to one block per element, capped by the cooperative ceiling.
      std::size_t nblocks = halo < max_grid ? halo : max_grid;
      if (nblocks == 0U) {
        nblocks = 1U;
      }
      std::size_t chunk = (halo + nblocks - 1U) / nblocks;
      if (chunk == 0U) {
        chunk = 1U;
      }

      std::size_t halo_v = halo;
      std::size_t chunk_v = chunk;
      int left_v = left;
      int right_v = right;
      int launch_iters = warmup;
      std::uint64_t base_v = base;
      void* args[] = {&recv_left, &recv_right, &send_left, &send_right, &signals,
                      &halo_v,    &chunk_v,    &left_v,     &right_v,
                      &launch_iters, &base_v};
      const dim3 grid(static_cast<unsigned>(nblocks));
      const dim3 block(block_size);

      // The signal counters are never reset -- a memset races with in-flight
      // proxied signals and can drop one, deadlocking the wait. Instead each launch
      // waits for a monotonic threshold base + it; base is the running signal total
      // (one per direction per iteration) every PE has sent, identical across PEs.
      auto launch = [&](int iters) {
        launch_iters = iters;
        base_v = base;
        const int status = nvshmemx_collective_launch(
            reinterpret_cast<const void*>(halo_persistent_kernel), grid, block, args, 0, 0);
        if (status != 0) {
          throw std::runtime_error("nvshmemx_collective_launch failed");
        }
      };

      // Warmup (untimed).
      launch(warmup);
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup)");
      base += static_cast<std::uint64_t>(warmup);

      // Timed: one persistent-kernel launch, divided by the iteration count.
      nvshmem_barrier_all();
      gpu_bench::wall_timer timer;
      launch(iterations);
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(timed)");
      const double elapsed = timer.seconds();
      const double avg = iterations > 0 ? elapsed / static_cast<double>(iterations) : 0.0;
      base += static_cast<std::uint64_t>(iterations);

      host_left.assign(halo, 0.0F);
      host_right.assign(halo, 0.0F);
      check_cuda(cudaMemcpy(host_left.data(), recv_left, halo * sizeof(float), cudaMemcpyDeviceToHost),
                 "cudaMemcpy(recv_left)");
      check_cuda(cudaMemcpy(host_right.data(), recv_right, halo * sizeof(float), cudaMemcpyDeviceToHost),
                 "cudaMemcpy(recv_right)");
      int local_ok = 1;
      for (std::size_t i = 0; i < halo; ++i) {
        if (!gpu_bench::nearly_equal(host_left[i], expect_left) ||
            !gpu_bench::nearly_equal(host_right[i], expect_right)) {
          local_ok = 0;
          break;
        }
      }

      /* The timed loop runs inside one persistent kernel, so there are no
       * per-iteration host samples to reduce: this variant can only report
       * MAX over PEs of the amortized average, not the AVG(MAX per iteration)
       * the other benchmarks report (see include/stats/collective.hpp). */
      int global_ok = 1;
      double max_avg = 0.0;
      double min_avg = 0.0;
      MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
      MPI_Reduce(&avg, &max_avg, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&avg, &min_avg, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

      if (pe == 0) {
        gpu_bench::bench_report report;
        report.name = "cuda_nvshmem_halo_1d_optimized";
        report.n = halo;
        report.ranks = pes;
        report.bytes_per_iter = 4U * halo * sizeof(float);
        report.iterations = iterations;
        report.warmup = warmup;
        report.time_per_iter_s = max_avg;  // slowest-PE amortized per-iteration time
        report.min_s = min_avg;            // no per-iteration distribution (loop is in-kernel)
        report.max_s = max_avg;
        report.valid = global_ok != 0;
        report.extra = "halo_elems=" + std::to_string(halo) +
                       " topology=ring bw=sendrecv sync=quiet-signal variant=persistent-multiblock" +
                       " blocks=" + std::to_string(nblocks);
        gpu_bench::print_report(report);
      }
      nvshmem_barrier_all();
    }

    nvshmem_free(signals);
    nvshmem_free(buf);
    nvshmem_finalize();
    nvshmem_initialized = false;

    MPI_Finalize();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "rank " << mpi_rank << ": " << error.what() << '\n';
    if (nvshmem_initialized) {
      nvshmem_global_exit(1);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
