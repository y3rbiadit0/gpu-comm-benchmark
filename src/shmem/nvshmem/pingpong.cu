#include <mpi.h>

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

namespace {

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
  }
}

// Blocks to move `elements`, given the cooperative-launch ceiling.
//
// One block per element wastes the grid on small messages: a 4 KiB payload
// split across 216 blocks gives each block ~19 bytes to move and pays a full
// grid.sync() to coordinate it. Measured cost of getting this wrong on
// pingpong 1n2g: 4 KiB latency 3.4 us -> 17.9 us. Give every block a useful
// chunk instead, so small messages collapse to a single block (the low-latency
// path) and only large ones spread out to chase bandwidth.
constexpr std::size_t min_elements_per_block = 4096;  // 16 KiB of float

std::size_t blocks_for(std::size_t elements, std::size_t max_grid) {
  if (elements == 0U || max_grid == 0U) {
    return 1U;
  }
  const std::size_t wanted = (elements + min_elements_per_block - 1U) / min_elements_per_block;
  return wanted < 1U ? 1U : (wanted > max_grid ? max_grid : wanted);
}

// Device Initiated - PingPong
//  Following Best Practices for NVSHMEM APIs
//  https://docs.nvidia.com/nvshmem/release-notes-install-guide/best-practice-guide/apis.html
//    --> Sending data: https://docs.nvidia.com/nvshmem/api/gen/api/rma.html#nvshmem-put
//    --> Signaling: https://docs.nvidia.com/nvshmem/api/gen/api/signal.html#nvshmem-signal
//    --> Point-to-point synchronization: https://docs.nvidia.com/nvshmem/api/gen/api/sync.html#
//   pe 0: send data to pe 1, signal pe 1, wait for signal from pe 1
//   pe 1: wait for signal from pe 0, send data to pe 0, signal pe 0
//
// The payload is split across a cooperative grid rather than moved by one
// block. `nvshmemx_float_put_block` is issued by a single thread block, so a
// one-block kernel is limited by what one SM can drive - measured at ~15 GB/s
// on NVLink, where two independent MPI stacks reach ~88 GB/s, and *falling*
// past 8 MiB. That is an SM limit, not a link limit. halo_1d.cu solved the same
// problem the same way; this mirrors it, including the transport-aware block
// cap, since without IBGDA every block's remote op is a separate proxied
// operation and too many blocks flood the host proxy inter-node.
//
// grid.sync() carries the ping-pong dependency across blocks: every block
// completes its chunk before block 0 raises the single signal per leg.
__global__ void pingpong_kernel(float* send, float* recv, std::uint64_t* flag, std::size_t count,
                                std::size_t chunk, int iterations, int pe, int peer,
                                std::uint64_t base) {
  cg::grid_group grid = cg::this_grid();
  const std::size_t off = static_cast<std::size_t>(blockIdx.x) * chunk;
  const std::size_t len = off < count ? (count - off < chunk ? count - off : chunk) : 0U;

  for (int it = 0; it < iterations; ++it) {
    const std::uint64_t signal = base + static_cast<std::uint64_t>(it) + 1U;
    if (pe == 0) {
      if (len != 0U) {
        nvshmemx_float_put_nbi_block(recv + off, send + off, len, peer);
      }
      __syncthreads();
      if (len != 0U && threadIdx.x == 0) {
        // Complete this block's cooperative NBI operations before the grid
        // leader publishes the leg-complete signal.
        nvshmem_quiet();
      }
      grid.sync();  // every active block has completed its chunk
      if (blockIdx.x == 0 && threadIdx.x == 0) {
        nvshmem_fence();
        nvshmemx_signal_op(flag, signal, NVSHMEM_SIGNAL_SET, peer);
        nvshmem_signal_wait_until(flag, NVSHMEM_CMP_GE, signal);
      }
      grid.sync();  // reopen the next iteration only after the pong landed
    } else {
      if (blockIdx.x == 0 && threadIdx.x == 0) {
        nvshmem_signal_wait_until(flag, NVSHMEM_CMP_GE, signal);
      }
      grid.sync();  // no block moves the pong before the ping has arrived
      if (len != 0U) {
        nvshmemx_float_put_nbi_block(recv + off, recv + off, len, peer);
      }
      __syncthreads();
      if (len != 0U && threadIdx.x == 0) {
        nvshmem_quiet();
      }
      grid.sync();
      if (blockIdx.x == 0 && threadIdx.x == 0) {
        nvshmem_fence();
        nvshmemx_signal_op(flag, signal, NVSHMEM_SIGNAL_SET, peer);
      }
      grid.sync();
    }
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
    if (mpi_ranks != 2) {
      throw std::runtime_error("pingpong requires exactly 2 ranks");
    }

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
      throw std::runtime_error("no CUDA devices available");
    }
    check_cuda(cudaSetDevice(mpi_rank % device_count), "cudaSetDevice");

    // grid.sync() and in-kernel NVSHMEM point-to-point both require a
    // cooperative launch, as in halo_1d.cu.
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
    if (pes != 2) {
      throw std::runtime_error("pingpong requires exactly 2 PEs");
    }
    const int peer = pe == 0 ? 1 : 0;

    const auto max_elems = gpu_bench::parse_size_arg(argc, argv, 1U << 22U);
    const auto iterations = gpu_bench::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = gpu_bench::parse_positive_int_arg(argc, argv, 3, 20);
    const auto message_sizes = gpu_bench::parse_size_list_arg(argc, argv, 4, max_elems);

    std::vector<float> host_send(max_elems);
    for (std::size_t i = 0; i < max_elems; ++i) {
      host_send[i] = static_cast<float>(i % 1024U);
    }

    auto* device_send = static_cast<float*>(nvshmem_malloc(max_elems * sizeof(float)));
    auto* device_recv = static_cast<float*>(nvshmem_malloc(max_elems * sizeof(float)));
    auto* flag = static_cast<std::uint64_t*>(nvshmem_malloc(sizeof(std::uint64_t)));
    if (device_send == nullptr || device_recv == nullptr || flag == nullptr) {
      throw std::runtime_error("failed to allocate NVSHMEM symmetric memory");
    }
    check_cuda(cudaMemcpy(device_send, host_send.data(), max_elems * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy(send)");
    check_cuda(cudaMemset(flag, 0, sizeof(std::uint64_t)), "cudaMemset(flag)");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(init)");
    nvshmem_barrier_all();

    constexpr int block_size = 256;

    // Largest grid that can run concurrently -- the cooperative-launch ceiling.
    int blocks_per_sm = 0;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &blocks_per_sm, pingpong_kernel, block_size, 0),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
    std::size_t max_grid = static_cast<std::size_t>(blocks_per_sm > 0 ? blocks_per_sm : 1) *
                           static_cast<std::size_t>(sm_count);

    // Transport-aware block cap, as in halo_1d.cu: without IBGDA every block's
    // remote put is a separate proxied IB operation, so many blocks flood the
    // host proxy and multi-block backfires inter-node. Intra-node (IPC) the cap
    // is unset so bandwidth still scales.
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

    std::uint64_t base = 0;  // monotonically increasing signal base across launches and sizes
    for (std::size_t size : message_sizes) {
      const std::size_t nblocks = blocks_for(size, max_grid);
      std::size_t chunk = (size + nblocks - 1U) / nblocks;
      if (chunk == 0U) {
        chunk = 1U;
      }

      std::size_t count_v = size;
      std::size_t chunk_v = chunk;
      int pe_v = pe;
      int peer_v = peer;
      int launch_iters = warmup;
      std::uint64_t base_v = base;
      void* args[] = {&device_send, &device_recv, &flag,  &count_v, &chunk_v,
                      &launch_iters, &pe_v,       &peer_v, &base_v};
      const dim3 grid(static_cast<unsigned>(nblocks));
      const dim3 block(block_size);

      auto launch = [&](int iters, std::uint64_t signal_base, const char* what) {
        launch_iters = iters;
        base_v = signal_base;
        const int status = nvshmemx_collective_launch(
            reinterpret_cast<const void*>(pingpong_kernel), grid, block, args, 0, 0);
        if (status != 0) {
          throw std::runtime_error(std::string("nvshmemx_collective_launch failed: ") + what);
        }
      };

      launch(warmup, base, "warmup");
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup)");
      base += static_cast<std::uint64_t>(warmup);

      nvshmem_barrier_all();
      gpu_bench::wall_timer timer;
      launch(iterations, base, "timed");
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(timed)");
      const double elapsed = timer.seconds();
      base += static_cast<std::uint64_t>(iterations);
      nvshmem_barrier_all();

      if (pe == 0) {
        std::vector<float> host_recv(size);
        check_cuda(cudaMemcpy(host_recv.data(), device_recv, size * sizeof(float), cudaMemcpyDeviceToHost),
                   "cudaMemcpy(recv)");
        bool ok = true;
        for (std::size_t i = 0; i < size && ok; ++i) {
          ok = gpu_bench::nearly_equal(host_recv[i], host_send[i]);
        }

        /* Amortized per-round-trip time; reported as one-way latency. The
         * ping/pong loop runs inside one kernel launch, so there are no
         * per-iteration samples and no distribution to report. */
        const double round_trip = elapsed / static_cast<double>(iterations);
        gpu_bench::bench_report report;
        report.name = "cuda_nvshmem_pingpong";
        report.n = size;
        report.ranks = pes;
        report.bytes_per_iter = size * sizeof(float);
        report.iterations = iterations;
        report.warmup = warmup;
        report.time_per_iter_s = 0.5 * round_trip;
        report.min_s = 0.5 * round_trip;
        report.max_s = 0.5 * round_trip;
        report.valid = ok;
        gpu_bench::print_report(report);
      }
    }

    nvshmem_barrier_all();
    nvshmem_free(flag);
    nvshmem_free(device_recv);
    nvshmem_free(device_send);
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
