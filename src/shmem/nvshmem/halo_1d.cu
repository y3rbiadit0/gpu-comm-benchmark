#include <mpi.h>

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli.hpp"
#include "collective_stats_mpi.hpp"
#include "report.hpp"
#include "timing.hpp"
#include "validation.hpp"

// Comm-only 1D halo exchange benchmark, NVSHMEM (device-initiated).
//
// Periodic ring with a swept halo width H. Unlike the MPI/NCCL backends, the
// exchange is issued from *inside a kernel*: each PE writes its boundary
// directly into the neighbour's halo with a block-scoped put, then sets a remote
// signal. Synchronisation is point-to-point (signal wait), not a global
// barrier_all, so the timed loop reflects neighbour latency rather than barrier
// scalability.
//
// Symmetric buffer, identical layout on every PE:
//
//   [ left_halo(cap) | interior(2*cap) | right_halo(cap) ]   cap = max halo width
//
// Because the layout is symmetric, the destination address on a neighbour is the
// same pointer we use locally (recv_left / recv_right); only the target PE
// differs. Interior markers (exact in float) let each PE validate locally. MPI
// is used for init and cross-PE reductions. Reported GB/s is send+receive
// ("bus") bandwidth: 4 * H * sizeof(float).

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

// One block performs the full exchange. recv_left/recv_right are symmetric
// addresses, so they name the right slot on the neighbour too.
__global__ void halo_exchange_kernel(float* recv_left, float* recv_right, const float* send_left,
                                     const float* send_right, std::uint64_t* signals, std::size_t halo,
                                     std::uint64_t value, int left, int right) {
  // The block API keeps the transfer cooperative across all halo sizes.
  nvshmemx_float_put_signal_nbi_block(recv_left, send_right, halo, signals + sig_left, value,
                                      NVSHMEM_SIGNAL_SET, right);
  nvshmemx_float_put_signal_nbi_block(recv_right, send_left, halo, signals + sig_right, value,
                                      NVSHMEM_SIGNAL_SET, left);
  __syncthreads();
  if (threadIdx.x == 0) {
    // Incoming signals acknowledge neighbours' puts, not this PE's outbound
    // NBI operations. Complete those before allowing the timed kernel to end.
    nvshmem_quiet();
    nvshmem_signal_wait_until(signals + sig_left, NVSHMEM_CMP_GE, value);
    nvshmem_signal_wait_until(signals + sig_right, NVSHMEM_CMP_GE, value);
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
    {
      constexpr int block_size = 256;
      const auto grid_size = static_cast<int>((n_local + block_size - 1U) / block_size);
      fill_interior_kernel<<<grid_size, block_size>>>(interior, n_local, cap, left_marker, right_marker);
      check_cuda(cudaGetLastError(), "fill_interior_kernel");
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(fill)");
    }
    nvshmem_barrier_all();

    std::vector<float> host_left;
    std::vector<float> host_right;
    std::uint64_t value = 0;

    for (const std::size_t halo : halo_sizes) {
      float* send_left = interior;
      float* send_right = interior + n_local - halo;
      float* recv_left = interior - halo;
      float* recv_right = interior + n_local;

      constexpr int block_size = 256;
      std::size_t halo_v = halo;
      int left_v = left;
      int right_v = right;
      MPI_Barrier(MPI_COMM_WORLD);
      const auto stats = gpu_bench::run_benchmark(warmup, iterations, [&]() {
        ++value;
        void* args[] = {&recv_left, &recv_right, &send_left, &send_right, &signals,
                        &halo_v, &value, &left_v, &right_v};
        const int status = nvshmemx_collective_launch(
            reinterpret_cast<const void*>(halo_exchange_kernel), dim3(1), dim3(block_size), args, 0, 0);
        if (status != 0) {
          throw std::runtime_error("nvshmemx_collective_launch failed");
        }
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(halo)");
      });

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

      int global_ok = 1;
      MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
      const auto global = gpu_bench::collective_stats(stats);

      if (pe == 0) {
        gpu_bench::bench_report report;
        report.name = "cuda_nvshmem_halo_1d";
        report.n = halo;
        report.ranks = pes;
        report.bytes_per_iter = 4U * halo * sizeof(float);
        report.iterations = iterations;
        report.warmup = warmup;
        report.time_per_iter_s = global.avg_s;
        report.min_s = global.min_s;
        report.max_s = global.max_s;
        gpu_bench::set_distribution(report, global);
        report.valid = global_ok != 0;
        report.extra = "halo_elems=" + std::to_string(halo) + " topology=ring bw=sendrecv sync=signal";
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
