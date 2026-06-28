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
#include "report.hpp"
#include "timing.hpp"
#include "validation.hpp"

namespace {

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
  }
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
__global__ void pingpong_kernel(float* send, float* recv, std::uint64_t* flag, std::size_t count,
                                int iterations, int pe, int peer, std::uint64_t base) {
  for (int it = 0; it < iterations; ++it) {
    const std::uint64_t signal = base + static_cast<std::uint64_t>(it) + 1U;
    if (pe == 0) {
      nvshmemx_float_put_block(recv, send, count, peer);
      __syncthreads();
      if (threadIdx.x == 0) {
        nvshmem_fence();
        nvshmemx_signal_op(flag, signal, NVSHMEM_SIGNAL_SET, peer);
        nvshmem_signal_wait_until(flag, NVSHMEM_CMP_GE, signal);
      }
      __syncthreads();
    } else {
      if (threadIdx.x == 0) {
        nvshmem_signal_wait_until(flag, NVSHMEM_CMP_GE, signal);
      }
      __syncthreads();
      nvshmemx_float_put_block(recv, recv, count, peer);
      __syncthreads();
      if (threadIdx.x == 0) {
        nvshmem_fence();
        nvshmemx_signal_op(flag, signal, NVSHMEM_SIGNAL_SET, peer);
      }
      __syncthreads();
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

    const auto max_elems = comm_playground::parse_size_arg(argc, argv, 1U << 22U);
    const auto iterations = comm_playground::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = comm_playground::parse_positive_int_arg(argc, argv, 3, 20);
    const auto message_sizes = comm_playground::parse_size_list_arg(argc, argv, 4, max_elems);

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
    std::uint64_t base = 0;  // monotonically increasing signal base across launches and sizes
    for (std::size_t size : message_sizes) {
      pingpong_kernel<<<1, block_size>>>(device_send, device_recv, flag, size, warmup, pe, peer, base);
      check_cuda(cudaGetLastError(), "pingpong_kernel(warmup)");
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup)");
      base += static_cast<std::uint64_t>(warmup);

      nvshmem_barrier_all();
      comm_playground::wall_timer timer;
      pingpong_kernel<<<1, block_size>>>(device_send, device_recv, flag, size, iterations, pe, peer, base);
      check_cuda(cudaGetLastError(), "pingpong_kernel(timed)");
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
          ok = comm_playground::nearly_equal(host_recv[i], host_send[i]);
        }

        // Amortized per-round-trip time; reported as one-way latency.
        const double round_trip = elapsed / static_cast<double>(iterations);
        comm_playground::bench_report report;
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
        comm_playground::print_report(report);
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
