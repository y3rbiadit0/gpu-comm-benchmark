#include <mpi.h>

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli.hpp"
#include "stats/collective_mpi.hpp"
#include "partition.hpp"
#include "report.hpp"
#include "benchmarks/cg_phases.hpp"
#include "benchmarks/cg_step.hpp"
#include "timing.hpp"
#include "validation.hpp"

namespace cg = cooperative_groups;

namespace {

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
  }
}

void check_nvshmem(int status, const char* call) {
  if (status != 0) {
    throw std::runtime_error(std::string(call) + " failed with status " + std::to_string(status));
  }
}

// CG iteration communication skeleton (see src/mpi/cuda/application/cg_step.cu). The halo
// columns and the reduction scalars live in NVSHMEM symmetric memory; the field
// stays in plain device memory.
//
// Every phase is stream-ordered and the step synchronizes once at the end, the
// same shape the NCCL backend has. The earlier version blocked the host inside
// pack and compute and closed the halo with nvshmem_quiet() +
// nvshmem_barrier_all(), which serialized a step NCCL was free to overlap and
// paid a global barrier for a two-neighbour exchange.
//
// Dropping the barrier is safe because the reduction already orders the
// iterations: it is a collective over TEAM_WORLD, so no PE can enqueue the next
// iteration's halo until this iteration's reduce completed, which required
// every PE to have finished its compute -- and compute is the only reader of
// the halo buffers. That is the same reason the NCCL backend needs no barrier
// here.

constexpr int sig_west = 0;  // raised by my left neighbour: my recv_west has landed
constexpr int sig_east = 1;  // raised by my right neighbour: my recv_east has landed

// Blocks to move `elements`, given the cooperative-launch ceiling. Same
// constant and same reason as halo_1d.cu: a small column split across many
// blocks gives each a few bytes and still pays the full grid.sync(). A CG halo
// is one column, so this collapses to a single block at the usual sizes.
constexpr std::size_t min_elements_per_block = 4096;  // 16 KiB of float

// Block size for the halo kernel. Named because the occupancy query that sizes
// the grid must use the same value the launch does.
constexpr int halo_block_size = 256;

std::size_t blocks_for(std::size_t elements, std::size_t max_grid) {
  if (elements == 0U || max_grid == 0U) {
    return 1U;
  }
  const std::size_t wanted = elements / min_elements_per_block;
  const std::size_t least = wanted < 1U ? 1U : wanted;
  return least > max_grid ? max_grid : least;
}

// Device-initiated west/east exchange, following halo_1d.cu: blocks each move a
// chunk with a plain (signal-less) block put, every block that issued work
// completes its own operations, and only after a grid-wide completion point
// does block 0 raise one signal per direction. One signal per direction (not
// one per block) is load-bearing -- without IBGDA each block's remote signal is
// a separate proxied op, and concurrent signal-adds can be dropped on the proxy
// path, leaving the waiter spinning forever.
//
// The chain is open, not a ring: rank 0 has no left and rank pes-1 no right.
// Those ends neither signal nor wait in that direction, and their halo column
// stays at the zero the setup memset left, which is the boundary condition the
// validation expects.
__global__ void halo_exchange_kernel(float* recv_west, float* recv_east, const float* send_west,
                                     const float* send_east, std::uint64_t* signals,
                                     std::size_t side, std::size_t chunk, int left, int right,
                                     std::uint64_t threshold) {
  cg::grid_group grid = cg::this_grid();
  const std::size_t off = static_cast<std::size_t>(blockIdx.x) * chunk;
  const std::size_t len = off < side ? (side - off < chunk ? side - off : chunk) : 0U;

  if (len != 0U) {
    // My west column lands in the left neighbour's east halo, my east column in
    // the right neighbour's west halo.
    if (left >= 0) {
      nvshmemx_float_put_nbi_block(recv_east + off, send_west + off, len, left);
    }
    if (right >= 0) {
      nvshmemx_float_put_nbi_block(recv_west + off, send_east + off, len, right);
    }
  }
  __syncthreads();
  if (len != 0U && threadIdx.x == 0) {
    // Complete this block's cooperative NBI operations before the grid leader
    // publishes the exchange-complete signals.
    nvshmem_quiet();
  }
  grid.sync();  // every active block has completed its puts

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    // I wrote the left neighbour's recv_east, so I raise its sig_east; I wrote
    // the right neighbour's recv_west, so I raise its sig_west. Single writer
    // per counter, so SIGNAL_ADD(+1) is safe and the threshold is monotone.
    if (left >= 0) {
      nvshmemx_signal_op(signals + sig_east, 1U, NVSHMEM_SIGNAL_ADD, left);
    }
    if (right >= 0) {
      nvshmemx_signal_op(signals + sig_west, 1U, NVSHMEM_SIGNAL_ADD, right);
    }
    // Symmetrically, my recv_west comes from the left and my recv_east from the
    // right, so those are the counters I wait on.
    if (left >= 0) {
      nvshmem_signal_wait_until(signals + sig_west, NVSHMEM_CMP_GE, threshold);
    }
    if (right >= 0) {
      nvshmem_signal_wait_until(signals + sig_east, NVSHMEM_CMP_GE, threshold);
    }
  }
  grid.sync();  // compute must not read a halo before the wait cleared
}

__global__ void init_p_kernel(float* p, std::size_t side, std::size_t local_cols, std::size_t width) {
  const auto jj = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto i = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (jj < local_cols && i < side) {
    p[i * width + (jj + 1U)] = 1.0F;
  }
}

__global__ void pack_column_kernel(const float* padded, float* contiguous, std::size_t side, std::size_t width,
                                   std::size_t column) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < side) {
    contiguous[i] = padded[i * width + column];
  }
}

__global__ void unpack_column_kernel(float* padded, const float* contiguous, std::size_t side, std::size_t width,
                                     std::size_t column) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < side) {
    padded[i * width + column] = contiguous[i];
  }
}

__global__ void spmv_kernel(const float* p, float* q, std::size_t side, std::size_t local_cols,
                            std::size_t width) {
  const auto jj = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto i = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (jj < local_cols && i < side) {
    const auto j = jj + 1U;
    const float north = i > 0 ? p[(i - 1U) * width + j] : 0.0F;
    const float south = i + 1U < side ? p[(i + 1U) * width + j] : 0.0F;
    const float west = p[i * width + (j - 1U)];
    const float east = p[i * width + (j + 1U)];
    q[i * width + j] = 0.25F * (north + south + west + east);
  }
}

// Grid-stride dot over the interior with a per-block reduction, so each block
// issues a single atomicAdd instead of one per element (avoids 16M-way
// contention on a single scalar). Launch with a 1D block of 256 threads.
__global__ void cg_dot_kernel(const float* p, const float* q, double* partial_pq, double* partial_qq,
                              std::size_t side, std::size_t local_cols, std::size_t width) {
  __shared__ double shared_pq[256];
  __shared__ double shared_qq[256];
  const std::size_t total = side * local_cols;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  double thread_pq = 0.0;
  double thread_qq = 0.0;
  for (auto idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total; idx += stride) {
    const auto off = (idx / local_cols) * width + (idx % local_cols + 1U);
    const double pv = p[off];
    const double qv = q[off];
    thread_pq += pv * qv;
    thread_qq += qv * qv;
  }
  shared_pq[threadIdx.x] = thread_pq;
  shared_qq[threadIdx.x] = thread_qq;
  __syncthreads();
  for (unsigned s = blockDim.x / 2U; s > 0U; s >>= 1U) {
    if (threadIdx.x < s) {
      shared_pq[threadIdx.x] += shared_pq[threadIdx.x + s];
      shared_qq[threadIdx.x] += shared_qq[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) {
    atomicAdd(partial_pq, shared_pq[0]);
    atomicAdd(partial_qq, shared_qq[0]);
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
    const int device = mpi_rank % device_count;
    check_cuda(cudaSetDevice(device), "cudaSetDevice");

    // grid.sync() and in-kernel NVSHMEM point-to-point synchronization both
    // require a cooperative launch, as in halo_1d.cu.
    int coop_supported = 0;
    check_cuda(cudaDeviceGetAttribute(&coop_supported, cudaDevAttrCooperativeLaunch, device),
               "cudaDeviceGetAttribute(cooperative)");
    if (coop_supported == 0) {
      throw std::runtime_error("device does not support cooperative launch");
    }
    int sm_count = 0;
    check_cuda(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device),
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

    const auto max_side = gpu_bench::parse_size_arg(argc, argv, 1U << 9U);
    const auto iterations = gpu_bench::parse_positive_int_arg(argc, argv, 2, 50);
    const auto warmup = gpu_bench::parse_positive_int_arg(argc, argv, 3, 10);
    const auto sides = gpu_bench::parse_size_list_or_single(argc, argv, 4, max_side);
    const bool phase_pass = gpu_bench::cg_phases_requested();
    const int left = pe == 0 ? -1 : pe - 1;
    const int right = pe + 1 == pes ? -1 : pe + 1;

    // One allocation for the largest side in the sweep; smaller sides use a
    // prefix of it. side * (local_cols + 2) grows with side, so the largest
    // side needs the most room. nvshmem_malloc is collective, so every PE must
    // size from max_side rather than from its own slab.
    const auto max_field_elems = max_side * (gpu_bench::local_count(max_side, pe, pes) + 2U);

    float* p_field = nullptr;
    float* q_field = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&p_field), max_field_elems * sizeof(float)), "cudaMalloc(p)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&q_field), max_field_elems * sizeof(float)), "cudaMalloc(q)");
    auto* send_west = static_cast<float*>(nvshmem_malloc(max_side * sizeof(float)));
    auto* send_east = static_cast<float*>(nvshmem_malloc(max_side * sizeof(float)));
    auto* recv_west = static_cast<float*>(nvshmem_malloc(max_side * sizeof(float)));
    auto* recv_east = static_cast<float*>(nvshmem_malloc(max_side * sizeof(float)));
    // The two dot-product scalars are adjacent, which lets one memset clear
    // both -- but they are reduced by TWO separate one-element calls, not one
    // two-element call.
    //
    // Fusing them is faster (measured: 23.3 us against 31.7 us for NCCL's two
    // at 1n4g) and was tried. It is wrong here. aCG issues
    // acgcomm_allreduce(..., 1, ACG_DOUBLE, ...) at every call site and its
    // logs measure 2.0 reductions per iteration at 8 bytes each. This benchmark
    // exists to predict that, so it has to make the same calls; a fused
    // benchmark would predict a solver nobody runs. It also restores
    // like-for-like comparison with the five backends that issue two.
    auto* partial = static_cast<double*>(nvshmem_malloc(2U * sizeof(double)));
    auto* result = static_cast<double*>(nvshmem_malloc(2U * sizeof(double)));
    // Two counters, never reset: the neighbour pair never changes across the
    // side sweep, so one monotone threshold covers the whole run. A reset would
    // race in-flight proxied signals and drop one (halo_1d.cu).
    auto* signals = static_cast<std::uint64_t*>(nvshmem_malloc(2U * sizeof(std::uint64_t)));
    if (send_west == nullptr || send_east == nullptr || recv_west == nullptr || recv_east == nullptr ||
        partial == nullptr || result == nullptr || signals == nullptr) {
      throw std::runtime_error("failed to allocate NVSHMEM symmetric memory");
    }
    double* const partial_pq = partial;
    double* const partial_qq = partial + 1;
    check_cuda(cudaMemset(signals, 0, 2U * sizeof(std::uint64_t)), "cudaMemset(signals)");

    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    // Largest grid that can run concurrently -- the cooperative-launch ceiling.
    int blocks_per_sm = 0;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, halo_exchange_kernel,
                                                             halo_block_size, 0),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
    std::size_t max_grid = static_cast<std::size_t>(blocks_per_sm > 0 ? blocks_per_sm : 1) *
                           static_cast<std::size_t>(sm_count);

    // Transport-aware block cap, as in halo_1d.cu: without IBGDA every block's
    // remote put is a separate proxied IB operation, so many blocks flood the
    // host proxy inter-node.
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

    const auto sync = [&]() {
      check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(phase)");
    };

    // Counts halo launches, so the signal threshold is base + 1 on the next
    // one. Every PE runs the same warmup, iterations, and optional phase pass,
    // so this stays identical across PEs.
    std::uint64_t halo_launches = 0;

    int all_sides_ok = 1;
    for (const std::size_t side : sides) {
      const auto local_cols = gpu_bench::local_count(side, pe, pes);
      const auto col_offset = gpu_bench::local_offset(side, pe, pes);
      const auto width = local_cols + 2U;
      const auto field_elems = side * width;

      check_cuda(cudaMemset(p_field, 0, field_elems * sizeof(float)), "cudaMemset(p)");
      check_cuda(cudaMemset(q_field, 0, field_elems * sizeof(float)), "cudaMemset(q)");
      check_cuda(cudaMemset(recv_west, 0, side * sizeof(float)), "cudaMemset(recv_west)");
      check_cuda(cudaMemset(recv_east, 0, side * sizeof(float)), "cudaMemset(recv_east)");

      const dim3 block2d(16, 16);
      const dim3 grid2d(static_cast<unsigned>((local_cols + block2d.x - 1U) / block2d.x),
                        static_cast<unsigned>((side + block2d.y - 1U) / block2d.y));
      constexpr int block1d = 256;
      const auto grid1d = static_cast<int>((side + block1d - 1) / block1d);
      const auto dot_grid =
          static_cast<int>(std::min<std::size_t>((side * local_cols + 255U) / 256U, 4096U));

      if (local_cols > 0) {
        init_p_kernel<<<grid2d, block2d>>>(p_field, side, local_cols, width);
        check_cuda(cudaGetLastError(), "init_p_kernel");
      }
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(init)");

      const auto halo_blocks = blocks_for(side, max_grid);
      const auto halo_chunk = std::max<std::size_t>((side + halo_blocks - 1U) / halo_blocks, 1U);

      /* The step, split into the four phases the analysis decomposes it into.
       * Everything here is stream-ordered and asynchronous, so composing them
       * and synchronizing once at the end is exactly the step this benchmark
       * has always timed. The phase pass synchronizes between them instead,
       * which serializes work the step is otherwise free to overlap - which is
       * why it is a separate pass and not instrumentation of the headline
       * loop. */
      const auto pack = [&]() {
        if (local_cols > 0) {
          pack_column_kernel<<<grid1d, block1d, 0, stream>>>(p_field, send_west, side, width, 1U);
          pack_column_kernel<<<grid1d, block1d, 0, stream>>>(p_field, send_east, side, width, local_cols);
        }
      };
      const auto halo = [&]() {
        std::size_t side_v = side;
        std::size_t chunk_v = halo_chunk;
        int left_v = left;
        int right_v = right;
        std::uint64_t threshold = halo_launches + 1U;
        void* args[] = {&recv_west, &recv_east, &send_west, &send_east, &signals,
                        &side_v,    &chunk_v,   &left_v,    &right_v,   &threshold};
        const int status = nvshmemx_collective_launch(
            reinterpret_cast<const void*>(halo_exchange_kernel),
            dim3(static_cast<unsigned>(halo_blocks)), dim3(halo_block_size), args, 0, stream);
        if (status != 0) {
          throw std::runtime_error("nvshmemx_collective_launch failed");
        }
        ++halo_launches;
      };
      const auto compute = [&]() {
        check_cuda(cudaMemsetAsync(partial, 0, 2U * sizeof(double), stream), "cudaMemset(partial)");
        if (local_cols > 0) {
          unpack_column_kernel<<<grid1d, block1d, 0, stream>>>(p_field, recv_west, side, width, 0U);
          unpack_column_kernel<<<grid1d, block1d, 0, stream>>>(p_field, recv_east, side, width, local_cols + 1U);
          spmv_kernel<<<grid2d, block2d, 0, stream>>>(p_field, q_field, side, local_cols, width);
          cg_dot_kernel<<<dot_grid, block1d, 0, stream>>>(p_field, q_field, partial_pq, partial_qq, side,
                                                          local_cols, width);
        }
      };
      const auto reduce = [&]() {
        check_nvshmem(
            nvshmemx_double_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD, result, partial, 1, stream),
            "nvshmemx_double_sum_reduce_on_stream(pq)");
        check_nvshmem(
            nvshmemx_double_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD, result + 1, partial + 1, 1,
                                                 stream),
            "nvshmemx_double_sum_reduce_on_stream(qq)");
      };

      nvshmem_barrier_all();
      MPI_Barrier(MPI_COMM_WORLD);
      const auto stats = gpu_bench::run_benchmark(warmup, iterations, [&]() {
        pack();
        halo();
        compute();
        reduce();
        sync();
      });
      const auto global = gpu_bench::collective_stats(stats);

      gpu_bench::cg_phase_stats phase_global;
      if (phase_pass) {
        nvshmem_barrier_all();
        MPI_Barrier(MPI_COMM_WORLD);
        const auto phase_samples =
            gpu_bench::measure_cg_phases(warmup, iterations, sync, pack, halo, compute, reduce);
        for (int phase = 0; phase < gpu_bench::cg_phase_count; ++phase) {
          phase_global[phase] = gpu_bench::collective_stats(gpu_bench::summarize(phase_samples[phase]));
        }
      }

      const auto ones = [](std::size_t, std::size_t) { return 1.0F; };
      const auto qval = [&](std::size_t i, std::size_t jg) { return gpu_bench::stencil5(i, jg, side, ones); };
      double ref_pq = 0.0;
      double ref_qq = 0.0;
      for (std::size_t i = 0; i < side; ++i) {
        for (std::size_t jg = 0; jg < side; ++jg) {
          const double q = qval(i, jg);
          ref_pq += q;
          ref_qq += q * q;
        }
      }
      double host_pq = 0.0;
      double host_qq = 0.0;
      check_cuda(cudaMemcpy(&host_pq, result, sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(pq)");
      check_cuda(cudaMemcpy(&host_qq, result + 1, sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(qq)");
      int local_ok = gpu_bench::nearly_equal(host_pq, ref_pq) && gpu_bench::nearly_equal(host_qq, ref_qq)
                         ? 1
                         : 0;
      if (local_cols > 0) {
        std::vector<float> host_q(field_elems);
        check_cuda(cudaMemcpy(host_q.data(), q_field, field_elems * sizeof(float), cudaMemcpyDeviceToHost),
                   "cudaMemcpy(q)");
        if (!gpu_bench::validate_columns(host_q.data(), side, local_cols, width, col_offset, qval)) {
          local_ok = 0;
        }
      }
      int global_ok = 1;
      MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      all_sides_ok = all_sides_ok && global_ok;

      if (pe == 0) {
        gpu_bench::bench_report report;
        report.name = "cuda_nvshmem_cg_step";
        report.n = side;
        report.ranks = pes;
        report.bytes_per_iter = 2U * side * sizeof(float);
        report.iterations = iterations;
        report.warmup = warmup;
        report.time_per_iter_s = global.avg_s;
        report.min_s = global.min_s;
        report.max_s = global.max_s;
        gpu_bench::set_distribution(report, global);
        if (phase_pass) {
          report.extra = gpu_bench::cg_phase_fields(phase_global);
        }
        report.valid = global_ok != 0;
        gpu_bench::print_report(report);
      }
    }

    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    nvshmem_free(signals);
    nvshmem_free(result);
    nvshmem_free(partial);
    nvshmem_free(recv_east);
    nvshmem_free(recv_west);
    nvshmem_free(send_east);
    nvshmem_free(send_west);
    check_cuda(cudaFree(q_field), "cudaFree(q)");
    check_cuda(cudaFree(p_field), "cudaFree(p)");
    nvshmem_finalize();
    nvshmem_initialized = false;

    MPI_Finalize();
    return all_sides_ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "rank " << mpi_rank << ": " << error.what() << '\n';
    if (nvshmem_initialized) {
      nvshmem_global_exit(1);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
