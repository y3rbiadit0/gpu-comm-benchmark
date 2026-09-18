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
#include <sstream>
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

/* Device-initiated CG step: the whole iteration inside one kernel.
 *
 * The sibling ../cg_step.cu is the host-initiated composite. Its halo is
 * already device-initiated -- blocks issue their own puts and signals under
 * nvshmemx_collective_launch -- but the rest of the step is a host-side
 * pipeline: four stream-ordered phases per iteration, with the two scalar sums
 * issued as nvshmemx_double_sum_reduce_on_stream. The host still walks the
 * iteration.
 *
 * This backend moves the iteration itself onto the GPU. One cooperative launch
 * runs `steps` complete iterations; pack, halo, unpack, stencil, both dot
 * products and both global sums all happen inside the kernel, separated by
 * grid.sync() instead of by stream order. The reductions are NVSHMEM's
 * *device* collectives, called from the kernel.
 *
 * It exists to predict aCG's `--solver acg-device --comm nvshmem`, whose CG
 * loop is resident on the GPU in exactly this shape (acg/cg-kernels-cuda.cu:
 * acgsolvercuda_cg_kernel, launched once per solve through
 * nvshmemx_collective_launch). The host-initiated cg_step has no counterpart
 * for that solver, so the device solver had no benchmark to be predicted from.
 *
 * The two backends differ in one variable. Same grid decomposition, same
 * stencil, same dot products, same two one-element reductions, same signal
 * protocol, same validation. Only the initiation changes. The difference
 * between them is therefore the cost of host-walking the iteration, which is
 * the quantity that separates aCG's host-NVSHMEM solver from its device one.
 */

// Reduction scope, mirroring aCG's ACG_NVSHMEM_ALLREDUCE_SCOPE cache variable.
// aCG's default is thread scope and its archived runs were built that way, so
// that is the default here too: the prediction has to be made by the same call
// the solver makes. -DGPU_BENCH_NVSHMEM_ALLREDUCE_BLOCK or _WARP select the
// other two, for the same one-at-a-time comparison that knob is there to allow.
#if defined(GPU_BENCH_NVSHMEM_ALLREDUCE_BLOCK)
constexpr const char* reduce_scope_name = "block";
#elif defined(GPU_BENCH_NVSHMEM_ALLREDUCE_WARP)
constexpr const char* reduce_scope_name = "warp";
#else
constexpr const char* reduce_scope_name = "thread";
#endif

constexpr int sig_west = 0;  // raised by my left neighbour: my recv_west has landed
constexpr int sig_east = 1;  // raised by my right neighbour: my recv_east has landed

// Same constant and same reason as ../cg_step.cu and halo_1d.cu: a small column
// split across many blocks gives each a few bytes and still pays the full
// grid.sync().
constexpr std::size_t min_elements_per_block = 4096;  // 16 KiB of float

// Fixed, because cg_dot_device's shared-memory tree is sized for it and the
// occupancy query that sizes the grid must use the value the launch does.
constexpr int step_block_size = 256;

std::size_t blocks_for(std::size_t elements, std::size_t max_grid) {
  if (elements == 0U || max_grid == 0U) {
    return 1U;
  }
  const std::size_t wanted = elements / min_elements_per_block;
  const std::size_t least = wanted < 1U ? 1U : wanted;
  return least > max_grid ? max_grid : least;
}

/* Phases, as __device__ functions over the whole cooperative grid.
 *
 * The host-initiated backend launches each of these with a grid shaped for its
 * own work. Here one grid does all of them, and it is sized by what a
 * cooperative launch can hold resident, not by the problem. So every phase is a
 * grid-stride loop: the same arithmetic, reached from a grid that is usually
 * far smaller than the domain. */

__device__ void pack_columns_device(cg::grid_group grid, const float* p, float* send_west,
                                    float* send_east, std::size_t side, std::size_t local_cols,
                                    std::size_t width) {
  if (local_cols == 0U) {
    return;
  }
  for (std::size_t i = grid.thread_rank(); i < side; i += grid.num_threads()) {
    send_west[i] = p[i * width + 1U];
    send_east[i] = p[i * width + local_cols];
  }
}

__device__ void unpack_columns_device(cg::grid_group grid, float* p, const float* recv_west,
                                      const float* recv_east, std::size_t side,
                                      std::size_t local_cols, std::size_t width) {
  if (local_cols == 0U) {
    return;
  }
  for (std::size_t i = grid.thread_rank(); i < side; i += grid.num_threads()) {
    p[i * width] = recv_west[i];
    p[i * width + local_cols + 1U] = recv_east[i];
  }
}

__device__ void spmv_device(cg::grid_group grid, const float* p, float* q, std::size_t side,
                            std::size_t local_cols, std::size_t width) {
  const std::size_t total = side * local_cols;
  for (std::size_t idx = grid.thread_rank(); idx < total; idx += grid.num_threads()) {
    const std::size_t i = idx / local_cols;
    const std::size_t j = (idx % local_cols) + 1U;
    const float north = i > 0 ? p[(i - 1U) * width + j] : 0.0F;
    const float south = i + 1U < side ? p[(i + 1U) * width + j] : 0.0F;
    const float west = p[i * width + (j - 1U)];
    const float east = p[i * width + (j + 1U)];
    q[i * width + j] = 0.25F * (north + south + west + east);
  }
}

// Per-block tree reduction then one atomicAdd per block per scalar, as in
// ../cg_step.cu: a per-element atomicAdd would put the whole domain in
// contention on two doubles.
__device__ void cg_dot_device(cg::grid_group grid, cg::thread_block block,
                              const float* p, const float* q, double* partial_pq,
                              double* partial_qq, std::size_t side, std::size_t local_cols,
                              std::size_t width) {
  __shared__ double shared_pq[step_block_size];
  __shared__ double shared_qq[step_block_size];

  const std::size_t total = side * local_cols;
  double thread_pq = 0.0;
  double thread_qq = 0.0;
  for (std::size_t idx = grid.thread_rank(); idx < total; idx += grid.num_threads()) {
    const std::size_t off = (idx / local_cols) * width + (idx % local_cols + 1U);
    const double pv = p[off];
    const double qv = q[off];
    thread_pq += pv * qv;
    thread_qq += qv * qv;
  }

  shared_pq[block.thread_rank()] = thread_pq;
  shared_qq[block.thread_rank()] = thread_qq;
  block.sync();
  for (unsigned s = step_block_size / 2U; s > 0U; s >>= 1U) {
    if (block.thread_rank() < s) {
      shared_pq[block.thread_rank()] += shared_pq[block.thread_rank() + s];
      shared_qq[block.thread_rank()] += shared_qq[block.thread_rank() + s];
    }
    block.sync();
  }
  if (block.thread_rank() == 0U) {
    atomicAdd(partial_pq, shared_pq[0]);
    atomicAdd(partial_qq, shared_qq[0]);
  }
}

/* One global sum of one double, issued from inside the kernel.
 *
 * TWO separate one-element calls per iteration, not one two-element call, for
 * the reason spelled out in ../cg_step.cu: aCG calls
 * acgcomm_allreduce(..., 1, ACG_DOUBLE, ...) at each of its two call sites and
 * its logs record 2.0 reductions per iteration at 8 bytes each. A fused
 * benchmark would predict a solver nobody runs.
 *
 * Whichever scope is selected, the call is made by one block (or one thread) of
 * the grid, and every PE makes it the same number of times in the same order,
 * which is what a team collective requires. */
__device__ void reduce_scalar_device(cg::grid_group grid, cg::thread_block block,
                                     double* dest, const double* source) {
  (void)grid;
  (void)block;
#if defined(GPU_BENCH_NVSHMEM_ALLREDUCE_BLOCK)
  if (grid.block_rank() == 0) {
    nvshmemx_double_sum_reduce_block(NVSHMEM_TEAM_WORLD, dest, source, 1);
  }
#elif defined(GPU_BENCH_NVSHMEM_ALLREDUCE_WARP)
  if (grid.block_rank() == 0 && block.thread_rank() < warpSize) {
    nvshmemx_double_sum_reduce_warp(NVSHMEM_TEAM_WORLD, dest, source, 1);
  }
#else
  if (grid.thread_rank() == 0) {
    nvshmem_double_sum_reduce(NVSHMEM_TEAM_WORLD, dest, source, 1);
  }
#endif
}

/* `steps` complete CG iterations, without returning to the host.
 *
 * Halo protocol as in ../cg_step.cu and halo_1d.cu: signal-less block puts,
 * each participating block completes its own operations, and only after a
 * grid-wide completion point does block 0 raise one signal per direction --
 * one, not one per block, because without IBGDA each remote signal is a
 * separate proxied operation and concurrent signal-adds can be dropped there.
 * The counters are never reset, so the threshold keeps climbing across steps
 * and across launches.
 *
 * Why no barrier between iterations. Step s+1's put writes the neighbour's
 * recv buffers, which step s's compute reads, so the two must not overlap. The
 * reductions already prevent it: they are TEAM_WORLD collectives, so no PE
 * leaves step s's reduce until every PE entered it, and every PE enters it
 * having finished the compute that reads recv. This is the same argument the
 * host-initiated backend makes about its stream-ordered reduce, which is why
 * neither needs nvshmem_barrier_all() in the timed loop.
 *
 * Only the first `halo_blocks` blocks issue puts, while all of them compute.
 * In the host-initiated backend those are separate launches and so separately
 * sized; here one grid does both, and the transport's preference for few
 * blocks (GPU_BENCH_NVSHMEM_MAX_BLOCKS, which the inter-node proxy needs) must
 * not become a cap on the stencil's parallelism. */
__global__ void cg_step_device_kernel(float* p_field, float* q_field, float* send_west,
                                      float* send_east, float* recv_west, float* recv_east,
                                      std::uint64_t* signals, double* partial, double* result,
                                      std::size_t side, std::size_t local_cols, std::size_t width,
                                      std::size_t halo_chunk, unsigned halo_blocks, int left,
                                      int right, std::uint64_t base_threshold, int steps) {
  cg::thread_block block = cg::this_thread_block();
  cg::grid_group grid = cg::this_grid();

  for (int step = 0; step < steps; ++step) {
    const std::uint64_t threshold = base_threshold + static_cast<std::uint64_t>(step) + 1U;

    // ---- pack, and clear the accumulators the dot products add into ----
    pack_columns_device(grid, p_field, send_west, send_east, side, local_cols, width);
    if (grid.thread_rank() == 0) {
      partial[0] = 0.0;
      partial[1] = 0.0;
    }
    grid.sync();

    // ---- halo ----
    const bool sending = blockIdx.x < halo_blocks;
    if (sending) {
      const std::size_t off = static_cast<std::size_t>(blockIdx.x) * halo_chunk;
      const std::size_t len = off < side ? (side - off < halo_chunk ? side - off : halo_chunk) : 0U;
      if (len != 0U) {
        // My west column lands in the left neighbour's east halo, my east
        // column in the right neighbour's west halo.
        if (left >= 0) {
          nvshmemx_float_put_nbi_block(recv_east + off, send_west + off, len, left);
        }
        if (right >= 0) {
          nvshmemx_float_put_nbi_block(recv_west + off, send_east + off, len, right);
        }
        block.sync();
        if (block.thread_rank() == 0) {
          // Complete this block's cooperative NBI operations before the grid
          // leader publishes the exchange-complete signals.
          nvshmem_quiet();
        }
      }
    }
    grid.sync();  // every sending block has completed its puts

    if (blockIdx.x == 0 && threadIdx.x == 0) {
      // Single writer per counter, so SIGNAL_ADD(+1) is safe and the threshold
      // is monotone. The chain is open: rank 0 has no left and rank pes-1 no
      // right, and those ends neither signal nor wait in that direction, so
      // their halo column keeps the zero the setup memset left -- which is the
      // boundary condition the validation expects.
      if (left >= 0) {
        nvshmemx_signal_op(signals + sig_east, 1U, NVSHMEM_SIGNAL_ADD, left);
      }
      if (right >= 0) {
        nvshmemx_signal_op(signals + sig_west, 1U, NVSHMEM_SIGNAL_ADD, right);
      }
      if (left >= 0) {
        nvshmem_signal_wait_until(signals + sig_west, NVSHMEM_CMP_GE, threshold);
      }
      if (right >= 0) {
        nvshmem_signal_wait_until(signals + sig_east, NVSHMEM_CMP_GE, threshold);
      }
    }
    grid.sync();  // compute must not read a halo before the wait cleared

    // ---- compute ----
    unpack_columns_device(grid, p_field, recv_west, recv_east, side, local_cols, width);
    grid.sync();  // the stencil reads the ghost columns just written
    spmv_device(grid, p_field, q_field, side, local_cols, width);
    grid.sync();  // the dot products read q across block boundaries
    cg_dot_device(grid, block, p_field, q_field, partial, partial + 1, side, local_cols, width);
    grid.sync();  // every block's atomicAdd has landed

    // ---- reduce ----
    // A grid sync after each one, not just after the pair, which is how aCG's
    // device kernel brackets every reduction it issues. Under block and warp
    // scope the call is made by a subset of the grid, so the sync is what puts
    // the rest of the grid behind the collective; leaving it out would also
    // put two TEAM_WORLD collectives back to back with nothing between them.
    reduce_scalar_device(grid, block, result, partial);
    grid.sync();
    reduce_scalar_device(grid, block, result + 1, partial + 1);
    grid.sync();  // no block may start step s+1's put before the reduce returned
  }
}

__global__ void init_p_kernel(float* p, std::size_t side, std::size_t local_cols, std::size_t width) {
  const auto jj = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto i = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (jj < local_cols && i < side) {
    p[i * width + (jj + 1U)] = 1.0F;
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

    /* Iterations per launch, the knob this backend exists to turn.
     *
     * 1 keeps one launch per iteration: communication is device-initiated, but
     * the host still walks the loop, so the comparison against the
     * host-initiated backend isolates the reduction path alone.
     *
     * >1 makes the loop resident, which is aCG's device solver. The reported
     * usec is then per iteration, the launch amortized over the steps, and the
     * difference from the 1 case is what leaving the host out of the loop is
     * worth. Every PE must use the same value: it decides how many collectives
     * the kernel issues. */
    const int steps_per_launch =
        gpu_bench::parse_positive_int_env("GPU_BENCH_CG_STEPS_PER_LAUNCH", 1);
    {
      int agreed_steps = steps_per_launch;
      MPI_Allreduce(MPI_IN_PLACE, &agreed_steps, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      if (agreed_steps != steps_per_launch) {
        throw std::runtime_error("GPU_BENCH_CG_STEPS_PER_LAUNCH differs between ranks");
      }
    }

    /* No phase pass.
     *
     * The other cg_step backends offer one by synchronizing between phases.
     * Here the phases are grid.sync()-separated regions of a single kernel:
     * timing them apart would mean cutting the kernel into four cooperative
     * launches, which is the host-initiated backend -- a different measurement
     * wearing this one's name. The honest report is one number per iteration.
     *
     * aCG's device solver has the same shape and the same consequence: its
     * gemv/dot/allreduce/halo timers are all zero and the whole solve lands in
     * `other`, which is why the cross-level comparison for that solver cannot
     * be made per operation. This backend reproduces that limitation rather
     * than papering over it. */
    if (gpu_bench::cg_phases_requested() && pe == 0) {
      std::cerr << "note: cuda_nvshmem_device_cg_step has no phase pass; "
                   "the iteration is one kernel (see the comment in this file)\n";
    }

    const int left = pe == 0 ? -1 : pe - 1;
    const int right = pe + 1 == pes ? -1 : pe + 1;

    const auto max_field_elems = max_side * (gpu_bench::local_count(max_side, pe, pes) + 2U);

    float* p_field = nullptr;
    float* q_field = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&p_field), max_field_elems * sizeof(float)), "cudaMalloc(p)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&q_field), max_field_elems * sizeof(float)), "cudaMalloc(q)");
    auto* send_west = static_cast<float*>(nvshmem_malloc(max_side * sizeof(float)));
    auto* send_east = static_cast<float*>(nvshmem_malloc(max_side * sizeof(float)));
    auto* recv_west = static_cast<float*>(nvshmem_malloc(max_side * sizeof(float)));
    auto* recv_east = static_cast<float*>(nvshmem_malloc(max_side * sizeof(float)));
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
    check_cuda(cudaMemset(signals, 0, 2U * sizeof(std::uint64_t)), "cudaMemset(signals)");

    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    // The cooperative-launch ceiling for the fused kernel. Its static shared
    // memory (the dot product's two tree buffers) lowers this below what the
    // host-initiated backend's individual kernels reach, which is one of the
    // costs of fusing them.
    int blocks_per_sm = 0;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, cg_step_device_kernel,
                                                             step_block_size, 0),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
    if (blocks_per_sm <= 0) {
      throw std::runtime_error("cg_step_device_kernel does not fit for a cooperative launch");
    }
    const auto grid_blocks =
        static_cast<std::size_t>(blocks_per_sm) * static_cast<std::size_t>(sm_count);

    // Transport-aware cap, as in ../cg_step.cu -- but on the sending blocks
    // only. Without IBGDA every block's remote put is a separate proxied IB
    // operation, so many blocks flood the host proxy inter-node; the stencil
    // has no such limit and keeps the whole grid.
    std::size_t halo_block_cap = 0;
    if (const char* cap_env = std::getenv("GPU_BENCH_NVSHMEM_MAX_BLOCKS")) {
      halo_block_cap = std::strtoull(cap_env, nullptr, 10);
    } else if (const char* nodes_env = std::getenv("GPU_BENCH_JOB_NODES")) {
      if (std::strtol(nodes_env, nullptr, 10) > 1) {
        halo_block_cap = 8;
      }
    }

    // Counts iterations launched, so the signal threshold continues from where
    // the last launch left it. Every PE runs the same warmup and iterations, so
    // this stays identical across PEs.
    std::uint64_t steps_launched = 0;

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

      if (local_cols > 0) {
        const dim3 block2d(16, 16);
        const dim3 grid2d(static_cast<unsigned>((local_cols + block2d.x - 1U) / block2d.x),
                          static_cast<unsigned>((side + block2d.y - 1U) / block2d.y));
        init_p_kernel<<<grid2d, block2d>>>(p_field, side, local_cols, width);
        check_cuda(cudaGetLastError(), "init_p_kernel");
      }
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(init)");

      auto halo_blocks = blocks_for(side, grid_blocks);
      if (halo_block_cap > 0 && halo_block_cap < halo_blocks) {
        halo_blocks = halo_block_cap;
      }
      const auto halo_chunk = std::max<std::size_t>((side + halo_blocks - 1U) / halo_blocks, 1U);

      const auto launch = [&]() {
        float* p_arg = p_field;
        float* q_arg = q_field;
        float* sw = send_west;
        float* se = send_east;
        float* rw = recv_west;
        float* re = recv_east;
        std::uint64_t* sig = signals;
        double* par = partial;
        double* res = result;
        std::size_t side_v = side;
        std::size_t cols_v = local_cols;
        std::size_t width_v = width;
        std::size_t chunk_v = halo_chunk;
        auto halo_blocks_v = static_cast<unsigned>(halo_blocks);
        int left_v = left;
        int right_v = right;
        std::uint64_t base = steps_launched;
        int steps_v = steps_per_launch;
        void* args[] = {&p_arg,    &q_arg,   &sw,      &se,           &rw,      &re,
                        &sig,      &par,     &res,     &side_v,       &cols_v,  &width_v,
                        &chunk_v,  &halo_blocks_v,     &left_v,       &right_v, &base,
                        &steps_v};
        const int status = nvshmemx_collective_launch(
            reinterpret_cast<const void*>(cg_step_device_kernel),
            dim3(static_cast<unsigned>(grid_blocks)), dim3(step_block_size), args, 0, stream);
        if (status != 0) {
          throw std::runtime_error("nvshmemx_collective_launch failed");
        }
        steps_launched += static_cast<std::uint64_t>(steps_per_launch);
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(step)");
      };

      nvshmem_barrier_all();
      MPI_Barrier(MPI_COMM_WORLD);
      const auto stats = gpu_bench::run_benchmark(warmup, iterations, launch);
      const auto global = gpu_bench::collective_stats(stats);

      // Each sample is one launch of `steps_per_launch` iterations, so the
      // reported figure is per iteration and directly comparable with the
      // host-initiated backend's.
      const double per_step = 1.0 / static_cast<double>(steps_per_launch);

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
        report.name = "cuda_nvshmem_device_cg_step";
        report.n = side;
        report.ranks = pes;
        report.bytes_per_iter = 2U * side * sizeof(float);
        report.iterations = iterations;
        report.warmup = warmup;
        report.time_per_iter_s = global.avg_s * per_step;
        report.min_s = global.min_s * per_step;
        report.max_s = global.max_s * per_step;
        gpu_bench::set_distribution(report, global, per_step);
        std::ostringstream extra;
        extra << "initiation=device steps_per_launch=" << steps_per_launch
              << " reduce_scope=" << reduce_scope_name << " grid_blocks=" << grid_blocks
              << " halo_blocks=" << halo_blocks;
        report.extra = extra.str();
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
