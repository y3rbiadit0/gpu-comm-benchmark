#include <mpi.h>

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

#include "stats/collective_mpi.hpp"
#include "moe.hpp"
#include "report.hpp"
#include "timing.hpp"

// Device-initiated MoE dispatch/combine.
//
// The host-driven version issued one blocking nvshmem_float_put per peer and
// closed each phase with nvshmem_quiet() + nvshmem_barrier_all(). That cost
// 2*pes host round trips and two global barriers per iteration, against a NCCL
// backend that submits every send/recv in one ncclGroupStart/End and syncs the
// stream once. Two of those three costs are not inherent to the pattern:
//
//  1. The puts belong on the device. Each phase is a put kernel followed by a
//     one-block signal/wait kernel, so the host issues four launches per
//     iteration instead of 2*pes blocking calls.
//
//  2. barrier_all is stronger than the pattern needs. A receiver only has to
//     know that its own incoming puts landed, which is what a per-source signal
//     says. The global barrier additionally makes every PE wait on peers it
//     never exchanged with, which is pure cost under skewed routing, where most
//     of the count matrix is zero.
//
// What remains is four launches and one device sync per iteration. The kernels
// deliberately do *not* run the whole measured batch: a persistent kernel would
// amortize the launch away on this backend only, and the reported number would
// stop being comparable. See halo_1d.cu for the batched/isolated split where
// that distinction is made explicit on both sides.
//
// The inter-node grid is capped, like halo_1d.cu, but at a different value and
// for a slightly different reason. There each block sends to a *different*
// neighbour; here blocks split *one peer's* payload, so the cap is not about
// how many peers are in flight but how many concurrent proxied block-puts the
// host proxy can absorb.
//
// Measured at 2n1g, usec per iteration, uniform / locality80 / hotspot80:
//
//    16 blocks   1932 / 1404 / 2670     stddev ~2 us
//    32 blocks   1729 / 1027 / 2493     stddev ~2 us
//    64 blocks   1656 /  877 / 2439     stddev ~2 us   <- optimum
//   992 blocks   3112 / 2371 / 4219     stddev 97-364 us
//
// The curve is flat around 64 and collapses well before the full grid. The
// variance is the tell: an order of magnitude more jitter at 992 is queue
// contention, not steady-state slowness. Roughly half the payload goes remote
// at 2n1g, so the full grid puts ~512 proxied block-puts in flight at once.
//
// 64 was measured at 2n1g with this payload only. The optimum plausibly depends
// on peers x payload, so re-sweep with GPU_BENCH_NVSHMEM_MAX_BLOCKS before
// trusting it at 4n4g or 8n4g. Intra-node is uncapped: IPC has no proxy, and
// the self-copy wants all the parallelism it can get.

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

// Elements one block should move before a second block is worth involving.
// Same constant as halo_1d.cu and pingpong.cu: splitting a small payload across
// many blocks gives each one only a few bytes to move.
constexpr std::size_t min_elements_per_block = 4096;  // 16 KiB of float

// One block-sized piece of one peer's transfer. Built on the host, where the
// routing plan already lives, and read straight from the kernel: the plan is
// fixed for a routing case, so this is uploaded once and reused every
// iteration.
struct put_work {
  const float* src;  // local source
  float* dst;        // destination, as the local address of the symmetric object
  std::size_t len;
  int peer;
};

// One direction of the exchange. `signals` points at this phase's own block of
// `pes` counters, so dispatch and combine never share a slot and each slot has
// exactly one writer.
struct moe_phase {
  const put_work* work;
  int work_count;
  const int* signal_peers;  // peers this PE sends to, so must signal
  int signal_count;
  const int* wait_active;   // size pes; nonzero = this PE sends to me, so wait on its slot
  std::uint64_t* signals;
};

// Move one phase's payload. No cross-block synchronisation, so this is an
// ordinary launch: blocks are independent, and the only ordering that matters
// -- every block's quiet before any signal is raised -- is supplied by the
// kernel boundary.
//
// This replaced a single cooperative kernel that carried both phases and used
// four cg::grid_group::sync() calls. Measured at 1n1g, where moe does no
// communication at all, the old kernel spent roughly 49 us of its 100 us
// outside the copy itself, against a ~51 us floor for 64 MiB of read+write
// traffic on A100. Ma et al. (arXiv:2606.05951) §VI.D make the same point about
// NVSHMEM's own API: `_grid`-scoped collectives are deliberately absent because
// synchronising across CTAs inside a running kernel "is more expensive than
// ending the kernel and launching a stream-ordered communication kernel".
//
// An ordinary launch is legal here because this kernel never waits on another
// PE: NVSHMEM requires a collective launch only for kernels that use blocking
// synchronisation (wait_until, barrier, device collectives). Device RMA plus a
// local quiet is not one. If this ever hangs, promoting it to
// nvshmemx_collective_launch is the first thing to try.
__global__ void moe_put_kernel(const put_work* work, int work_count, int my_pe) {
  // The loop bound depends only on blockIdx/gridDim, so every thread of a block
  // runs the same iterations -- required, because the block put is a block-wide
  // collective. Blocks past the work count fall straight through.
  for (int w = static_cast<int>(blockIdx.x); w < work_count;
       w += static_cast<int>(gridDim.x)) {
    const put_work item = work[w];
    if (item.peer == my_pe) {
      for (std::size_t i = threadIdx.x; i < item.len; i += blockDim.x) {
        item.dst[i] = item.src[i];
      }
    } else {
      nvshmemx_float_put_nbi_block(item.dst, item.src, item.len, item.peer);
    }
  }
  __syncthreads();
  if (static_cast<int>(blockIdx.x) < work_count && threadIdx.x == 0) {
    // Complete this block's cooperative NBI operations. The kernel does not end
    // until every block has, which is what the next kernel's signal relies on.
    nvshmem_quiet();
  }
}

// Raise one signal per peer, then wait for the incoming ones. A single block:
// one signal per peer per phase is load-bearing, not a simplification --
// without IBGDA each block's remote signal is a separate proxied op, and
// concurrent signal-adds can be dropped on the proxy path, leaving the waiter
// spinning forever (halo_1d.cu).
//
// This one needs nvshmemx_collective_launch: nvshmem_signal_wait_until blocks
// on another PE's progress, so every PE's kernel has to be resident.
__global__ void moe_signal_kernel(const int* signal_peers, int signal_count,
                                  const int* wait_active, int pes,
                                  std::uint64_t* signals, int my_pe,
                                  std::uint64_t threshold) {
  // The counter lives on the receiver and is indexed by the sender, so this is
  // one address sent to many peers. Single writer per slot, so SIGNAL_ADD(+1)
  // is safe and the threshold is monotone.
  for (int i = static_cast<int>(threadIdx.x); i < signal_count;
       i += static_cast<int>(blockDim.x)) {
    nvshmemx_signal_op(signals + my_pe, 1U, NVSHMEM_SIGNAL_ADD, signal_peers[i]);
  }
  // Raise every outgoing signal before any thread parks on an incoming one.
  __syncthreads();
  for (int s = static_cast<int>(threadIdx.x); s < pes; s += static_cast<int>(blockDim.x)) {
    if (wait_active[s] != 0) {
      nvshmem_signal_wait_until(signals + s, NVSHMEM_CMP_GE, threshold);
    }
  }
}

// Splits one peer's transfer into block-sized pieces and appends them.
void append_work(std::vector<put_work>& work, const float* src, float* dst, std::size_t count,
                 int peer) {
  if (count == 0) {
    return;
  }
  // Floor, not ceil: a peer with 5000 elements gets one block with all of it
  // rather than two with 2500 each. Only a full min_elements_per_block of
  // surplus earns another block and another stride through the work list.
  std::size_t pieces = count / min_elements_per_block;
  if (pieces == 0U) {
    pieces = 1U;
  }
  const std::size_t chunk = (count + pieces - 1U) / pieces;
  for (std::size_t off = 0; off < count; off += chunk) {
    const std::size_t len = std::min(chunk, count - off);
    work.push_back(put_work{src + off, dst + off, len, peer});
  }
}

// Uploads a host vector to plain (non-symmetric) device memory. These are
// read-only per-routing-case inputs, not communication buffers.
template <typename T>
T* upload(const std::vector<T>& host, const char* what) {
  if (host.empty()) {
    return nullptr;
  }
  void* device = nullptr;
  check_cuda(cudaMalloc(&device, host.size() * sizeof(T)), what);
  check_cuda(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice),
             what);
  return static_cast<T*>(device);
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
    if (argc > 6) {
      throw std::invalid_argument(
          "usage: cuda_nvshmem_moe <tokens_per_rank> [hidden] [iterations] [warmup] [routing_cases]");
    }
    const auto tokens = gpu_bench::parse_moe_size_arg(argc, argv, 1, 16384U, "token count");
    const auto hidden = gpu_bench::parse_moe_size_arg(argc, argv, 2, 256U, "hidden size");
    const auto iterations = gpu_bench::parse_moe_positive_int_arg(argc, argv, 3, 100, "iteration count");
    const auto warmup = gpu_bench::parse_moe_positive_int_arg(argc, argv, 4, 20, "warmup count");
    const auto routing_cases = gpu_bench::parse_moe_routing_cases(argc, argv, 5);
    const auto payload_elements = gpu_bench::moe_checked_multiply(tokens, hidden, "MoE payload");
    const auto payload_bytes =
        gpu_bench::moe_checked_multiply(payload_elements, sizeof(float), "MoE payload");
    const auto bytes = gpu_bench::moe_checked_multiply(
        gpu_bench::moe_checked_multiply(2U, payload_elements, "MoE useful bytes"), sizeof(float),
        "MoE useful bytes");

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
    check_nvshmem(nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr), "nvshmemx_init_attr");
    nvshmem_initialized = true;

    const int pe = nvshmem_my_pe();
    const int pes = nvshmem_n_pes();
    if (pe != mpi_rank || pes != mpi_ranks) {
      throw std::runtime_error("NVSHMEM PE layout does not match MPI rank layout");
    }

    constexpr int block_size = 256;

    // Largest grid that can run concurrently -- the cooperative-launch ceiling.
    int blocks_per_sm = 0;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, moe_put_kernel,
                                                             block_size, 0),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
    std::size_t max_grid = static_cast<std::size_t>(blocks_per_sm > 0 ? blocks_per_sm : 1) *
                           static_cast<std::size_t>(sm_count);

    // Inter-node proxy cap; see the measured sweep at the top of this file.
    std::size_t block_cap = 0;
    if (const char* cap_env = std::getenv("GPU_BENCH_NVSHMEM_MAX_BLOCKS")) {
      block_cap = std::strtoull(cap_env, nullptr, 10);
    } else if (const char* nodes_env = std::getenv("GPU_BENCH_JOB_NODES")) {
      if (std::strtol(nodes_env, nullptr, 10) > 1) {
        block_cap = 64;
      }
    }
    if (block_cap > 0 && block_cap < max_grid) {
      max_grid = block_cap;
    }

    // Two blocks of pes counters: [0, pes) for dispatch, [pes, 2*pes) for
    // combine.
    const auto signal_bytes = 2U * static_cast<std::size_t>(pes) * sizeof(std::uint64_t);
    auto* signals = static_cast<std::uint64_t*>(nvshmem_malloc(signal_bytes));
    if (signals == nullptr) {
      throw std::runtime_error("failed to allocate NVSHMEM symmetric memory");
    }

    int all_cases_ok = 1;
    for (const auto routing : routing_cases) {
      std::vector<gpu_bench::moe_plan> plans;
      plans.reserve(static_cast<std::size_t>(pes));
      for (int plan_pe = 0; plan_pe < pes; ++plan_pe) {
        plans.push_back(gpu_bench::make_moe_plan(tokens, hidden, plan_pe, pes, routing));
      }
      const auto& plan = plans[static_cast<std::size_t>(pe)];
      const auto host_send = gpu_bench::pack_moe_send(plan);
      std::vector<float> host_dispatch(plan.recv_elements);
      std::vector<float> host_combined(plan.send_elements);

      const auto dispatch_elements = std::max<std::size_t>(
          gpu_bench::moe_checked_multiply(plan.max_expert_tokens, hidden, "MoE dispatch buffer"), 1U);
      const auto dispatch_bytes =
          gpu_bench::moe_checked_multiply(dispatch_elements, sizeof(float), "MoE dispatch buffer");
      auto* device_send = static_cast<float*>(nvshmem_malloc(payload_bytes));
      auto* device_dispatch = static_cast<float*>(nvshmem_malloc(dispatch_bytes));
      auto* device_combined = static_cast<float*>(nvshmem_malloc(payload_bytes));
      if (device_send == nullptr || device_dispatch == nullptr || device_combined == nullptr) {
        throw std::runtime_error("failed to allocate NVSHMEM symmetric memory");
      }

      check_cuda(cudaMemcpy(device_send, host_send.data(), payload_bytes, cudaMemcpyHostToDevice),
                 "cudaMemcpy(send)");
      check_cuda(cudaMemset(device_dispatch, 0, dispatch_bytes), "cudaMemset(dispatch)");
      check_cuda(cudaMemset(device_combined, 0, payload_bytes), "cudaMemset(combined)");
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(init)");

      // Dispatch: my tokens to the PE that owns each expert. Combine: the same
      // traffic reversed, so its active edges are the transpose of dispatch's.
      // Zero-count peers are skipped on both sides -- they neither signal nor
      // are waited on -- which is where the saving over barrier_all comes from
      // under skewed routing.
      std::vector<put_work> dispatch_work;
      std::vector<put_work> combine_work;
      std::vector<int> dispatch_peers;
      std::vector<int> combine_peers;
      std::vector<int> dispatch_wait(static_cast<std::size_t>(pes), 0);
      std::vector<int> combine_wait(static_cast<std::size_t>(pes), 0);

      for (int peer = 0; peer < pes; ++peer) {
        const auto index = static_cast<std::size_t>(peer);
        const auto self = static_cast<std::size_t>(pe);

        const auto send_count = static_cast<std::size_t>(plan.send_counts[index]);
        if (send_count != 0) {
          append_work(dispatch_work, device_send + plan.send_displacements[index],
                      device_dispatch + plans[index].recv_displacements[self], send_count, peer);
          if (peer != pe) {
            dispatch_peers.push_back(peer);
            // Peer sends me the combine leg for exactly the tokens I sent it.
            combine_wait[index] = 1;
          }
        }

        const auto recv_count = static_cast<std::size_t>(plan.recv_counts[index]);
        if (recv_count != 0) {
          append_work(combine_work, device_dispatch + plan.recv_displacements[index],
                      device_combined + plans[index].send_displacements[self], recv_count, peer);
          if (peer != pe) {
            combine_peers.push_back(peer);
            // Peer sends me the dispatch leg for the tokens it routes to me.
            dispatch_wait[index] = 1;
          }
        }
      }

      auto* dev_dispatch_work = upload(dispatch_work, "cudaMalloc(dispatch work)");
      auto* dev_combine_work = upload(combine_work, "cudaMalloc(combine work)");
      auto* dev_dispatch_peers = upload(dispatch_peers, "cudaMalloc(dispatch peers)");
      auto* dev_combine_peers = upload(combine_peers, "cudaMalloc(combine peers)");
      auto* dev_dispatch_wait = upload(dispatch_wait, "cudaMalloc(dispatch wait)");
      auto* dev_combine_wait = upload(combine_wait, "cudaMalloc(combine wait)");

      moe_phase dispatch{dev_dispatch_work, static_cast<int>(dispatch_work.size()),
                         dev_dispatch_peers, static_cast<int>(dispatch_peers.size()),
                         dev_dispatch_wait, signals};
      moe_phase combine{dev_combine_work, static_cast<int>(combine_work.size()),
                        dev_combine_peers, static_cast<int>(combine_peers.size()),
                        dev_combine_wait, signals + pes};

      // One block per work item, capped by the cooperative ceiling; the blocks
      // stride over the list so a hot expert's many pieces spread across them.
      const auto work_blocks =
          std::max<std::size_t>(std::max(dispatch_work.size(), combine_work.size()), 1U);
      const auto nblocks = std::min(work_blocks, max_grid);

      // Signal counters are reset per routing case, not per iteration. Within a
      // case the active edge set is fixed, so every active slot advances by one
      // per step and the threshold base + it holds for all of them. Across
      // cases the edge set changes -- a slot active under `uniform` can be idle
      // under `hotspot80` -- so a base that kept running would set a threshold
      // no one was going to reach, and the waiter would spin forever.
      //
      // The reset is safe only here. Every signal raised is also waited on (the
      // two edge sets are exact transposes), the previous case ended with
      // cudaDeviceSynchronize inside its last step, and the barriers below
      // bracket the memset: the first drains the previous case, the second
      // keeps a peer's first signal of this case from racing my memset. A reset
      // anywhere inside the timed loop would race in-flight proxied signals and
      // drop one, which is the hang halo_1d.cu documents.
      nvshmem_barrier_all();
      check_cuda(cudaMemset(signals, 0, signal_bytes), "cudaMemset(signals)");
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(signals)");
      nvshmem_barrier_all();
      std::uint64_t signal_base = 0;

      const dim3 grid(static_cast<unsigned>(nblocks));
      const dim3 block(block_size);
      const dim3 signal_grid(1);

      // One phase: move the payload, then raise and await the signals. The
      // kernel boundary between the two carries the ordering that
      // cg::grid_group::sync() used to.
      const auto run_phase = [&](const moe_phase& ph, std::uint64_t threshold) {
        moe_put_kernel<<<grid, block>>>(ph.work, ph.work_count, pe);
        check_cuda(cudaGetLastError(), "moe_put_kernel");

        // At one PE every transfer is a local copy: nothing to signal, nothing
        // to wait for, and the signal kernel's two loops are both empty. Skip
        // the launch rather than pay for a no-op -- measured at 1n1g, the two
        // collective launches were ~8.8 us of a 92.8 us iteration, against 84.0
        // us for the host-driven version this replaced.
        //
        // The condition is `pes == 1` and not "this PE has no peers": a
        // collective launch must be called by every PE, so the predicate has to
        // be globally uniform. Under skewed routing an individual PE can have
        // an empty edge set while others do not, and skipping on that would
        // hang.
        if (pes == 1) {
          return;
        }

        const int* peers = ph.signal_peers;
        int count = ph.signal_count;
        const int* active = ph.wait_active;
        int pes_v = pes;
        std::uint64_t* sig = ph.signals;
        int pe_v = pe;
        std::uint64_t th = threshold;
        void* args[] = {&peers, &count, &active, &pes_v, &sig, &pe_v, &th};
        check_nvshmem(nvshmemx_collective_launch(
                          reinterpret_cast<const void*>(moe_signal_kernel),
                          signal_grid, block, args, 0, 0),
                      "nvshmemx_collective_launch(moe_signal_kernel)");
      };

      // One launch per measured iteration, matching the NCCL backend's one
      // group submit + one stream sync. `signal_base` counts launches within
      // this routing case, warmup included, and is identical on every PE
      // because every PE runs the same warmup + iterations -- so every active
      // slot holds exactly that count and the threshold below is reached.
      const auto step = [&]() {
        const std::uint64_t threshold = signal_base + 1U;
        run_phase(dispatch, threshold);
        run_phase(combine, threshold);
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(moe step)");
        signal_base += 1U;
      };

      // The signal-reset barrier above already aligned the PEs; this one keeps
      // the MPI-side alignment the other backends also do before timing.
      MPI_Barrier(MPI_COMM_WORLD);
      const auto stats = gpu_bench::run_benchmark(warmup, iterations, step);

      const auto global = gpu_bench::collective_stats(stats);
      const double time_per_iter = global.avg_s;

      if (plan.recv_elements > 0) {
        check_cuda(cudaMemcpy(host_dispatch.data(), device_dispatch, plan.recv_elements * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy(dispatch)");
      }
      check_cuda(cudaMemcpy(host_combined.data(), device_combined, payload_bytes, cudaMemcpyDeviceToHost),
                 "cudaMemcpy(combined)");
      int local_ok = gpu_bench::validate_moe_dispatch(host_dispatch.data(), plan) &&
                             gpu_bench::validate_moe_combined(host_combined.data(), host_send)
                         ? 1
                         : 0;
      int global_ok = 1;
      MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      all_cases_ok = std::min(all_cases_ok, global_ok);

      check_cuda(cudaFree(dev_combine_wait), "cudaFree(combine wait)");
      check_cuda(cudaFree(dev_dispatch_wait), "cudaFree(dispatch wait)");
      check_cuda(cudaFree(dev_combine_peers), "cudaFree(combine peers)");
      check_cuda(cudaFree(dev_dispatch_peers), "cudaFree(dispatch peers)");
      check_cuda(cudaFree(dev_combine_work), "cudaFree(combine work)");
      check_cuda(cudaFree(dev_dispatch_work), "cudaFree(dispatch work)");

      nvshmem_free(device_combined);
      nvshmem_free(device_dispatch);
      nvshmem_free(device_send);

      if (pe == 0) {
        const double useful_gbytes_per_s = time_per_iter > 0.0 ? static_cast<double>(bytes) / time_per_iter / 1.0e9
                                                               : 0.0;
        const double imbalance = static_cast<double>(plan.max_expert_tokens) / static_cast<double>(tokens);
        std::ostringstream extra;
        extra << "case=" << gpu_bench::moe_routing_name(routing)
              << " routing=" << gpu_bench::moe_routing_name(routing) << " tokens=" << tokens
              << " hidden=" << hidden << " top_k=1 max_expert_tokens=" << plan.max_expert_tokens
              << " expert_imbalance=" << imbalance << " useful_gbytes_per_s=" << useful_gbytes_per_s
              << " status=" << (global_ok ? "OK" : "ERROR") << " memory=device_symmetric"
              << " submission=device-initiated completion=quiet-signal blocks=" << nblocks
              << " kernels=split";

        gpu_bench::bench_report report;
        report.name = "cuda_nvshmem_moe";
        report.n = tokens;
        report.ranks = pes;
        report.bytes_per_iter = bytes;
        report.iterations = iterations;
        report.warmup = warmup;
        report.time_per_iter_s = time_per_iter;
        report.min_s = global.min_s;
        report.max_s = global.max_s;
        gpu_bench::set_distribution(report, global);
        report.valid = global_ok != 0;
        report.extra = extra.str();
        gpu_bench::print_report(report);
      }

      // Every PE has left the timed loop and its validation reads; the next
      // routing case reallocates symmetric memory, which is collective.
      nvshmem_barrier_all();
    }

    nvshmem_free(signals);
    nvshmem_finalize();
    nvshmem_initialized = false;
    MPI_Finalize();
    return all_cases_ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "rank " << mpi_rank << ": " << error.what() << '\n';
    if (nvshmem_initialized) {
      nvshmem_global_exit(1);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
