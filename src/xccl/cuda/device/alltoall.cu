#include <mpi.h>

#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli.hpp"
#include "stats/collective_mpi.hpp"
#include "report.hpp"
#include "timing.hpp"
#include "validation.hpp"

// All-to-all issued from inside the kernel, through NCCL's device API.
//
// The measured operation is deliberately the same one src/xccl/cuda/microbench/
// alltoall.cu measures with the host API -- same layout, same validation, same
// timing loop -- so the pair isolates one variable: who initiates the transfer.
//
//   cuda_nccl         host enqueues a grouped ncclSend/ncclRecv per peer, per call
//   cuda_nccl_device  one kernel launch; the kernel puts to every peer itself
//
// The backend is named for the API, not the transport, because the device API
// has two and they are a runtime property of a run rather than a different
// programming model: GIN over the NIC, which this file uses, and LSA over
// NVLink, which is the natural second implementation of the same benchmark
// (nccl_device.h exposes ncclTeamLsa/ncclGetLsaPointer, and ll_a2a.h a
// slot-based low-latency alltoall primitive). A second transport belongs beside
// this file under the same backend, selected at run time and recorded in the
// log, the way cluster/leonardo/runtime/nccl-device.sh records NCCL_GIN_TYPE.
//
// What that buys is not a faster wire. It removes the per-operation host
// enqueue -- the ~20 us floor cuda_nccl pays before anything moves -- and lets
// communication be issued from inside a kernel that is doing something else.
// This benchmark measures the first half of that; a fused application benchmark
// is where the second half would show.
//
// Requirements, all checked at run time below rather than assumed:
//
//   - NCCL >= 2.28 for nccl_device.h. The CMake guard beside this file keeps
//     the target out of the build entirely against an older NCCL, so the check
//     here is about the *communicator*, not the headers.
//   - Buffers from ncclMemAlloc, registered as symmetric windows. Ordinary
//     cudaMalloc memory cannot be addressed by a remote rank's kernel.
//   - A GIN backend the machine can actually run. On Leonardo that is the CPU
//     proxy (NCCL_GIN_TYPE=2, pinned in cluster/leonardo/runtime/nccl-device.sh);
//     GDAKI initializes and then hangs, because the GPU cannot ring the NIC
//     doorbell here. NCCL picks GDAKI on its own, so leaving the selection to
//     the library is a hang, not a fallback.

namespace {

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
  }
}

void check_nccl(ncclResult_t status, const char* call) {
  if (status != ncclSuccess) {
    throw std::runtime_error(std::string(call) + ": " + ncclGetErrorString(status));
  }
}

// Grid shape. The CTA count is also declared to ncclDevCommCreate below: the
// barrier and signal resources are indexed by blockIdx.x, so the requirement
// and the launch must agree or a CTA indexes a resource that was never created.
//
// One CTA is the default, and the reason is worth stating because NCCL's own
// example uses 16. Every CTA runs its own ncclGinBarrierSession, twice -- once
// acquire, once release -- so the grid width multiplies the number of network
// barriers per exchange, and each of those is a collective over the CPU proxy.
// At 2n4g, 16 CTAs cost ~950 us per iteration, flat in message size, while only
// CTA 0 did any work: the put loop below is indexed by global thread id and
// there are only nRanks puts to issue, so with 8 ranks threads 0-7 of CTA 0
// satisfy all of them and the other 15 CTAs pay barrier cost for nothing.
//
// Overridable because the cost of that width is itself a measurement -- sweeping
// it shows what a GIN barrier costs on this machine.
constexpr int kThreadsPerCta = 512;

int gin_cta_count() {
  const char* value = std::getenv("GPU_BENCH_GIN_CTAS");
  if (value == nullptr || *value == '\0') return 1;
  const int parsed = std::atoi(value);
  if (parsed < 1) {
    throw std::runtime_error("GPU_BENCH_GIN_CTAS must be a positive integer");
  }
  return parsed;
}

// One put per peer, then wait for every peer's put to land.
//
// Adapted from NCCL's own docs/examples/06_device_api/02_alltoall_gin, and the
// mapping is the example's: puts are spread over the grid by global thread id,
// every put carries signal index blockIdx.x, and the single CTA that owns this
// rank's signal index waits for nRanks increments on it. The mapping is kept
// rather than simplified because it is the version whose signal and barrier
// bookkeeping is known to be correct.
//
// It also means the grid is not doing the work: there is one descriptor per
// peer to issue, the NIC does the transfer, and at one rank per GPU a single
// CTA issues all of them. Hence the default grid of one CTA.
__global__ void gin_alltoall_kernel(ncclWindow_t sendwin, std::size_t sendoffset,
                                    ncclWindow_t recvwin, std::size_t recvoffset,
                                    std::size_t count, ncclDevComm devComm) {
  const int ginContext = 0;
  const unsigned int signalIndex = blockIdx.x;
  ncclGin gin{devComm, ginContext};
  const uint64_t signalValue = gin.readSignal(signalIndex);

  ncclGinBarrierSession<ncclCoopCta> bar{ncclCoopCta(), gin, ncclTeamTagWorld(), blockIdx.x};
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::None);

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int nthreads = blockDim.x * gridDim.x;
  const std::size_t block_bytes = count * sizeof(float);

  for (int peer = tid; peer < devComm.nRanks; peer += nthreads) {
    gin.put(ncclTeamWorld(devComm), peer,
            recvwin, recvoffset + devComm.rank * block_bytes,
            sendwin, sendoffset + peer * block_bytes,
            block_bytes, ncclGin_WeakSignalInc{signalIndex});
  }

  const int receivingCta = (devComm.rank % nthreads) / blockDim.x;
  if (blockIdx.x == receivingCta) {
    gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + devComm.nRanks);
  }

  // flush() makes the local send buffer safe to reuse; it does not imply remote
  // completion. The closing barrier is what makes the exchange collective, and
  // it is why one kernel launch is one complete all-to-all for timing purposes.
  gin.flush(ncclCoopCta());
  bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::None);
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);

  ncclComm_t comm = nullptr;
  cudaStream_t stream = nullptr;

  try {
    const auto max_count = gpu_bench::parse_size_arg(argc, argv, 1U << 16U);
    const auto iterations = gpu_bench::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = gpu_bench::parse_positive_int_arg(argc, argv, 3, 20);
    const auto message_sizes = gpu_bench::parse_size_list_arg(argc, argv, 4, max_count);
    const auto max_total = static_cast<std::size_t>(ranks) * max_count;
    const auto max_bytes = max_total * sizeof(float);

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
      throw std::runtime_error("no CUDA devices available");
    }
    check_cuda(cudaSetDevice(rank % device_count), "cudaSetDevice");
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    ncclUniqueId id;
    if (rank == 0) {
      check_nccl(ncclGetUniqueId(&id), "ncclGetUniqueId");
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    check_nccl(ncclCommInitRank(&comm, ranks, id, rank), "ncclCommInitRank");

    // Ask the communicator rather than the headers. A build against 2.31 can
    // still land on a machine, a transport or a topology where the device API
    // or GIN is unavailable, and a run that silently measured something else
    // would be worse than one that stops here.
    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    check_nccl(ncclCommQueryProperties(comm, &props), "ncclCommQueryProperties");
    if (!props.deviceApiSupport) {
      throw std::runtime_error("this NCCL communicator has no device API support");
    }
    if (props.ginType == NCCL_GIN_TYPE_NONE) {
      throw std::runtime_error(
          "this NCCL communicator has no GIN backend; set NCCL_GIN_TYPE (2 = CPU proxy)");
    }

    // Symmetric memory, registered once at the largest swept size. Every size
    // then uses the leading ranks*count elements of the same window, which is
    // the same convention the host-API alltoall uses for its allocation.
    void* device_send = nullptr;
    void* device_recv = nullptr;
    check_nccl(ncclMemAlloc(&device_send, max_bytes), "ncclMemAlloc(send)");
    check_nccl(ncclMemAlloc(&device_recv, max_bytes), "ncclMemAlloc(recv)");

    ncclWindow_t send_window = nullptr;
    ncclWindow_t recv_window = nullptr;
    check_nccl(ncclCommWindowRegister(comm, device_send, max_bytes, &send_window,
                                      NCCL_WIN_COLL_SYMMETRIC),
               "ncclCommWindowRegister(send)");
    check_nccl(ncclCommWindowRegister(comm, device_recv, max_bytes, &recv_window,
                                      NCCL_WIN_COLL_SYMMETRIC),
               "ncclCommWindowRegister(recv)");

    const int ctas = gin_cta_count();

    ncclDevComm dev_comm;
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.worldGinBarrierCount = ctas;
    reqs.ginSignalCount = ctas;
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    check_nccl(ncclDevCommCreate(comm, &reqs, &dev_comm), "ncclDevCommCreate");

    std::vector<float> host_send(max_total);
    std::vector<float> host_recv(max_total);
    int all_sizes_ok = 1;

    for (const auto count : message_sizes) {
      const auto total = static_cast<std::size_t>(ranks) * count;
      const auto bytes = total * sizeof(float);

      gpu_bench::fill_alltoall_send(host_send.data(), rank, ranks, count);
      check_cuda(cudaMemcpy(device_send, host_send.data(), bytes, cudaMemcpyHostToDevice),
                 "cudaMemcpy(send)");
      check_cuda(cudaMemset(device_recv, 0, bytes), "cudaMemset(recv)");

      MPI_Barrier(MPI_COMM_WORLD);
      const auto stats = gpu_bench::run_benchmark(warmup, iterations, [&]() {
        gin_alltoall_kernel<<<ctas, kThreadsPerCta, 0, stream>>>(
            send_window, 0, recv_window, 0, count, dev_comm);
        check_cuda(cudaGetLastError(), "gin_alltoall_kernel launch");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(alltoall)");
      });

      const auto global = gpu_bench::collective_stats(stats);

      check_cuda(cudaMemcpy(host_recv.data(), device_recv, bytes, cudaMemcpyDeviceToHost),
                 "cudaMemcpy(recv)");
      const int local_ok = gpu_bench::validate_alltoall(host_recv.data(), rank, ranks, count) ? 1 : 0;
      int global_ok = 1;
      MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      all_sizes_ok = all_sizes_ok && global_ok;

      if (rank == 0) {
        gpu_bench::bench_report report;
        report.name = "cuda_nccl_device_alltoall";
        report.n = count;
        report.ranks = ranks;
        report.bytes_per_iter = bytes;
        report.iterations = iterations;
        report.warmup = warmup;
        report.time_per_iter_s = global.avg_s;
        report.min_s = global.min_s;
        report.max_s = global.max_s;
        gpu_bench::set_distribution(report, global);
        report.valid = global_ok != 0;
        // gin_type is recorded per run because it is the difference between a
        // CPU-rung doorbell and a GPU-rung one, and the two are not the same
        // measurement even though the kernel is identical.
        report.extra = "datatype=float32 count_per_peer=" + std::to_string(count) +
                       " gin_type=" + std::to_string(static_cast<int>(props.ginType)) +
                       " ctas=" + std::to_string(ctas) +
                       " bus_gbytes_per_s=" + std::to_string(gpu_bench::alltoall_bus_gbytes_per_s(
                                                  bytes, global.avg_s, ranks));
        gpu_bench::print_report(report);
      }
    }

    check_nccl(ncclDevCommDestroy(comm, &dev_comm), "ncclDevCommDestroy");
    check_nccl(ncclCommWindowDeregister(comm, send_window), "ncclCommWindowDeregister(send)");
    check_nccl(ncclCommWindowDeregister(comm, recv_window), "ncclCommWindowDeregister(recv)");
    check_nccl(ncclMemFree(device_send), "ncclMemFree(send)");
    check_nccl(ncclMemFree(device_recv), "ncclMemFree(recv)");
    // Finalize before destroy: the windows and the device communicator are
    // registered resources, and finalize is what flushes them.
    check_nccl(ncclCommFinalize(comm), "ncclCommFinalize");
    check_nccl(ncclCommDestroy(comm), "ncclCommDestroy");
    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");

    MPI_Finalize();
    return all_sizes_ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    if (comm != nullptr) ncclCommAbort(comm);
    if (stream != nullptr) cudaStreamDestroy(stream);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
