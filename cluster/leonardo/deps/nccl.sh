#!/usr/bin/env bash
set -euo pipefail

# A specific NCCL release, built from source against the nvhpc stack's CUDA.
#
# This target is only reached when GPU_BENCH_NCCL_VERSION names a release
# (layout.sh); the default, module, is the NCCL the nvhpc module already ships
# and needs no install. Each version lands in its own prefix, so several are
# installed side by side and a version can be A/B'd rather than swapped under
# the results -- the same arrangement deps/nvshmem.sh gets.
#
# Unlike NVSHMEM this is a source build, because NCCL is happy to be one:
#
#   - There is no redistributable that matches this stack. NVIDIA's NCCL
#     packages are built against a CUDA of their choosing, and the point of
#     building here is to hold the toolkit fixed at the one env/cuda.sh selects
#     (nvhpc 24.5's CUDA 12.4), so the library and the benchmarks that call it
#     agree on the CUDA runtime and on what nvcc emitted.
#   - NCCL's build is a plain Makefile that takes CUDA_HOME, NVCC and CXX, and
#     it needs no dependency this machine does not already have. The GDAKI/DOCA
#     sources dlopen libibverbs and libmlx5 through their own wrappers
#     (DOCA_VERBS_USE_NET_WRAPPER, set by makefiles/common.mk), so no MOFED
#     development headers are needed to compile them.
#
# The reason to run this at all is the device API: since 2.28 NCCL exposes
# communication that a kernel initiates itself (nccl_device.h) -- LSA over
# NVLink, and GIN, GPU-initiated networking, over the NIC. The NCCL inside the
# nvhpc 24.5 module predates all of it. Requirements the device API places on
# the machine, all of which Leonardo meets:
#
#   - CUDA >= 12.2 when compiling the GPU code, and sm_70 or newer. The GIN
#     headers gate themselves on exactly that (CUDA_VERSION >= 12020 &&
#     __CUDA_ARCH__ >= 700, src/include/nccl_device/gin/gin_device_common.h),
#     so CUDA 12.4 and sm_80 are inside the window and the probe at the bottom
#     of this file is what proves it rather than the version arithmetic.
#   - Driver >= 510.40.3, Volta or newer, NVIDIA NIC CX4 or newer.
#
# Which GIN backend actually runs is a runtime question, not a build one, and
# the answer on Leonardo is likely the CPU proxy rather than GDAKI: GDAKI is the
# same GPUDirect Async kernel-initiated path as NVSHMEM's classic IBGDA, which
# this machine cannot use (no PeerMappingOverride=1, see deps/nvshmem.sh), and
# it additionally wants rdma-core >= 44.0. Both backends are compiled in here.
# A selection between them (NCCL_GIN_PLUGIN and friends) belongs in
# runtime/mpi-cuda.sh, next to the other NCCL settings cuda_nccl runs with, and
# is worth writing once a benchmark actually calls the device API.

GPU_BENCH_BUILD_STACK=cuda
GPU_BENCH_BUILD_REQUIRES=""

# Only run standalone; bootstrap.sh sources this file for its metadata first.
[[ "${BASH_SOURCE[0]}" == "$0" || -n "${GPU_BENCH_BUILD_RUN:-}" ]] || return 0

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/_lib.sh"

version=${GPU_BENCH_NCCL_VERSION:-module}
if [[ "$version" == module ]]; then
    printf 'error: GPU_BENCH_NCCL_VERSION is module -- the nvhpc module ships\n' >&2
    printf 'an NCCL already and there is nothing to install. Name a release:\n' >&2
    printf '  GPU_BENCH_NCCL_VERSION=2.31.2-1 %s nccl\n' \
        "$(dirname "$script_dir")/bootstrap.sh" >&2
    exit 2
fi

# layout.sh maps the selected version to this prefix, and the same mapping is
# what the preset, LD_LIBRARY_PATH and the CMake hints resolve later. Reading it
# back here rather than recomputing it keeps one definition of where a version
# lives.
prefix=${NCCL_HOME:?layout.sh must map GPU_BENCH_NCCL_VERSION to NCCL_HOME}

# env/cuda.sh points NCCL_HOME at the nvhpc module's NCCL when no version is
# selected. If that is still the value here, layout.sh did not repoint it and
# `make install` would write into the module tree -- refuse rather than corrupt
# a site installation that every other run on this machine reads.
case "$prefix" in
    "$GPU_BENCH_PREFIX_ROOT"/*) ;;
    *)
        printf 'error: NCCL_HOME=%s is not under %s\n' "$prefix" "$GPU_BENCH_PREFIX_ROOT" >&2
        printf 'this target installs there; NCCL_HOME looks like the module NCCL,\n' >&2
        printf 'which means GPU_BENCH_NCCL_VERSION was not seen by layout.sh\n' >&2
        exit 2
        ;;
esac

# NCCL tags releases as v<version>, and <version> already carries the package
# revision (2.31.2-1). GPU_BENCH_NCCL_REF is for a branch or a commit.
repo=${GPU_BENCH_NCCL_REPO:-https://github.com/NVIDIA/nccl.git}
ref=${GPU_BENCH_NCCL_REF:-v$version}
src=$GPU_BENCH_SRC_DIR/nccl-$version
build=$GPU_BENCH_BUILD_DIR/nccl-$version
arch=${GPU_BENCH_CUDA_ARCH:-80}
gencode="-gencode=arch=compute_${arch},code=sm_${arch}"
jobs=${GPU_BENCH_JOBS:-${SLURM_CPUS_PER_TASK:-16}}

: "${CUDA_ROOT:?env/cuda.sh must define CUDA_ROOT}"
: "${CUDACXX:?env/cuda.sh must define CUDACXX}"

# The toolkit NCCL is built against. It defaults to the stack's, which is the
# point of building here -- but it is worth being able to move, because the
# library this build is compared against does not share it: the nvhpc 24.5
# module's NCCL reports itself as 2.18.5+cuda12.2, and Leonardo's driver is
# 12.2 as well (cudaDriverVersion 12020 in any NCCL_DEBUG=INFO log). A 12.4
# build therefore runs a toolkit ahead of the driver, which minor-version
# compatibility permits but which is a difference between the two libraries
# rather than a shared baseline. Pointing this at cuda/12.2 removes that
# difference; libnccl.so exports a C ABI and links cudart statically, which is
# exactly how the module's own 12.2 build is already consumed by benchmarks
# this stack compiles with 12.4.
#
#   GPU_BENCH_NCCL_CUDA_HOME=$(bash -lc 'module load cuda/12.2 >/dev/null 2>&1; echo $CUDA_HOME')
#
# The device-API probe below deliberately keeps using the stack's own nvcc: what
# it has to prove is that the benchmarks, which are built with that one, can
# compile against whatever this produces.
nccl_cuda_home=${GPU_BENCH_NCCL_CUDA_HOME:-$CUDA_ROOT}
[[ -x "$nccl_cuda_home/bin/nvcc" ]] || {
    printf 'error: no nvcc under GPU_BENCH_NCCL_CUDA_HOME=%s\n' "$nccl_cuda_home" >&2
    exit 2
}

# src/device/Makefile generates the collective kernels with a Python script and
# stops with a wall of text if it cannot. Say so here instead, where the fix
# (`module load python`) is next to the message.
command -v python3 >/dev/null 2>&1 || {
    printf 'error: NCCL generates its device kernels with python3, which is not on PATH\n' >&2
    printf 'add `module load python` to env/cuda.sh -- loading it in this shell\n' >&2
    printf 'will not help, the stack purges modules before it loads its own\n' >&2
    exit 2
}

# --- host compiler ----------------------------------------------------------
#
# NCCL compiles its host sources with $(CXX) and hands the same compiler to nvcc
# as -ccbin, so this one choice decides both. The cuda stack's compiler is
# nvc++, and building the library with the compiler that builds everything else
# on the stack is the point of doing this under nvhpc at all -- one toolchain,
# one libstdc++, no third compiler introduced the way deps/nvshmem.sh describes.
#
# It is also the choice most likely to be rejected: NCCL's flags are written for
# gcc, and -ccbin nvc++ is a supported but far less travelled nvcc path than
# -ccbin g++. libnccl.so exports a C ABI and links no C++ of ours, so a g++
# build is a perfectly good library for this stack to consume -- it is what
# NVIDIA's own packages are. Probe first and fall back to that, loudly, rather
# than fail twenty minutes into the device kernels. Set GPU_BENCH_NCCL_CXX to
# pin either one and skip the probe's opinion.
nccl_host_cxx_ok() {
    local cxx=$1 dir log
    dir=$(mktemp -d)
    log="$dir/probe.log"

    # A C++14 translation unit under NCCL's own host flags (makefiles/common.mk),
    # then the same compiler as nvcc's -ccbin under NCCL's device flags. The two
    # ways this fails are a flag nvc++ does not accept and an nvcc that will not
    # drive it; both surface here in a couple of seconds.
    cat >"$dir/probe.cc" <<'PROBE'
#include <memory>
#include <vector>
template <typename T> struct Box { T v; };
extern "C" int probe(int n) {
  auto f = [n](int x) { return x + n; };
  std::vector<Box<int>> v{{f(1)}};
  return v[0].v;
}
PROBE
    cat >"$dir/probe.cu" <<'PROBE'
__global__ void probe_kernel(int* out) { *out = threadIdx.x; }
PROBE

    if ! "$cxx" -fPIC -fvisibility=hidden -Wall -Wno-unused-function \
        -Wno-sign-compare -std=c++14 -Wvla -c "$dir/probe.cc" \
        -o "$dir/probe.o" >"$log" 2>&1; then
        printf '   %s rejects NCCL host flags:\n' "$cxx"
        sed 's/^/     /' "$log"
        rm -rf "$dir"
        return 1
    fi

    if ! "$CUDACXX" -ccbin "$cxx" $gencode -std=c++14 --expt-extended-lambda \
        -Xptxas -maxrregcount=96 -c "$dir/probe.cu" -o "$dir/probe.cu.o" \
        >"$log" 2>&1; then
        printf '   nvcc will not drive %s as -ccbin:\n' "$cxx"
        sed 's/^/     /' "$log"
        rm -rf "$dir"
        return 1
    fi

    rm -rf "$dir"
    return 0
}

gpu_bench_build_log "host compiler probe"
if [[ -n "${GPU_BENCH_NCCL_CXX:-}" ]]; then
    host_cxx=$GPU_BENCH_NCCL_CXX
    nccl_host_cxx_ok "$host_cxx" || {
        printf 'error: GPU_BENCH_NCCL_CXX=%s cannot build NCCL (see above)\n' "$host_cxx" >&2
        exit 2
    }
    printf '   %s ok (pinned by GPU_BENCH_NCCL_CXX)\n' "$host_cxx"
else
    host_cxx=${CXX:-nvc++}
    if nccl_host_cxx_ok "$host_cxx"; then
        printf '   %s ok\n' "$host_cxx"
    else
        printf '   falling back to g++; libnccl.so is a C ABI, so this changes\n'
        printf '   nothing the benchmarks see. Pin with GPU_BENCH_NCCL_CXX to override.\n'
        host_cxx=g++
        nccl_host_cxx_ok "$host_cxx" || {
            printf 'error: neither %s nor g++ can build NCCL\n' "${CXX:-nvc++}" >&2
            exit 2
        }
        printf '   %s ok\n' "$host_cxx"
    fi
fi

# --- build ------------------------------------------------------------------

if gpu_bench_build_done "$prefix/include/nccl_device.h"; then
    gpu_bench_build_log "NCCL already at $prefix"
else
    gpu_bench_build_log "NCCL source ($ref)"
    gpu_bench_clone_at "$repo" "$ref" "$src"

    [[ -f "$src/makefiles/version.mk" ]] || {
        printf 'error: %s does not look like an NCCL tree; wrong ref?\n' "$src" >&2
        exit 2
    }

    gpu_bench_build_log "NCCL $version (CXX=$host_cxx, CUDA_HOME=$nccl_cuda_home)"
    # NVCC_GENCODE is the difference between a five minute build and an hour of
    # it: left alone, common.mk compiles every collective for every architecture
    # the toolkit knows. Leonardo is A100 only.
    #
    # Everything else is left at NCCL's defaults, including CUDARTLIB=cudart_static
    # -- a statically linked CUDA runtime is one fewer library that has to agree
    # with whatever the job's LD_LIBRARY_PATH resolves first.
    # PREFIX is passed to the build as well as to the install: nccl.pc is
    # generated during the build with $(PREFIX) substituted into it, and the
    # install step will not regenerate a file it already considers up to date.
    make -C "$src" -j"$jobs" src.build \
        BUILDDIR="$build" \
        PREFIX="$prefix" \
        CUDA_HOME="$nccl_cuda_home" \
        NVCC="$nccl_cuda_home/bin/nvcc" \
        CXX="$host_cxx" \
        NVCC_GENCODE="$gencode"

    # Replace rather than merge: NCCL installs a version-suffixed libnccl.so.X.Y.Z
    # plus symlinks, so unpacking a second release over a first leaves the older
    # library in place with the newer symlinks pointing past it.
    rm -rf "$prefix"
    mkdir -p "$prefix"
    make -C "$src" src.install \
        BUILDDIR="$build" \
        PREFIX="$prefix" \
        CUDA_HOME="$nccl_cuda_home" \
        NVCC="$nccl_cuda_home/bin/nvcc" \
        CXX="$host_cxx" \
        NVCC_GENCODE="$gencode" >/dev/null
fi

for required in include/nccl.h include/nccl_device.h lib/libnccl.so \
                lib/libnccl_static.a lib/pkgconfig/nccl.pc; do
    [[ -e "$prefix/$required" ]] || {
        printf 'error: %s is missing from the install\n' "$required" >&2
        exit 2
    }
done

# What was actually installed, where the bootstrap log will keep it. The version
# is read back from the header rather than echoed from the variable, so a ref
# that does not match the version string cannot go unnoticed.
gpu_bench_build_log "NCCL build result"
grep -E '^#define NCCL_(MAJOR|MINOR|PATCH|SUFFIX|VERSION_CODE)' "$prefix/include/nccl.h" \
    | sed 's/^/   /' || true
if [[ -d "$prefix/include/nccl_device/gin/gdaki" ]]; then
    printf '   GIN backends: proxy (CPU-initiated) and GDAKI\n'
else
    printf '   GIN backends: proxy (CPU-initiated) only -- no GDAKI headers installed\n'
fi

# --- device API probe -------------------------------------------------------
#
# The library being installed says nothing about whether this stack's nvcc can
# compile a kernel against nccl_device.h, which is the entire reason for
# building a 2.31 here. NCCL ships an example that uses the GIN device API for
# an alltoall; building it with the stack's own nvcc is a more honest check than
# any probe written here, and it fails at the point where the fix is a version
# choice rather than in the middle of a preset build later.
example="$src/docs/examples/06_device_api/02_alltoall_gin/c"
if [[ ! -d "$example" ]]; then
    printf 'note: no device API example in this ref; skipping the compile probe\n'
else
    gpu_bench_build_log "device API probe: NCCL's own GIN alltoall example"
    # The build tree is absent when the install was already there and the build
    # above was skipped; the probe still runs, so it still needs somewhere to
    # put its log.
    mkdir -p "$build"
    if make -C "$example" -s \
        NCCL_HOME="$prefix" \
        CUDA_HOME="$CUDA_ROOT" \
        NVCC="$CUDACXX" \
        CXX="$host_cxx" \
        NVCC_GENCODE="$gencode" >"$build/device-probe.log" 2>&1; then
        printf '   nccl_device.h compiles and links with %s for sm_%s\n' \
            "$(basename "$CUDACXX")" "$arch"
        make -C "$example" -s clean >/dev/null 2>&1 || true
    else
        printf 'error: %s cannot compile a kernel against this NCCL device API\n' "$CUDACXX" >&2
        sed 's/^/   /' "$build/device-probe.log" >&2
        printf 'the device API needs CUDA >= 12.2 and sm_70+; this stack has\n' >&2
        printf 'CUDA_ROOT=%s and GPU_BENCH_CUDA_ARCH=%s\n' "$CUDA_ROOT" "$arch" >&2
        exit 2
    fi
fi

printf '\nNCCL %s: %s\n' "$version" "$prefix"
