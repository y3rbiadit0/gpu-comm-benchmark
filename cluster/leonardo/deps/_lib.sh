#!/usr/bin/env bash

# Shared helpers for cluster/leonardo/deps/<target>.sh.
#
# Each build target is a standalone script that can be run directly, or through
# cluster/leonardo/bootstrap.sh which resolves ordering. A target declares:
#
#   GPU_BENCH_BUILD_STACK      env/<stack>.sh it needs (cuda or sycl)
#   GPU_BENCH_BUILD_REQUIRES   other targets to build first, space separated
#
# Whether to skip work already done is each target's own call: an install prefix
# existing is enough for OSHMPI, while oneCCL always re-enters cmake so a source
# change is picked up.
#
# Everything is built under $SCRATCH, including clones. Only install prefixes
# land in $HOME.

# The layout owns every path a target produces; targets never invent their own.
# shellcheck disable=SC1090
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/layout.sh"

gpu_bench_build_log() { printf '\n== %s\n' "$*"; }

# gpu_bench_clone_at <url> <ref> <dir>
#
# Leaves <dir> checked out at <ref> in a detached head, whether it was just
# cloned or already present. Fetches only when the ref is missing, so re-running
# offline works once the clone exists.
gpu_bench_clone_at() {
    local url=$1 ref=$2 dir=$3

    if [[ ! -d "$dir/.git" ]]; then
        gpu_bench_build_log "cloning $url -> $dir"
        mkdir -p "$(dirname "$dir")"
        git clone --quiet "$url" "$dir"
    fi

    if ! git -C "$dir" rev-parse --verify --quiet "$ref^{commit}" >/dev/null; then
        git -C "$dir" fetch --quiet --tags origin "$ref" 2>/dev/null ||
            git -C "$dir" fetch --quiet origin
    fi

    local resolved
    resolved=$(git -C "$dir" rev-parse --verify --quiet "origin/$ref^{commit}" ||
               git -C "$dir" rev-parse --verify --quiet "$ref^{commit}") || {
        printf 'error: ref %s not found in %s\n' "$ref" "$url" >&2
        return 2
    }

    git -C "$dir" checkout --quiet --detach "$resolved"
    printf '   %s at %s (%s)\n' "$(basename "$dir")" "$ref" "${resolved:0:8}"
}

# gpu_bench_build_done <path> - true when the target is already built.
gpu_bench_build_done() {
    [[ -n "${GPU_BENCH_FORCE:-}" ]] && return 1
    [[ -e "$1" ]]
}
