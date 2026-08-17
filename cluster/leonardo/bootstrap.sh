#!/usr/bin/env bash
set -euo pipefail

# Builds the dependencies a preset needs, in order, then the playground.
#
#   ./cluster/leonardo/bootstrap.sh                 # default target set
#   ./cluster/leonardo/bootstrap.sh oneccl-oshmpi   # one target and its requires
#   ./cluster/leonardo/bootstrap.sh --list
#   CP_FORCE=1 ./cluster/leonardo/bootstrap.sh ...  # rebuild even if installed
#
# Each target is cluster/leonardo/build/<name>.sh and declares the stack it needs,
# what it requires, and a path that proves it is already built. Adding a backend
# means adding one file there - and one in runtime/, and one directory per
# experiment.
#
# The only prerequisite this cannot install for you is a DPC++ compiler; point
# DPCPP_HOME at it (see cluster/leonardo/env/sycl.sh).

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build_dir="$script_dir/build"

default_targets=(oneccl-oshmpi playground)

list_targets() {
    local f
    for f in "$build_dir"/*.sh; do
        [[ "$(basename "$f")" == _* ]] && continue
        basename "$f" .sh
    done
}

if [[ ${1:-} == --list ]]; then
    list_targets
    exit 0
fi

requested=("$@")
[[ ${#requested[@]} -eq 0 ]] && requested=("${default_targets[@]}")

# Depth-first over CP_BUILD_REQUIRES, so each target appears once, after its
# dependencies. Plain strings rather than associative arrays: the target count is
# small and this runs anywhere, including bash 3.2.
seen=" "
ordered=()
stack=""

resolve() {
    local target=$1 file="$build_dir/$1.sh"
    case "$seen" in *" $target "*) return 0 ;; esac
    if [[ ! -f "$file" ]]; then
        printf 'error: no such build target: %s\n' "$target" >&2
        printf 'available: %s\n' "$(list_targets | tr '\n' ' ')" >&2
        exit 2
    fi
    seen="$seen$target "

    # Sourced for metadata only; the guard in each target stops it running here.
    local CP_BUILD_REQUIRES="" CP_BUILD_STACK="" CP_BUILD_PROVIDES=""
    # shellcheck disable=SC1090
    source "$file"

    # Every target currently needs the sycl stack, and environment.sh mutates the
    # shell, so mixing stacks in one invocation would build against the wrong
    # toolchain. Refuse rather than do that silently.
    if [[ -n "$stack" && -n "$CP_BUILD_STACK" && "$CP_BUILD_STACK" != "$stack" ]]; then
        printf 'error: %s needs the %s stack, but %s is already selected\n' \
            "$target" "$CP_BUILD_STACK" "$stack" >&2
        printf 'build them in separate invocations\n' >&2
        exit 2
    fi
    [[ -n "$CP_BUILD_STACK" ]] && stack=$CP_BUILD_STACK

    local dep
    for dep in $CP_BUILD_REQUIRES; do resolve "$dep"; done
    ordered+=("$target")
}

for target in "${requested[@]}"; do resolve "$target"; done

printf 'building: %s\n' "${ordered[*]}"

# shellcheck disable=SC1090
source "$script_dir/environment.sh" "${stack:-sycl}"

for target in "${ordered[@]}"; do
    CP_BUILD_RUN=1 bash "$build_dir/$target.sh"
done

printf '\nbootstrap complete\n'
printf 'submit an experiment with:\n'
printf '  tools/submit_all.sh allreduce\n'
