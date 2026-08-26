SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:

LEONARDO_CUDA_PRESETS := leonardo-cuda-mpi leonardo-cuda-nccl leonardo-cuda-nvshmem leonardo-oshmpi
LEONARDO_SYCL_PRESETS := leonardo-sycl-mpi leonardo-sycl-oneccl leonardo-sycl-oneccl-oshmpi
LEONARDO_PRESETS := $(LEONARDO_CUDA_PRESETS) $(LEONARDO_SYCL_PRESETS)

.PHONY: help bootstrap submit configure build clean leonardo leonardo-cuda leonardo-sycl leonardo-clean

help:
	@printf '%s\n' \
	  'Targets:' \
	  '  make bootstrap                  Build dependencies (oneCCL, OSHMPI) then the' \
	  '                                  benchmarks. TARGETS=... for specific ones,' \
	  '                                  see cluster/leonardo/bootstrap.sh --list' \
	  '  make submit                     Submit experiments (BENCH=allreduce ...)' \
	  '  make configure PRESET=<preset>  Configure one CMake preset' \
	  '  make build PRESET=<preset>      Build one CMake preset' \
	  '  make clean PRESET=<preset>      Remove one preset or group build directory' \
	  '                                  Groups: leonardo, leonardo-cuda, leonardo-sycl' \
	  '  make leonardo                   Build all Leonardo presets' \
	  '  make leonardo-cuda              Build Leonardo CUDA-stack presets' \
	  '  make leonardo-sycl              Build Leonardo SYCL-stack presets' \
	  '  make leonardo-clean             Remove Leonardo build directories'

bootstrap:
	./cluster/leonardo/bootstrap.sh $(TARGETS)

submit:
	cluster/harness/launch.sh --all $(BENCH)

configure:
	@test -n "$(PRESET)" || { echo 'missing PRESET=<preset>'; exit 2; }
	cmake --preset "$(PRESET)"

build:
	@test -n "$(PRESET)" || { echo 'missing PRESET=<preset>'; exit 2; }
	cmake --build --preset "$(PRESET)"

clean:
	@test -n "$(PRESET)" || { echo 'missing PRESET=<preset>'; exit 2; }
	case "$(PRESET)" in
	  leonardo) paths="$(addprefix build/,$(LEONARDO_PRESETS))" ;;
	  leonardo-cuda) paths="$(addprefix build/,$(LEONARDO_CUDA_PRESETS))" ;;
	  leonardo-sycl) paths="$(addprefix build/,$(LEONARDO_SYCL_PRESETS))" ;;
	  *) paths="build/$(PRESET)" ;;
	esac
	printf 'Removing:%s\n' " $${paths}"
	rm -rf $${paths}

leonardo: leonardo-cuda leonardo-sycl

leonardo-cuda:
	source cluster/leonardo/environment.sh cuda
	for preset in $(LEONARDO_CUDA_PRESETS); do
	  cmake --preset "$$preset"
	  cmake --build --preset "$$preset"
	done

leonardo-sycl:
	source cluster/leonardo/environment.sh sycl
	for preset in $(LEONARDO_SYCL_PRESETS); do
	  cmake --preset "$$preset"
	  cmake --build --preset "$$preset"
	done

leonardo-clean:
	$(MAKE) clean PRESET=leonardo
