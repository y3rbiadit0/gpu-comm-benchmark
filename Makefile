SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:

LEONARDO_CUDA_PRESETS := leonardo-cuda-mpi leonardo-cuda-nccl leonardo-cuda-nvshmem leonardo-oshmpi
LEONARDO_SYCL_PRESETS := leonardo-sycl-mpi leonardo-sycl-oneccl

.PHONY: help configure build clean leonardo leonardo-cuda leonardo-sycl leonardo-clean

help:
	@printf '%s\n' \
	  'Targets:' \
	  '  make configure PRESET=<preset>  Configure one CMake preset' \
	  '  make build PRESET=<preset>      Build one CMake preset' \
	  '  make clean PRESET=<preset>      Remove one build directory' \
	  '  make leonardo                   Build all Leonardo presets' \
	  '  make leonardo-cuda              Build Leonardo CUDA-stack presets' \
	  '  make leonardo-sycl              Build Leonardo SYCL-stack presets' \
	  '  make leonardo-clean             Remove Leonardo build directories'

configure:
	@test -n "$(PRESET)" || { echo 'missing PRESET=<preset>'; exit 2; }
	cmake --preset "$(PRESET)"

build:
	@test -n "$(PRESET)" || { echo 'missing PRESET=<preset>'; exit 2; }
	cmake --build --preset "$(PRESET)"

clean:
	@test -n "$(PRESET)" || { echo 'missing PRESET=<preset>'; exit 2; }
	rm -rf "build/$(PRESET)"

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
	rm -rf $(addprefix build/,$(LEONARDO_CUDA_PRESETS) $(LEONARDO_SYCL_PRESETS))
