# Leonardo Vector Add Experiments

These jobs follow the same style as the Leonardo experiment scripts in `acg-sycl`: fixed topology launchers, Slurm logs under `logs/`, and per-trial benchmark output under `results/`.

## Build

CUDA + MPI:

```bash
source cluster/leonardo/environment.sh cuda
cmake --preset leonardo-cuda-mpi
cmake --build --preset leonardo-cuda-mpi
```

SYCL + MPI:

```bash
source cluster/leonardo/environment.sh sycl
cmake --preset leonardo-sycl-mpi
cmake --build --preset leonardo-sycl-mpi
```

## Submit

```bash
sbatch cluster/leonardo/experiments/vector_add/cuda_mpi/1n1g.sh
sbatch cluster/leonardo/experiments/vector_add/cuda_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/vector_add/cuda_mpi/2n4g.sh
```

```bash
sbatch cluster/leonardo/experiments/vector_add/sycl_mpi/1n1g.sh
sbatch cluster/leonardo/experiments/vector_add/sycl_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/vector_add/sycl_mpi/2n4g.sh
```

## Overrides

```bash
CP_N=16777216
CP_NTRIALS=5
CP_RESULT_NAME=vector-add-cuda-mpi-custom
CP_BINARY=/path/to/vector_add_binary
```

Example:

```bash
CP_N=16777216 CP_NTRIALS=5 sbatch cluster/leonardo/experiments/vector_add/cuda_mpi/1n4g.sh
```

Outputs are written to:

```text
results/<result-name>/vector_add/<job-name>-<job-id>-<trial>-stdout.txt
results/<result-name>/vector_add/<job-name>-<job-id>-<trial>-stderr.txt
```
