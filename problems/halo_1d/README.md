# 1D Halo Stencil

Purpose: neighbor communication and one-sided models.

Expected pattern: each rank updates an interior 1D segment and exchanges boundary values with neighboring ranks.

For SHMEM-style implementations (`cuda_nvshmem`, `oshmpi`), halo exchange should use one-sided operations: each PE writes its boundary value directly into the neighbor's ghost cell, then synchronizes before the stencil update.
