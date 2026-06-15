# Dot Product

Purpose: collective communication.

Expected pattern: each rank computes a local partial dot product, then communicator-specific collectives produce the global scalar.
