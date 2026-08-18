#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_bench {

enum class moe_routing { uniform, locality80, hotspot80 };

inline const char* moe_routing_name(moe_routing routing) {
  switch (routing) {
    case moe_routing::uniform:
      return "uniform";
    case moe_routing::locality80:
      return "locality80";
    case moe_routing::hotspot80:
      return "hotspot80";
  }
  throw std::invalid_argument("unknown MoE routing case");
}

// SplitMix64 with its fixed published constants gives every backend the same
// integer mapping without depending on implementation-defined std::hash behavior.
inline std::uint64_t moe_hash(std::uint64_t value) {
  value += UINT64_C(0x9e3779b97f4a7c15);
  value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31U);
}

inline std::uint64_t moe_hash_tuple(int source, std::size_t token, std::uint64_t salt) {
  const auto source_hash = moe_hash(static_cast<std::uint64_t>(source) ^ UINT64_C(0x243f6a8885a308d3));
  const auto token_hash = moe_hash(static_cast<std::uint64_t>(token) ^ UINT64_C(0x13198a2e03707344));
  return moe_hash(source_hash ^ token_hash ^ salt);
}

inline int moe_expert_for(int source, std::size_t token, int ranks, moe_routing routing) {
  if (ranks <= 0 || source < 0 || source >= ranks) {
    throw std::invalid_argument("invalid rank information for MoE routing");
  }
  if (ranks == 1) {
    return 0;
  }

  const auto decision = moe_hash_tuple(source, token, UINT64_C(0xa4093822299f31d0));
  const auto selection = moe_hash_tuple(source, token, UINT64_C(0x082efa98ec4e6c89));
  switch (routing) {
    case moe_routing::uniform:
      return static_cast<int>(decision % static_cast<std::uint64_t>(ranks));
    case moe_routing::locality80:
      if (decision % 100U < 80U) {
        return source;
      }
      {
        const int other = static_cast<int>(selection % static_cast<std::uint64_t>(ranks - 1));
        return other >= source ? other + 1 : other;
      }
    case moe_routing::hotspot80:
      if (decision % 100U < 80U) {
        return 0;
      }
      return 1 + static_cast<int>(selection % static_cast<std::uint64_t>(ranks - 1));
  }
  throw std::invalid_argument("unknown MoE routing case");
}

inline float moe_payload_value(int source, std::size_t token, std::size_t feature) {
  const auto token_key = moe_hash_tuple(source, token, UINT64_C(0x452821e638d01377));
  const auto value = moe_hash(token_key ^ moe_hash(static_cast<std::uint64_t>(feature)));
  return static_cast<float>(value & UINT64_C(0x00ffffff));
}

inline std::size_t moe_checked_multiply(std::size_t lhs, std::size_t rhs, const char* description) {
  if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs) {
    throw std::overflow_error(std::string(description) + " size overflows size_t");
  }
  return lhs * rhs;
}

inline std::size_t moe_checked_add(std::size_t lhs, std::size_t rhs, const char* description) {
  if (lhs > std::numeric_limits<std::size_t>::max() - rhs) {
    throw std::overflow_error(std::string(description) + " size overflows size_t");
  }
  return lhs + rhs;
}

inline std::size_t parse_moe_size_arg(int argc, char** argv, int index, std::size_t default_value,
                                      const char* description) {
  if (argc <= index) {
    return default_value;
  }
  errno = 0;
  char* end = nullptr;
  const auto value = std::strtoull(argv[index], &end, 10);
  if (argv[index][0] == '-' || errno == ERANGE || end == argv[index] || *end != '\0' || value == 0 ||
      value > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument(std::string("expected a positive ") + description);
  }
  return static_cast<std::size_t>(value);
}

inline int parse_moe_positive_int_arg(int argc, char** argv, int index, int default_value,
                                      const char* description) {
  if (argc <= index) {
    return default_value;
  }
  errno = 0;
  char* end = nullptr;
  const auto value = std::strtol(argv[index], &end, 10);
  if (errno == ERANGE || end == argv[index] || *end != '\0' || value <= 0 ||
      value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string("expected a positive ") + description);
  }
  return static_cast<int>(value);
}

inline moe_routing parse_moe_routing(const std::string& value) {
  if (value == "uniform") {
    return moe_routing::uniform;
  }
  if (value == "locality80") {
    return moe_routing::locality80;
  }
  if (value == "hotspot80") {
    return moe_routing::hotspot80;
  }
  throw std::invalid_argument("unknown MoE routing case: " + value);
}

inline std::vector<moe_routing> parse_moe_routing_cases(int argc, char** argv, int index) {
  if (argc <= index) {
    return {moe_routing::uniform, moe_routing::locality80, moe_routing::hotspot80};
  }

  const std::string input(argv[index]);
  std::vector<moe_routing> cases;
  std::size_t begin = 0;
  while (begin <= input.size()) {
    const auto comma = input.find(',', begin);
    const auto end = comma == std::string::npos ? input.size() : comma;
    if (end == begin) {
      throw std::invalid_argument("expected a comma-separated list of MoE routing cases");
    }
    cases.push_back(parse_moe_routing(input.substr(begin, end - begin)));
    if (comma == std::string::npos) {
      break;
    }
    begin = comma + 1U;
  }
  return cases;
}

struct moe_plan {
  std::size_t tokens = 0;
  std::size_t hidden = 0;
  int rank = 0;
  int ranks = 0;
  moe_routing routing = moe_routing::uniform;
  std::vector<std::size_t> send_token_counts;
  std::vector<std::size_t> send_token_displacements;
  std::vector<std::size_t> recv_token_counts;
  std::vector<std::size_t> recv_token_displacements;
  std::vector<int> send_counts;
  std::vector<int> send_displacements;
  std::vector<int> recv_counts;
  std::vector<int> recv_displacements;
  std::size_t send_elements = 0;
  std::size_t recv_elements = 0;
  std::size_t max_expert_tokens = 0;
};

inline int moe_mpi_value(std::size_t value, const char* description) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(std::string(description) + " exceeds MPI int range");
  }
  return static_cast<int>(value);
}

inline moe_plan make_moe_plan(std::size_t tokens, std::size_t hidden, int rank, int ranks,
                              moe_routing routing) {
  if (tokens == 0 || hidden == 0) {
    throw std::invalid_argument("MoE tokens and hidden size must be positive");
  }
  if (ranks <= 0 || rank < 0 || rank >= ranks) {
    throw std::invalid_argument("invalid rank information for MoE plan");
  }

  const auto rank_count = static_cast<std::size_t>(ranks);
  std::vector<std::size_t> matrix(moe_checked_multiply(rank_count, rank_count, "MoE count matrix"), 0U);
  for (int source = 0; source < ranks; ++source) {
    for (std::size_t token = 0; token < tokens; ++token) {
      const int expert = moe_expert_for(source, token, ranks, routing);
      ++matrix[static_cast<std::size_t>(source) * rank_count + static_cast<std::size_t>(expert)];
    }
  }

  moe_plan plan;
  plan.tokens = tokens;
  plan.hidden = hidden;
  plan.rank = rank;
  plan.ranks = ranks;
  plan.routing = routing;
  plan.send_token_counts.resize(rank_count);
  plan.send_token_displacements.resize(rank_count);
  plan.recv_token_counts.resize(rank_count);
  plan.recv_token_displacements.resize(rank_count);
  plan.send_counts.resize(rank_count);
  plan.send_displacements.resize(rank_count);
  plan.recv_counts.resize(rank_count);
  plan.recv_displacements.resize(rank_count);

  std::size_t send_tokens = 0;
  std::size_t recv_tokens = 0;
  for (int peer = 0; peer < ranks; ++peer) {
    const auto peer_index = static_cast<std::size_t>(peer);
    const auto send_count = matrix[static_cast<std::size_t>(rank) * rank_count + peer_index];
    const auto recv_count = matrix[peer_index * rank_count + static_cast<std::size_t>(rank)];
    plan.send_token_counts[peer_index] = send_count;
    plan.send_token_displacements[peer_index] = send_tokens;
    plan.recv_token_counts[peer_index] = recv_count;
    plan.recv_token_displacements[peer_index] = recv_tokens;
    send_tokens = moe_checked_add(send_tokens, send_count, "MoE send buffer");
    recv_tokens = moe_checked_add(recv_tokens, recv_count, "MoE receive buffer");
  }
  if (send_tokens != tokens) {
    throw std::logic_error("MoE plan did not route every local token exactly once");
  }

  for (int expert = 0; expert < ranks; ++expert) {
    std::size_t expert_tokens = 0;
    for (int source = 0; source < ranks; ++source) {
      expert_tokens = moe_checked_add(
          expert_tokens,
          matrix[static_cast<std::size_t>(source) * rank_count + static_cast<std::size_t>(expert)],
          "MoE expert token count");
    }
    if (expert_tokens > plan.max_expert_tokens) {
      plan.max_expert_tokens = expert_tokens;
    }
  }

  plan.send_elements = moe_checked_multiply(send_tokens, hidden, "MoE send buffer");
  plan.recv_elements = moe_checked_multiply(recv_tokens, hidden, "MoE receive buffer");
  for (std::size_t peer = 0; peer < rank_count; ++peer) {
    plan.send_counts[peer] =
        moe_mpi_value(moe_checked_multiply(plan.send_token_counts[peer], hidden, "MoE send count"),
                      "MoE send count");
    plan.send_displacements[peer] =
        moe_mpi_value(moe_checked_multiply(plan.send_token_displacements[peer], hidden, "MoE send displacement"),
                      "MoE send displacement");
    plan.recv_counts[peer] =
        moe_mpi_value(moe_checked_multiply(plan.recv_token_counts[peer], hidden, "MoE receive count"),
                      "MoE receive count");
    plan.recv_displacements[peer] = moe_mpi_value(
        moe_checked_multiply(plan.recv_token_displacements[peer], hidden, "MoE receive displacement"),
        "MoE receive displacement");
  }
  moe_mpi_value(plan.send_elements, "MoE send buffer");
  moe_mpi_value(plan.recv_elements, "MoE receive buffer");
  return plan;
}

inline std::vector<float> pack_moe_send(const moe_plan& plan) {
  std::vector<float> packed(plan.send_elements);
  auto cursor = plan.send_token_displacements;
  for (std::size_t token = 0; token < plan.tokens; ++token) {
    const int expert = moe_expert_for(plan.rank, token, plan.ranks, plan.routing);
    const auto packed_token = cursor[static_cast<std::size_t>(expert)]++;
    const auto offset = packed_token * plan.hidden;
    for (std::size_t feature = 0; feature < plan.hidden; ++feature) {
      packed[offset + feature] = moe_payload_value(plan.rank, token, feature);
    }
  }
  return packed;
}

// MPI_Alltoallv places dispatch data in source-rank blocks. Within each block,
// routing-order packing means matching source tokens remain in increasing order.
inline bool validate_moe_dispatch(const float* dispatch, const moe_plan& plan) {
  for (int source = 0; source < plan.ranks; ++source) {
    std::size_t incoming_token = plan.recv_token_displacements[static_cast<std::size_t>(source)];
    for (std::size_t token = 0; token < plan.tokens; ++token) {
      if (moe_expert_for(source, token, plan.ranks, plan.routing) != plan.rank) {
        continue;
      }
      const auto offset = incoming_token++ * plan.hidden;
      for (std::size_t feature = 0; feature < plan.hidden; ++feature) {
        if (dispatch[offset + feature] != moe_payload_value(source, token, feature)) {
          return false;
        }
      }
    }
    const auto source_end = plan.recv_token_displacements[static_cast<std::size_t>(source)] +
                            plan.recv_token_counts[static_cast<std::size_t>(source)];
    if (incoming_token != source_end) {
      return false;
    }
  }
  return true;
}

inline bool validate_moe_combined(const float* combined, const std::vector<float>& packed) {
  for (std::size_t i = 0; i < packed.size(); ++i) {
    if (combined[i] != packed[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace gpu_bench
