#pragma once

#include <chrono>

namespace comm_playground {

class wall_timer {
 public:
  wall_timer() : start_(clock::now()) {}

  double seconds() const {
    return std::chrono::duration<double>(clock::now() - start_).count();
  }

 private:
  using clock = std::chrono::steady_clock;
  clock::time_point start_;
};

}  // namespace comm_playground
