#pragma once

#include <oneapi/ccl.hpp>

namespace gpu_bench {

class ccl_group_scope {
 public:
  ccl_group_scope() {
    ccl::group_start();
    active_ = true;
  }

  ~ccl_group_scope() {
    if (active_) {
      try {
        ccl::group_end();
      } catch (...) {
      }
    }
  }

  ccl_group_scope(const ccl_group_scope&) = delete;
  ccl_group_scope& operator=(const ccl_group_scope&) = delete;

  void end() {
    active_ = false;
    ccl::group_end();
  }

 private:
  bool active_ = false;
};

}  // namespace gpu_bench
