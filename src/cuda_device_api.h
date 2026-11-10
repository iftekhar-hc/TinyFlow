/*!
 *  Copyright (c) 2017 by Contributors
 * \file device_api.h
 * \brief Device specific API
 */
#ifndef DLSYS_RUNTIME_CUDA_DEVICE_API_H_
#define DLSYS_RUNTIME_CUDA_DEVICE_API_H_

#include "c_runtime_api.h"
#include "device_api.h"
#include <cuda_runtime.h>

#include <assert.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace dlsys {
namespace runtime {

class CUDADeviceAPI : public DeviceAPI {
public:
  CUDADeviceAPI()
      : active_bytes_(0), peak_bytes_(0), reserved_bytes_(0),
        malloc_count_(0), use_pool_(true) {}

  ~CUDADeviceAPI() override { EmptyCache(); }

  void *AllocDataSpace(DLContext ctx, size_t size, size_t alignment) final;

  void FreeDataSpace(DLContext ctx, void *ptr) final;

  void CopyDataFromTo(const void *from, void *to, size_t size,
                      DLContext ctx_from, DLContext ctx_to,
                      DLStreamHandle stream) final;

  void StreamSync(DLContext ctx, DLStreamHandle stream) final;

  size_t GetActiveBytes() const { return active_bytes_; }
  size_t GetPeakBytes() const { return peak_bytes_; }
  size_t GetReservedBytes() const { return reserved_bytes_; }
  size_t GetMallocCount() const { return malloc_count_; }

  void ResetPeakStats() {
    std::lock_guard<std::mutex> lock(mutex_);
    peak_bytes_ = active_bytes_;
  }

  void SetUsePool(bool use_pool) {
    std::lock_guard<std::mutex> lock(mutex_);
    use_pool_ = use_pool;
  }

  void EmptyCache();

  static CUDADeviceAPI *Global();

private:
  mutable std::mutex mutex_;
  std::unordered_map<size_t, std::vector<void *>> free_bins_;
  std::unordered_map<void *, size_t> allocated_sizes_;
  size_t active_bytes_;
  size_t peak_bytes_;
  size_t reserved_bytes_;
  size_t malloc_count_;
  bool use_pool_;
};

} // namespace runtime
} // namespace dlsys
#endif // DLSYS_RUNTIME_CUDA_DEVICE_API_H_
