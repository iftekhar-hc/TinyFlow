/*!
 *  Copyright (c) 2017 by Contributors
 * \file cuda_device_api.cc
 * \brief GPU specific API
 */

#include "./cuda_device_api.h"
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CALL(func)                                                        \
  {                                                                            \
    cudaError_t e = (func);                                                    \
    assert((e == cudaSuccess) || (e == cudaErrorCudartUnloading));             \
  }

namespace dlsys {
namespace runtime {

CUDADeviceAPI *CUDADeviceAPI::Global() {
  static CUDADeviceAPI inst;
  return &inst;
}

static void GPUCopy(const void *from, void *to, size_t size,
                    cudaMemcpyKind kind, cudaStream_t stream) {
  if (stream != 0) {
    CUDA_CALL(cudaMemcpyAsync(to, from, size, kind, stream));
  } else {
    CUDA_CALL(cudaMemcpy(to, from, size, kind));
  }
}

void *CUDADeviceAPI::AllocDataSpace(DLContext ctx, size_t size,
                                    size_t alignment) {
  CUDA_CALL(cudaSetDevice(ctx.device_id));
  assert((256 % alignment) == 0U);
  // Round up to 256-byte alignment (minimum 256 bytes)
  size_t rounded_size = ((size + 255) / 256) * 256;
  if (rounded_size == 0) {
    rounded_size = 256;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  void *ret = nullptr;
  if (use_pool_) {
    auto it = free_bins_.find(rounded_size);
    if (it != free_bins_.end() && !it->second.empty()) {
      ret = it->second.back();
      it->second.pop_back();
    }
  }
  if (ret == nullptr) {
    CUDA_CALL(cudaMalloc(&ret, rounded_size));
    reserved_bytes_ += rounded_size;
    malloc_count_ += 1;
  }
  allocated_sizes_[ret] = rounded_size;
  active_bytes_ += rounded_size;
  if (active_bytes_ > peak_bytes_) {
    peak_bytes_ = active_bytes_;
  }
  return ret;
}

void CUDADeviceAPI::FreeDataSpace(DLContext ctx, void *ptr) {
  if (ptr == nullptr) {
    return;
  }
  CUDA_CALL(cudaSetDevice(ctx.device_id));
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocated_sizes_.find(ptr);
  if (it != allocated_sizes_.end()) {
    size_t rounded_size = it->second;
    allocated_sizes_.erase(it);
    if (active_bytes_ >= rounded_size) {
      active_bytes_ -= rounded_size;
    } else {
      active_bytes_ = 0;
    }
    if (use_pool_) {
      free_bins_[rounded_size].push_back(ptr);
      return;
    } else {
      if (reserved_bytes_ >= rounded_size) {
        reserved_bytes_ -= rounded_size;
      }
    }
  }
  CUDA_CALL(cudaFree(ptr));
}

void CUDADeviceAPI::EmptyCache() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto &kv : free_bins_) {
    size_t bin_size = kv.first;
    for (void *ptr : kv.second) {
      cudaError_t e = cudaFree(ptr);
      (void)e;
      if (reserved_bytes_ >= bin_size) {
        reserved_bytes_ -= bin_size;
      } else {
        reserved_bytes_ = 0;
      }
    }
    kv.second.clear();
  }
  free_bins_.clear();
}

void CUDADeviceAPI::CopyDataFromTo(const void *from, void *to, size_t size,
                                   DLContext ctx_from, DLContext ctx_to,
                                   DLStreamHandle stream) {
  cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
  if (ctx_from.device_type == kGPU && ctx_to.device_type == kGPU) {
    CUDA_CALL(cudaSetDevice(ctx_from.device_id));
    if (ctx_from.device_id == ctx_to.device_id) {
      GPUCopy(from, to, size, cudaMemcpyDeviceToDevice, cu_stream);
    } else {
      cudaMemcpyPeerAsync(to, ctx_to.device_id, from, ctx_from.device_id, size,
                          cu_stream);
    }
  } else if (ctx_from.device_type == kGPU && ctx_to.device_type == kCPU) {
    CUDA_CALL(cudaSetDevice(ctx_from.device_id));
    GPUCopy(from, to, size, cudaMemcpyDeviceToHost, cu_stream);
  } else if (ctx_from.device_type == kCPU && ctx_to.device_type == kGPU) {
    CUDA_CALL(cudaSetDevice(ctx_to.device_id));
    GPUCopy(from, to, size, cudaMemcpyHostToDevice, cu_stream);
  } else {
    std::cerr << "expect copy from/to GPU or between GPU" << std::endl;
  }
}

void CUDADeviceAPI::StreamSync(DLContext ctx, DLStreamHandle stream) {
  CUDA_CALL(cudaSetDevice(ctx.device_id));
  CUDA_CALL(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
}

} // namespace runtime
} // namespace dlsys
