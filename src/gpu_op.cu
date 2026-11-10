#include "./c_runtime_api.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <float.h>

#define CHECK_GPU_ERR(ans)                                                     \
  { gpu_assert((ans), __FILE__, __LINE__); }

inline void gpu_assert(cudaError_t code, const char *file, int line,
                       bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPU_assert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

static inline size_t GetNumElements(const DLArrayHandle arr) {
  size_t n = 1;
  for (int i = 0; i < arr->ndim; ++i) {
    n *= static_cast<size_t>(arr->shape[i]);
  }
  return n;
}

static inline int GetGridSize1D(size_t n, int threads = 256) {
  size_t blocks = (n + threads - 1) / threads;
  if (blocks == 0)
    blocks = 1;
  if (blocks > 65535)
    blocks = 65535;
  return static_cast<int>(blocks);
}

// ============================================================================
// Warp & Block Reduction Primitives
// ============================================================================

__inline__ __device__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__inline__ __device__ float warp_reduce_max(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

__inline__ __device__ float block_reduce_sum(float val, float *shared) {
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  val = warp_reduce_sum(val);
  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  int num_warps = (blockDim.x + 31) >> 5;
  val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;
  if (wid == 0) {
    val = warp_reduce_sum(val);
    if (lane == 0) {
      shared[0] = val;
    }
  }
  __syncthreads();
  return shared[0];
}

__inline__ __device__ float block_reduce_max(float val, float *shared) {
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  val = warp_reduce_max(val);
  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  int num_warps = (blockDim.x + 31) >> 5;
  val = (threadIdx.x < num_warps) ? shared[lane] : -FLT_MAX;
  if (wid == 0) {
    val = warp_reduce_max(val);
    if (lane == 0) {
      shared[0] = val;
    }
  }
  __syncthreads();
  return shared[0];
}

__inline__ __device__ void welford_combine(float b_mean, float b_m2,
                                           float b_count, float &mean,
                                           float &m2, float &count) {
  if (b_count == 0.0f) {
    return;
  }
  float new_count = count + b_count;
  float nb_over_n = b_count / new_count;
  float delta = b_mean - mean;
  mean += delta * nb_over_n;
  m2 += b_m2 + delta * delta * count * nb_over_n;
  count = new_count;
}

__inline__ __device__ void warp_reduce_welford(float &mean, float &m2,
                                               float &count) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    float b_mean = __shfl_down_sync(0xffffffff, mean, offset);
    float b_m2 = __shfl_down_sync(0xffffffff, m2, offset);
    float b_count = __shfl_down_sync(0xffffffff, count, offset);
    welford_combine(b_mean, b_m2, b_count, mean, m2, count);
  }
}

// ============================================================================
// Elementwise, Broadcast, and Reduction Kernels
// ============================================================================

__global__ void array_set_kernel(size_t n, float *arr, float value) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += blockDim.x * gridDim.x) {
    arr[idx] = value;
  }
}

__global__ void broadcast_to_kernel(size_t in_size, size_t out_size,
                                    const float *input, float *output) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < out_size;
       idx += blockDim.x * gridDim.x) {
    output[idx] = input[idx % in_size];
  }
}

__global__ void broadcast_scalar_kernel(size_t out_size, const float *scalar,
                                        float *output) {
  float val = scalar[0];
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < out_size;
       idx += blockDim.x * gridDim.x) {
    output[idx] = val;
  }
}

__global__ void broadcast_axis_one_kernel(int nrow, int ncol,
                                          const float *input, float *output) {
  size_t total = static_cast<size_t>(nrow) * ncol;
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    int row = idx / ncol;
    output[idx] = input[row];
  }
}

__global__ void reduce_sum_axis_zero_kernel(size_t nrow, size_t ncol,
                                            const float *input, float *output) {
  for (size_t col = blockIdx.x * blockDim.x + threadIdx.x; col < ncol;
       col += blockDim.x * gridDim.x) {
    float sum = 0.0f;
    for (size_t row = 0; row < nrow; ++row) {
      sum += input[row * ncol + col];
    }
    output[col] = sum;
  }
}

__global__ void reduce_mean_axis_one_kernel(int nrow, int ncol,
                                            const float *input, float *output) {
  __shared__ float shared[32];
  int row = blockIdx.x;
  if (row >= nrow)
    return;
  const float *row_in = input + static_cast<size_t>(row) * ncol;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < ncol; col += blockDim.x) {
    sum += row_in[col];
  }
  sum = block_reduce_sum(sum, shared);
  if (threadIdx.x == 0) {
    output[row] = sum / static_cast<float>(ncol);
  }
}

__global__ void elem_add_kernel(size_t n, const float *a, const float *b,
                                float *out) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += blockDim.x * gridDim.x) {
    out[idx] = a[idx] + b[idx];
  }
}

__global__ void elem_sub_kernel(size_t n, const float *a, const float *b,
                                float *out) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += blockDim.x * gridDim.x) {
    out[idx] = a[idx] - b[idx];
  }
}

__global__ void elem_mul_kernel(size_t n, const float *a, const float *b,
                                float *out) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += blockDim.x * gridDim.x) {
    out[idx] = a[idx] * b[idx];
  }
}

__global__ void elem_div_kernel(size_t n, const float *a, const float *b,
                                float *out) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += blockDim.x * gridDim.x) {
    out[idx] = a[idx] / b[idx];
  }
}

__global__ void elem_add_const_kernel(size_t n, const float *a, float val,
                                      float *out) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += blockDim.x * gridDim.x) {
    out[idx] = a[idx] + val;
  }
}

__global__ void elem_mul_const_kernel(size_t n, const float *a, float val,
                                      float *out) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += blockDim.x * gridDim.x) {
    out[idx] = a[idx] * val;
  }
}

__global__ void elem_sqrt_kernel(size_t n, const float *a, float *out) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += blockDim.x * gridDim.x) {
    out[idx] = sqrtf(a[idx]);
  }
}

__global__ void relu_kernel(size_t n, const float *input, float *output) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += blockDim.x * gridDim.x) {
    float v = input[idx];
    output[idx] = v > 0.0f ? v : 0.0f;
  }
}

__global__ void relu_grad_kernel(size_t n, const float *input,
                                 const float *in_grad, float *output) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += blockDim.x * gridDim.x) {
    output[idx] = (input[idx] > 0.0f) ? in_grad[idx] : 0.0f;
  }
}

__inline__ __device__ float gelu_forward_scalar(float x) {
  const float kSqrt2OverPi = 0.7978845608028654f;
  const float kCoeff = 0.044715f;
  float x3 = x * x * x;
  float inner = kSqrt2OverPi * (x + kCoeff * x3);
  return 0.5f * x * (1.0f + tanhf(inner));
}

__inline__ __device__ float gelu_backward_scalar(float x, float dy) {
  const float kSqrt2OverPi = 0.7978845608028654f;
  const float kCoeff = 0.044715f;
  float x2 = x * x;
  float x3 = x2 * x;
  float inner = kSqrt2OverPi * (x + kCoeff * x3);
  float tanh_inner = tanhf(inner);
  float sech2 = 1.0f - tanh_inner * tanh_inner;
  float d_inner = kSqrt2OverPi * (1.0f + 3.0f * kCoeff * x2);
  float dx = 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech2 * d_inner;
  return dy * dx;
}

__global__ void gelu_kernel(size_t n, const float *input, float *output) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += blockDim.x * gridDim.x) {
    output[idx] = gelu_forward_scalar(input[idx]);
  }
}

__global__ void gelu_grad_kernel(size_t n, const float *input,
                                 const float *in_grad, float *output) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += blockDim.x * gridDim.x) {
    output[idx] = gelu_backward_scalar(input[idx], in_grad[idx]);
  }
}

// ============================================================================
// Softmax & Cross-Entropy Kernels (Forward & Fused Backward)
// ============================================================================

__global__ void softmax_warp_kernel(int nrow, int ncol, const float *input,
                                    float *output) {
  __shared__ float shared[32];
  int row = blockIdx.x;
  if (row >= nrow)
    return;
  const float *row_in = input + static_cast<size_t>(row) * ncol;
  float *row_out = output + static_cast<size_t>(row) * ncol;

  float local_max = -FLT_MAX;
  for (int col = threadIdx.x; col < ncol; col += blockDim.x) {
    local_max = fmaxf(local_max, row_in[col]);
  }
  float row_max = block_reduce_max(local_max, shared);

  float local_sum = 0.0f;
  for (int col = threadIdx.x; col < ncol; col += blockDim.x) {
    local_sum += expf(row_in[col] - row_max);
  }
  float row_sum = block_reduce_sum(local_sum, shared);
  float inv_sum = 1.0f / row_sum;

  for (int col = threadIdx.x; col < ncol; col += blockDim.x) {
    row_out[col] = expf(row_in[col] - row_max) * inv_sum;
  }
}

__global__ void softmax_cross_entropy_row_kernel(int nrow, int ncol,
                                                 const float *logits,
                                                 const float *labels,
                                                 float *output) {
  __shared__ float shared[32];
  int row = blockIdx.x;
  if (row >= nrow)
    return;
  const float *row_logits = logits + static_cast<size_t>(row) * ncol;
  const float *row_labels = labels + static_cast<size_t>(row) * ncol;

  float local_max = -FLT_MAX;
  for (int col = threadIdx.x; col < ncol; col += blockDim.x) {
    local_max = fmaxf(local_max, row_logits[col]);
  }
  float row_max = block_reduce_max(local_max, shared);

  float local_sum = 0.0f;
  for (int col = threadIdx.x; col < ncol; col += blockDim.x) {
    local_sum += expf(row_logits[col] - row_max);
  }
  float row_sum = block_reduce_sum(local_sum, shared);
  float log_sum = logf(row_sum);

  float local_loss = 0.0f;
  for (int col = threadIdx.x; col < ncol; col += blockDim.x) {
    local_loss -= row_labels[col] * (row_logits[col] - row_max - log_sum);
  }
  float row_loss = block_reduce_sum(local_loss, shared);

  if (threadIdx.x == 0) {
    atomicAdd(output, row_loss / static_cast<float>(nrow));
  }
}

__global__ void softmax_cross_entropy_grad_kernel(int nrow, int ncol,
                                                  const float *logits,
                                                  const float *labels,
                                                  const float *grad_out,
                                                  float *output) {
  __shared__ float shared[32];
  int row = blockIdx.x;
  if (row >= nrow)
    return;
  const float *row_logits = logits + static_cast<size_t>(row) * ncol;
  const float *row_labels = labels + static_cast<size_t>(row) * ncol;
  float *row_dx = output + static_cast<size_t>(row) * ncol;

  float local_max = -FLT_MAX;
  for (int col = threadIdx.x; col < ncol; col += blockDim.x) {
    local_max = fmaxf(local_max, row_logits[col]);
  }
  float row_max = block_reduce_max(local_max, shared);

  float local_sum = 0.0f;
  for (int col = threadIdx.x; col < ncol; col += blockDim.x) {
    local_sum += expf(row_logits[col] - row_max);
  }
  float row_sum = block_reduce_sum(local_sum, shared);
  float inv_sum = 1.0f / row_sum;
  float scale = (grad_out != nullptr ? grad_out[0] : 1.0f);

  for (int col = threadIdx.x; col < ncol; col += blockDim.x) {
    float prob = expf(row_logits[col] - row_max) * inv_sum;
    row_dx[col] = (prob - row_labels[col]) * scale;
  }
}

// ============================================================================
// Matrix Multiplication & Fused Linear+Bias+Activation Kernels
// ============================================================================

__global__ void matmul_naive_kernel(int M, int N, int K, const float *A,
                                    bool transA, const float *B, bool transB,
                                    float *C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      float a_val = transA ? A[k * M + row] : A[row * K + k];
      float b_val = transB ? B[col * K + k] : B[k * N + col];
      sum += a_val * b_val;
    }
    C[row * N + col] = sum;
  }
}

// Tier-2 2D Shared-Memory Tiled SGEMM (TILE = 32, 1 output element per thread)
__global__ void matmul_shared_tiled_kernel(int M, int N, int K,
                                           const float *__restrict__ A,
                                           bool transA,
                                           const float *__restrict__ B,
                                           bool transB,
                                           float *__restrict__ C) {
  constexpr int TILE = 32;
  __shared__ float As[TILE][TILE + 1];
  __shared__ float Bs[TILE][TILE + 1];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = blockIdx.y * TILE + ty;
  int col = blockIdx.x * TILE + tx;

  float acc = 0.0f;
  for (int k_base = 0; k_base < K; k_base += TILE) {
    int a_col = k_base + tx;
    if (row < M && a_col < K) {
      As[ty][tx] = transA ? A[static_cast<size_t>(a_col) * M + row]
                          : A[static_cast<size_t>(row) * K + a_col];
    } else {
      As[ty][tx] = 0.0f;
    }

    int b_row = k_base + ty;
    if (b_row < K && col < N) {
      Bs[ty][tx] = transB ? B[static_cast<size_t>(col) * K + b_row]
                          : B[static_cast<size_t>(b_row) * N + col];
    } else {
      Bs[ty][tx] = 0.0f;
    }

    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
      acc += As[ty][k] * Bs[k][tx];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[static_cast<size_t>(row) * N + col] = acc;
  }
}

// High-performance 2D Shared-Memory + Register-Blocked SGEMM with Fused Epilogue
// Tile dimensions: BM = 64, BN = 64, BK = 16, TM = 4, TN = 4 (256 threads/block)
template <bool kHasBias, int kActType>
__global__ void sgemm_register_tiled_kernel(int M, int N, int K,
                                            const float *__restrict__ A,
                                            bool transA,
                                            const float *__restrict__ B,
                                            bool transB,
                                            const float *__restrict__ bias,
                                            float *__restrict__ C) {
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 16;
  constexpr int TM = 4;
  constexpr int TN = 4;

  __shared__ float As[BK][BM + 4];
  __shared__ float Bs[BK][BN + 4];

  const int tid = threadIdx.x;
  const int ty = tid / 16; // 0..15
  const int tx = tid % 16; // 0..15

  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;

  float accum[TM][TN] = {{0.0f}};

  for (int k_base = 0; k_base < K; k_base += BK) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      int load_idx = tid + i * 256;
      int a_row = load_idx / BK;
      int a_col = load_idx % BK;
      int g_row = block_row + a_row;
      int g_col = k_base + a_col;
      float val_a = 0.0f;
      if (g_row < M && g_col < K) {
        val_a = transA ? A[static_cast<size_t>(g_col) * M + g_row]
                       : A[static_cast<size_t>(g_row) * K + g_col];
      }
      As[a_col][a_row] = val_a;

      int b_row = load_idx / BN;
      int b_col = load_idx % BN;
      int gb_row = k_base + b_row;
      int gb_col = block_col + b_col;
      float val_b = 0.0f;
      if (gb_row < K && gb_col < N) {
        val_b = transB ? B[static_cast<size_t>(gb_col) * K + gb_row]
                       : B[static_cast<size_t>(gb_row) * N + gb_col];
      }
      Bs[b_row][b_col] = val_b;
    }

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < BK; ++k) {
      float regA[TM];
      float regB[TN];
      #pragma unroll
      for (int m = 0; m < TM; ++m) {
        regA[m] = As[k][ty * TM + m];
      }
      #pragma unroll
      for (int n = 0; n < TN; ++n) {
        regB[n] = Bs[k][tx * TN + n];
      }
      #pragma unroll
      for (int m = 0; m < TM; ++m) {
        #pragma unroll
        for (int n = 0; n < TN; ++n) {
          accum[m][n] += regA[m] * regB[n];
        }
      }
    }

    __syncthreads();
  }

  float bias_reg[TN];
  if (kHasBias && bias != nullptr) {
    #pragma unroll
    for (int n = 0; n < TN; ++n) {
      int g_col = block_col + tx * TN + n;
      bias_reg[n] = (g_col < N) ? bias[g_col] : 0.0f;
    }
  }

  #pragma unroll
  for (int m = 0; m < TM; ++m) {
    int g_row = block_row + ty * TM + m;
    if (g_row < M) {
      #pragma unroll
      for (int n = 0; n < TN; ++n) {
        int g_col = block_col + tx * TN + n;
        if (g_col < N) {
          float val = accum[m][n];
          if (kHasBias && bias != nullptr) {
            val += bias_reg[n];
          }
          if (kActType == 1) {
            val = val > 0.0f ? val : 0.0f;
          } else if (kActType == 2) {
            val = gelu_forward_scalar(val);
          }
          C[static_cast<size_t>(g_row) * N + g_col] = val;
        }
      }
    }
  }
}

// ============================================================================
// Fused LayerNorm & AdaLN Forward/Backward Kernels (Welford + Warp Shuffle)
// ============================================================================

__inline__ __device__ void block_welford_reduce(float &mean, float &m2,
                                                float &count, float *s_mean,
                                                float *s_m2, float *s_count) {
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  int num_warps = (blockDim.x + 31) >> 5;

  warp_reduce_welford(mean, m2, count);
  if (lane == 0) {
    s_mean[wid] = mean;
    s_m2[wid] = m2;
    s_count[wid] = count;
  }
  __syncthreads();

  if (wid == 0) {
    mean = (threadIdx.x < num_warps) ? s_mean[lane] : 0.0f;
    m2 = (threadIdx.x < num_warps) ? s_m2[lane] : 0.0f;
    count = (threadIdx.x < num_warps) ? s_count[lane] : 0.0f;
    warp_reduce_welford(mean, m2, count);
    if (lane == 0) {
      s_mean[0] = mean;
      s_m2[0] = m2;
      s_count[0] = count;
    }
  }
  __syncthreads();
  mean = s_mean[0];
  m2 = s_m2[0];
  count = s_count[0];
}

__global__ void layernorm_forward_kernel(int M, int N, const float *__restrict__ x,
                                         const float *__restrict__ gamma,
                                         const float *__restrict__ beta,
                                         float *__restrict__ y,
                                         float *__restrict__ mean_out,
                                         float *__restrict__ rstd_out,
                                         float eps) {
  __shared__ float s_mean[32];
  __shared__ float s_m2[32];
  __shared__ float s_count[32];

  int row = blockIdx.x;
  if (row >= M)
    return;
  const float *row_x = x + static_cast<size_t>(row) * N;
  float *row_y = y + static_cast<size_t>(row) * N;

  float mean = 0.0f;
  float m2 = 0.0f;
  float count = 0.0f;

  for (int col = threadIdx.x; col < N; col += blockDim.x) {
    float val = row_x[col];
    count += 1.0f;
    float delta = val - mean;
    mean += delta / count;
    float delta2 = val - mean;
    m2 += delta * delta2;
  }

  block_welford_reduce(mean, m2, count, s_mean, s_m2, s_count);
  float var = m2 / static_cast<float>(N);
  float rstd = rsqrtf(var + eps);

  if (threadIdx.x == 0) {
    if (mean_out != nullptr)
      mean_out[row] = mean;
    if (rstd_out != nullptr)
      rstd_out[row] = rstd;
  }

  for (int col = threadIdx.x; col < N; col += blockDim.x) {
    float x_hat = (row_x[col] - mean) * rstd;
    float g = (gamma != nullptr) ? gamma[col] : 1.0f;
    float b = (beta != nullptr) ? beta[col] : 0.0f;
    row_y[col] = x_hat * g + b;
  }
}

__global__ void layernorm_backward_dx_kernel(int M, int N,
                                             const float *__restrict__ x,
                                             const float *__restrict__ dy,
                                             const float *__restrict__ gamma,
                                             float *__restrict__ dx,
                                             float eps) {
  __shared__ float s_mean[32];
  __shared__ float s_m2[32];
  __shared__ float s_count[32];

  int row = blockIdx.x;
  if (row >= M)
    return;
  const float *row_x = x + static_cast<size_t>(row) * N;
  const float *row_dy = dy + static_cast<size_t>(row) * N;
  float *row_dx = dx + static_cast<size_t>(row) * N;

  float mean = 0.0f;
  float m2 = 0.0f;
  float count = 0.0f;
  for (int col = threadIdx.x; col < N; col += blockDim.x) {
    float val = row_x[col];
    count += 1.0f;
    float delta = val - mean;
    mean += delta / count;
    float delta2 = val - mean;
    m2 += delta * delta2;
  }
  block_welford_reduce(mean, m2, count, s_mean, s_m2, s_count);
  float rstd = rsqrtf((m2 / static_cast<float>(N)) + eps);

  float sum_dy_g = 0.0f;
  float sum_dy_g_xhat = 0.0f;
  for (int col = threadIdx.x; col < N; col += blockDim.x) {
    float x_hat = (row_x[col] - mean) * rstd;
    float g = (gamma != nullptr) ? gamma[col] : 1.0f;
    float dy_g = row_dy[col] * g;
    sum_dy_g += dy_g;
    sum_dy_g_xhat += dy_g * x_hat;
  }
  sum_dy_g = block_reduce_sum(sum_dy_g, s_mean);
  sum_dy_g_xhat = block_reduce_sum(sum_dy_g_xhat, s_m2);

  float inv_N = 1.0f / static_cast<float>(N);
  for (int col = threadIdx.x; col < N; col += blockDim.x) {
    float x_hat = (row_x[col] - mean) * rstd;
    float g = (gamma != nullptr) ? gamma[col] : 1.0f;
    float dy_g = row_dy[col] * g;
    row_dx[col] = rstd * (dy_g - inv_N * (sum_dy_g + x_hat * sum_dy_g_xhat));
  }
}

__global__ void layernorm_backward_dgamma_dbeta_kernel(
    int M, int N, const float *__restrict__ x, const float *__restrict__ dy,
    float *__restrict__ dgamma, float *__restrict__ dbeta, float eps) {
  __shared__ float s_mean[32];
  __shared__ float s_m2[32];
  __shared__ float s_count[32];

  int row = blockIdx.x;
  if (row >= M)
    return;
  const float *row_x = x + static_cast<size_t>(row) * N;
  const float *row_dy = dy + static_cast<size_t>(row) * N;

  float mean = 0.0f;
  float m2 = 0.0f;
  float count = 0.0f;
  for (int col = threadIdx.x; col < N; col += blockDim.x) {
    float val = row_x[col];
    count += 1.0f;
    float delta = val - mean;
    mean += delta / count;
    float delta2 = val - mean;
    m2 += delta * delta2;
  }
  block_welford_reduce(mean, m2, count, s_mean, s_m2, s_count);
  float rstd = rsqrtf((m2 / static_cast<float>(N)) + eps);

  for (int col = threadIdx.x; col < N; col += blockDim.x) {
    float x_hat = (row_x[col] - mean) * rstd;
    float dy_val = row_dy[col];
    if (dgamma != nullptr)
      atomicAdd(&dgamma[col], dy_val * x_hat);
    if (dbeta != nullptr)
      atomicAdd(&dbeta[col], dy_val);
  }
}

__global__ void adaln_forward_kernel(int M, int N, const float *__restrict__ x,
                                     const float *__restrict__ scale,
                                     const float *__restrict__ shift,
                                     bool per_row_mod, float *__restrict__ y,
                                     float eps) {
  __shared__ float s_mean[32];
  __shared__ float s_m2[32];
  __shared__ float s_count[32];

  int row = blockIdx.x;
  if (row >= M)
    return;
  const float *row_x = x + static_cast<size_t>(row) * N;
  const float *row_scale =
      per_row_mod ? (scale + static_cast<size_t>(row) * N) : scale;
  const float *row_shift =
      per_row_mod ? (shift + static_cast<size_t>(row) * N) : shift;
  float *row_y = y + static_cast<size_t>(row) * N;

  float mean = 0.0f;
  float m2 = 0.0f;
  float count = 0.0f;
  for (int col = threadIdx.x; col < N; col += blockDim.x) {
    float val = row_x[col];
    count += 1.0f;
    float delta = val - mean;
    mean += delta / count;
    float delta2 = val - mean;
    m2 += delta * delta2;
  }
  block_welford_reduce(mean, m2, count, s_mean, s_m2, s_count);
  float rstd = rsqrtf((m2 / static_cast<float>(N)) + eps);

  for (int col = threadIdx.x; col < N; col += blockDim.x) {
    float x_hat = (row_x[col] - mean) * rstd;
    row_y[col] = x_hat * (1.0f + row_scale[col]) + row_shift[col];
  }
}

__global__ void adaln_backward_row_kernel(int M, int N,
                                          const float *__restrict__ x,
                                          const float *__restrict__ dy,
                                          const float *__restrict__ scale,
                                          bool per_row_mod,
                                          float *__restrict__ dx,
                                          float *__restrict__ dscale,
                                          float *__restrict__ dshift,
                                          float eps) {
  __shared__ float s_mean[32];
  __shared__ float s_m2[32];
  __shared__ float s_count[32];

  int row = blockIdx.x;
  if (row >= M)
    return;
  const float *row_x = x + static_cast<size_t>(row) * N;
  const float *row_dy = dy + static_cast<size_t>(row) * N;
  const float *row_scale =
      per_row_mod ? (scale + static_cast<size_t>(row) * N) : scale;
  float *row_dx = dx + static_cast<size_t>(row) * N;

  float mean = 0.0f;
  float m2 = 0.0f;
  float count = 0.0f;
  for (int col = threadIdx.x; col < N; col += blockDim.x) {
    float val = row_x[col];
    count += 1.0f;
    float delta = val - mean;
    mean += delta / count;
    float delta2 = val - mean;
    m2 += delta * delta2;
  }
  block_welford_reduce(mean, m2, count, s_mean, s_m2, s_count);
  float rstd = rsqrtf((m2 / static_cast<float>(N)) + eps);

  float sum_dy_g = 0.0f;
  float sum_dy_g_xhat = 0.0f;
  for (int col = threadIdx.x; col < N; col += blockDim.x) {
    float x_hat = (row_x[col] - mean) * rstd;
    float g = 1.0f + row_scale[col];
    float dy_val = row_dy[col];
    float dy_g = dy_val * g;
    sum_dy_g += dy_g;
    sum_dy_g_xhat += dy_g * x_hat;
    if (per_row_mod) {
      if (dscale != nullptr)
        dscale[static_cast<size_t>(row) * N + col] = dy_val * x_hat;
      if (dshift != nullptr)
        dshift[static_cast<size_t>(row) * N + col] = dy_val;
    }
  }
  sum_dy_g = block_reduce_sum(sum_dy_g, s_mean);
  sum_dy_g_xhat = block_reduce_sum(sum_dy_g_xhat, s_m2);

  float inv_N = 1.0f / static_cast<float>(N);
  for (int col = threadIdx.x; col < N; col += blockDim.x) {
    float x_hat = (row_x[col] - mean) * rstd;
    float g = 1.0f + row_scale[col];
    float dy_g = row_dy[col] * g;
    row_dx[col] = rstd * (dy_g - inv_N * (sum_dy_g + x_hat * sum_dy_g_xhat));
  }
}

// ============================================================================
// C Runtime API Entrypoints
// ============================================================================

int DLGpuArraySet(DLArrayHandle arr, float value) {
  size_t n = GetNumElements(arr);
  float *arr_data = static_cast<float *>(arr->data);
  array_set_kernel<<<GetGridSize1D(n), 256>>>(n, arr_data, value);
  CHECK_GPU_ERR(cudaGetLastError());
  return 0;
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
  size_t in_size = GetNumElements(input);
  size_t out_size = GetNumElements(output);
  assert(out_size % in_size == 0);
  const float *in_data = static_cast<const float *>(input->data);
  float *out_data = static_cast<float *>(output->data);
  broadcast_to_kernel<<<GetGridSize1D(out_size), 256>>>(in_size, out_size,
                                                        in_data, out_data);
  CHECK_GPU_ERR(cudaGetLastError());
  return 0;
}

int DLGpuBroadcastScalar(const DLArrayHandle scalar_input,
                         DLArrayHandle output) {
  size_t out_size = GetNumElements(output);
  const float *in_data = static_cast<const float *>(scalar_input->data);
  float *out_data = static_cast<float *>(output->data);
  broadcast_scalar_kernel<<<GetGridSize1D(out_size), 256>>>(out_size, in_data,
                                                            out_data);
  CHECK_GPU_ERR(cudaGetLastError());
  return 0;
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  assert(input->ndim >= 2);
  size_t nrow = static_cast<size_t>(input->shape[0]);
  size_t ncol = GetNumElements(input) / nrow;
  assert(GetNumElements(output) == ncol);
  const float *in_data = static_cast<const float *>(input->data);
  float *out_data = static_cast<float *>(output->data);
  reduce_sum_axis_zero_kernel<<<GetGridSize1D(ncol), 256>>>(nrow, ncol, in_data,
                                                            out_data);
  CHECK_GPU_ERR(cudaGetLastError());
  return 0;
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
  size_t n = GetNumElements(output);
  assert(GetNumElements(matA) == n && GetNumElements(matB) == n);
  const float *a = static_cast<const float *>(matA->data);
  const float *b = static_cast<const float *>(matB->data);
  float *out = static_cast<float *>(output->data);
  elem_add_kernel<<<GetGridSize1D(n), 256>>>(n, a, b, out);
  CHECK_GPU_ERR(cudaGetLastError());
  return 0;
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
  size_t n = GetNumElements(output);
  assert(GetNumElements(input) == n);
  const float *a = static_cast<const float *>(input->data);
  float *out = static_cast<float *>(output->data);
  elem_add_const_kernel<<<GetGridSize1D(n), 256>>>(n, a, val, out);
  CHECK_GPU_ERR(cudaGetLastError());
  return 0;
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
  size_t n = GetNumElements(output);
  assert(GetNumElements(matA) == n && GetNumElements(matB) == n);
  const float *a = static_cast<const float *>(matA->data);
  const float *b = static_cast<const float *>(matB->data);
  float *out = static_cast<float *>(output->data);
  elem_mul_kernel<<<GetGridSize1D(n), 256>>>(n, a, b, out);
  CHECK_GPU_ERR(cudaGetLastError());
  return 0;
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
  size_t n = GetNumElements(output);
  assert(GetNumElements(input) == n);
  const float *a = static_cast<const float *>(input->data);
  float *out = static_cast<float *>(output->data);
  elem_mul_const_kernel<<<GetGridSize1D(n), 256>>>(n, a, val, out);
  CHECK_GPU_ERR(cudaGetLastError());
  return 0;
}

int DLGpuMatrixMultiplyNaive(const DLArrayHandle matA, bool transposeA,
                             const DLArrayHandle matB, bool transposeB,
                             DLArrayHandle matC) {
  assert(matA->ndim == 2 && matB->ndim == 2 && matC->ndim == 2);
  int M = static_cast<int>(matC->shape[0]);
  int N = static_cast<int>(matC->shape[1]);
  int K = static_cast<int>(transposeA ? matA->shape[0] : matA->shape[1]);
  int K_B = static_cast<int>(transposeB ? matB->shape[1] : matB->shape[0]);
  assert(K == K_B);

  const float *A = static_cast<const float *>(matA->data);
  const float *B = static_cast<const float *>(matB->data);
  float *C = static_cast<float *>(matC->data);

  dim3 threads(16, 16);
  dim3 blocks((N + 15) / 16, (M + 15) / 16);
  matmul_naive_kernel<<<blocks, threads>>>(M, N, K, A, transposeA, B,
                                           transposeB, C);
  CHECK_GPU_ERR(cudaGetLastError());
  return 0;
}

int DLGpuMatrixMultiplyShared(const DLArrayHandle matA, bool transposeA,
                              const DLArrayHandle matB, bool transposeB,
                              DLArrayHandle matC) {
  assert(matA->ndim == 2 && matB->ndim == 2 && matC->ndim == 2);
  int M = static_cast<int>(matC->shape[0]);
  int N = static_cast<int>(matC->shape[1]);
  int K = static_cast<int>(transposeA ? matA->shape[0] : matA->shape[1]);
  int K_B = static_cast<int>(transposeB ? matB->shape[1] : matB->shape[0]);
  assert(K == K_B);

  const float *A = static_cast<const float *>(matA->data);
  const float *B = static_cast<const float *>(matB->data);
  float *C = static_cast<float *>(matC->data);

  dim3 threads(32, 32);
  dim3 blocks((N + 31) / 32, (M + 31) / 32);
  matmul_shared_tiled_kernel<<<blocks, threads>>>(M, N, K, A, transposeA, B,
                                                  transposeB, C);
  CHECK_GPU_ERR(cudaGetLastError());
  return 0;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
  assert(matA->ndim == 2 && matB->ndim == 2 && matC->ndim == 2);
  int M = static_cast<int>(matC->shape[0]);
  int N = static_cast<int>(matC->shape[1]);
  int K = static_cast<int>(transposeA ? matA->shape[0] : matA->shape[1]);
  int K_B = static_cast<int>(transposeB ? matB->shape[1] : matB->shape[0]);
  assert(K == K_B);

  const float *A = static_cast<const float *>(matA->data);
  const float *B = static_cast<const float *>(matB->data);
  float *C = static_cast<float *>(matC->data);

  dim3 threads(256);
  dim3 blocks((N + 63) / 64, (M + 63) / 64);
  sgemm_register_tiled_kernel<false, 0><<<blocks, threads>>>(
      M, N, K, A, transposeA, B, transposeB, nullptr, C);
  CHECK_GPU_ERR(cudaGetLastError());
  return 0;
}

int DLGpuFusedLinear(const DLArrayHandle matA, const DLArrayHandle matB,
                     const DLArrayHandle bias, DLArrayHandle output,
                     int act_type) {
  assert(matA->ndim == 2 && matB->ndim == 2 && output->ndim == 2);
  int M = static_cast<int>(output->shape[0]);
  int N = static_cast<int>(output->shape[1]);
  int K = static_cast<int>(matA->shape[1]);
  assert(static_cast<int>(matB->shape[0]) == K);
  assert(static_cast<int>(matB->shape[1]) == N);

  const float *A = static_cast<const float *>(matA->data);
  const float *B = static_cast<const float *>(matB->data);
  const float *b_ptr =
      (bias != nullptr) ? static_cast<const float *>(bias->data) : nullptr;
  float *C = static_cast<float *>(output->data);

  dim3 threads(256);
  dim3 blocks((N + 63) / 64, (M + 63) / 64);
  if (b_ptr != nullptr) {
    if (act_type == 1) {
      sgemm_register_tiled_kernel<true, 1><<<blocks, threads>>>(
          M, N, K, A, false, B, false, b_ptr, C);
    } else if (act_type == 2) {
      sgemm_register_tiled_kernel<true, 2><<<blocks, threads>>>(
          M, N, K, A, false, B, false, b_ptr, C);
    } else {
      sgemm_register_tiled_kernel<true, 0><<<blocks, threads>>>(
          M, N, K, A, false, B, false, b_ptr, C);
    }
  } else {
    if (act_type == 1) {
      sgemm_register_tiled_kernel<false, 1><<<blocks, threads>>>(
          M, N, K, A, false, B, false, nullptr, C);
    } else if (act_type == 2) {
      sgemm_register_tiled_kernel<false, 2><<<blocks, threads>>>(
          M, N, K, A, false, B, false, nullptr, C);
    } else {
      sgemm_register_tiled_kernel<false, 0><<<blocks, threads>>>(
          M, N, K, A, false, B, false, nullptr, C);
    }
  }
  CHECK_GPU_ERR(cudaGetLastError());
  return 0;
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
  size_t n = GetNumElements(output);
  assert(GetNumElements(input) == n);
  const float *in = static_cast<const float *>(input->data);
  float *out = static_cast<float *>(output->data);
  relu_kernel<<<GetGridSize1D(n), 256>>>(n, in, out);
  CHECK_GPU_ERR(cudaGetLastError());
  return 0;
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
  size_t n = GetNumElements(output);
  assert(GetNumElements(input) == n && GetNumElements(in_grad) == n);
  const float *in = static_cast<const float *>(input->data);
  const float *dy = static_cast<const float *>(in_grad->data);
  float *out = static_cast<float *>(output->data);
  relu_grad_kernel<<<GetGridSize1D(n), 256>>>(n, in, dy, out);
  CHECK_GPU_ERR(cudaGetLastError());
  return 0;
}

int DLGpuGelu(const DLArrayHandle input, DLArrayHandle output) {
  size_t n = GetNumElements(output);
  assert(GetNumElements(input) == n);
  const float *in = static_cast<const float *>(input->data);
  float *out = static_cast<float *>(output->data);
  gelu_kernel<<<GetGridSize1D(n), 256>>>(n, in, out);
  CHECK_GPU_ERR(cudaGetLastError());
  return 0;
}

int DLGpuGeluGrad(const DLArrayHandle input, const DLArrayHandle in_grad,
                  DLArrayHandle output) {
  size_t n = GetNumElements(output);
  assert(GetNumElements(input) == n && GetNumElements(in_grad) == n);
  const float *in = static_cast<const float *>(input->data);
  const float *dy = static_cast<const float *>(in_grad->data);
  float *out = static_cast<float *>(output->data);
  gelu_grad_kernel<<<GetGridSize1D(n), 256>>>(n, in, dy, out);
  CHECK_GPU_ERR(cudaGetLastError());
  return 0;
}

int DLGpuLayerNorm(const DLArrayHandle input, const DLArrayHandle gamma,
                   const DLArrayHandle beta, DLArrayHandle output,
                   DLArrayHandle mean, DLArrayHandle rstd, float eps) {
  assert(input->ndim == 2 && output->ndim == 2);
  int M = static_cast<int>(input->shape[0]);
  int N = static_cast<int>(input->shape[1]);
  const float *x = static_cast<const float *>(input->data);
  const float *g =
      (gamma != nullptr) ? static_cast<const float *>(gamma->data) : nullptr;
  const float *b =
      (beta != nullptr) ? static_cast<const float *>(beta->data) : nullptr;
  float *y = static_cast<float *>(output->data);
  float *m_ptr = (mean != nullptr) ? static_cast<float *>(mean->data) : nullptr;
  float *r_ptr = (rstd != nullptr) ? static_cast<float *>(rstd->data) : nullptr;

  layernorm_forward_kernel<<<M, 256>>>(M, N, x, g, b, y, m_ptr, r_ptr, eps);
  CHECK_GPU_ERR(cudaGetLastError());
  return 0;
}

int DLGpuLayerNormGrad(const DLArrayHandle input, const DLArrayHandle in_grad,
                       const DLArrayHandle gamma, DLArrayHandle grad_x,
                       DLArrayHandle grad_gamma, DLArrayHandle grad_beta,
                       float eps) {
  assert(input->ndim == 2 && in_grad->ndim == 2);
  int M = static_cast<int>(input->shape[0]);
  int N = static_cast<int>(input->shape[1]);
  const float *x = static_cast<const float *>(input->data);
  const float *dy = static_cast<const float *>(in_grad->data);
  const float *g =
      (gamma != nullptr) ? static_cast<const float *>(gamma->data) : nullptr;

  if (grad_x != nullptr) {
    float *dx = static_cast<float *>(grad_x->data);
    layernorm_backward_dx_kernel<<<M, 256>>>(M, N, x, dy, g, dx, eps);
  }
  if (grad_gamma != nullptr || grad_beta != nullptr) {
    float *dg = (grad_gamma != nullptr) ? static_cast<float *>(grad_gamma->data)
                                        : nullptr;
    float *db = (grad_beta != nullptr) ? static_cast<float *>(grad_beta->data)
                                       : nullptr;
    if (dg != nullptr)
      array_set_kernel<<<GetGridSize1D(N), 256>>>(N, dg, 0.0f);
    if (db != nullptr)
      array_set_kernel<<<GetGridSize1D(N), 256>>>(N, db, 0.0f);
    layernorm_backward_dgamma_dbeta_kernel<<<M, 256>>>(
        M, N, x, dy, dg, db, eps);
  }
  CHECK_GPU_ERR(cudaGetLastError());
  return 0;
}

int DLGpuAdaLN(const DLArrayHandle input, const DLArrayHandle scale,
               const DLArrayHandle shift, DLArrayHandle output, float eps) {
  assert(input->ndim == 2 && output->ndim == 2);
  int M = static_cast<int>(input->shape[0]);
  int N = static_cast<int>(input->shape[1]);
  size_t mod_elems = GetNumElements(scale);
  bool per_row_mod = (mod_elems == static_cast<size_t>(M) * N);
  assert(per_row_mod || mod_elems == static_cast<size_t>(N));

  const float *x = static_cast<const float *>(input->data);
  const float *sc = static_cast<const float *>(scale->data);
  const float *sh = static_cast<const float *>(shift->data);
  float *y = static_cast<float *>(output->data);

  adaln_forward_kernel<<<M, 256>>>(M, N, x, sc, sh, per_row_mod, y, eps);
  CHECK_GPU_ERR(cudaGetLastError());
  return 0;
}

int DLGpuAdaLNGrad(const DLArrayHandle input, const DLArrayHandle in_grad,
                   const DLArrayHandle scale, DLArrayHandle grad_x,
                   DLArrayHandle grad_scale, DLArrayHandle grad_shift,
                   float eps) {
  assert(input->ndim == 2 && in_grad->ndim == 2);
  int M = static_cast<int>(input->shape[0]);
  int N = static_cast<int>(input->shape[1]);
  size_t mod_elems = GetNumElements(scale);
  bool per_row_mod = (mod_elems == static_cast<size_t>(M) * N);

  const float *x = static_cast<const float *>(input->data);
  const float *dy = static_cast<const float *>(in_grad->data);
  const float *sc = static_cast<const float *>(scale->data);
  float *dx = static_cast<float *>(grad_x->data);
  float *dsc =
      (grad_scale != nullptr) ? static_cast<float *>(grad_scale->data) : nullptr;
  float *dsh =
      (grad_shift != nullptr) ? static_cast<float *>(grad_shift->data) : nullptr;

  adaln_backward_row_kernel<<<M, 256>>>(M, N, x, dy, sc, per_row_mod, dx, dsc,
                                        dsh, eps);
  if (!per_row_mod && (dsc != nullptr || dsh != nullptr)) {
    if (dsc != nullptr)
      array_set_kernel<<<GetGridSize1D(N), 256>>>(N, dsc, 0.0f);
    if (dsh != nullptr)
      array_set_kernel<<<GetGridSize1D(N), 256>>>(N, dsh, 0.0f);
    layernorm_backward_dgamma_dbeta_kernel<<<M, 256>>>(
        M, N, x, dy, dsc, dsh, eps);
  }
  CHECK_GPU_ERR(cudaGetLastError());
  return 0;
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
  assert(input->ndim == 2 && output->ndim == 2);
  int nrow = static_cast<int>(input->shape[0]);
  int ncol = static_cast<int>(input->shape[1]);
  const float *in = static_cast<const float *>(input->data);
  float *out = static_cast<float *>(output->data);
  softmax_warp_kernel<<<nrow, 256>>>(nrow, ncol, in, out);
  CHECK_GPU_ERR(cudaGetLastError());
  return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
  assert(input_a->ndim == 2 && input_b->ndim == 2);
  assert(input_a->shape[0] == input_b->shape[0] &&
         input_a->shape[1] == input_b->shape[1]);
  int nrow = static_cast<int>(input_a->shape[0]);
  int ncol = static_cast<int>(input_a->shape[1]);
  const float *logits = static_cast<const float *>(input_a->data);
  const float *labels = static_cast<const float *>(input_b->data);
  float *out = static_cast<float *>(output->data);

  CHECK_GPU_ERR(cudaMemset(out, 0, sizeof(float)));
  softmax_cross_entropy_row_kernel<<<nrow, 256>>>(nrow, ncol, logits, labels,
                                                  out);
  CHECK_GPU_ERR(cudaGetLastError());
  return 0;
}

int DLGpuSoftmaxCrossEntropyGrad(const DLArrayHandle input_a,
                                 const DLArrayHandle input_b,
                                 const DLArrayHandle in_grad,
                                 DLArrayHandle output) {
  assert(input_a->ndim == 2 && input_b->ndim == 2 && output->ndim == 2);
  int nrow = static_cast<int>(input_a->shape[0]);
  int ncol = static_cast<int>(input_a->shape[1]);
  const float *logits = static_cast<const float *>(input_a->data);
  const float *labels = static_cast<const float *>(input_b->data);
  const float *grad_out = (in_grad != nullptr)
                              ? static_cast<const float *>(in_grad->data)
                              : nullptr;
  float *out = static_cast<float *>(output->data);

  softmax_cross_entropy_grad_kernel<<<nrow, 256>>>(nrow, ncol, logits, labels,
                                                   grad_out, out);
  CHECK_GPU_ERR(cudaGetLastError());
  return 0;
}

int DLGpuReduceMeanAxisOne(const DLArrayHandle input, DLArrayHandle output) {
  assert(input->ndim == 2);
  int nrow = static_cast<int>(input->shape[0]);
  int ncol = static_cast<int>(input->shape[1]);
  assert(GetNumElements(output) == static_cast<size_t>(nrow));
  const float *in = static_cast<const float *>(input->data);
  float *out = static_cast<float *>(output->data);
  reduce_mean_axis_one_kernel<<<nrow, 256>>>(nrow, ncol, in, out);
  CHECK_GPU_ERR(cudaGetLastError());
  return 0;
}

int DLGpuBroadcastAxisOne(const DLArrayHandle input, DLArrayHandle output) {
  assert(output->ndim == 2);
  int nrow = static_cast<int>(output->shape[0]);
  int ncol = static_cast<int>(output->shape[1]);
  assert(GetNumElements(input) == static_cast<size_t>(nrow));
  const float *in = static_cast<const float *>(input->data);
  float *out = static_cast<float *>(output->data);
  size_t total = static_cast<size_t>(nrow) * ncol;
  broadcast_axis_one_kernel<<<GetGridSize1D(total), 256>>>(nrow, ncol, in, out);
  CHECK_GPU_ERR(cudaGetLastError());
  return 0;
}

int DLGpuMatrixElementwiseSub(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
  size_t n = GetNumElements(output);
  assert(GetNumElements(matA) == n && GetNumElements(matB) == n);
  const float *a = static_cast<const float *>(matA->data);
  const float *b = static_cast<const float *>(matB->data);
  float *out = static_cast<float *>(output->data);
  elem_sub_kernel<<<GetGridSize1D(n), 256>>>(n, a, b, out);
  CHECK_GPU_ERR(cudaGetLastError());
  return 0;
}

int DLGpuMatrixElementwiseDiv(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
  size_t n = GetNumElements(output);
  assert(GetNumElements(matA) == n && GetNumElements(matB) == n);
  const float *a = static_cast<const float *>(matA->data);
  const float *b = static_cast<const float *>(matB->data);
  float *out = static_cast<float *>(output->data);
  elem_div_kernel<<<GetGridSize1D(n), 256>>>(n, a, b, out);
  CHECK_GPU_ERR(cudaGetLastError());
  return 0;
}

int DLGpuSqrt(const DLArrayHandle input, DLArrayHandle output) {
  size_t n = GetNumElements(output);
  assert(GetNumElements(input) == n);
  const float *a = static_cast<const float *>(input->data);
  float *out = static_cast<float *>(output->data);
  elem_sqrt_kernel<<<GetGridSize1D(n), 256>>>(n, a, out);
  CHECK_GPU_ERR(cudaGetLastError());
  return 0;
}









