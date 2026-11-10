/*!
 *  Copyright (c) 2017 by Contributors
 * \file c_runtime_api.h
 * \brief DL runtime library.
 *
 */

#ifndef DLSYS_RUNTIME_C_RUNTIME_API_H_
#define DLSYS_RUNTIME_C_RUNTIME_API_H_

#ifdef __cplusplus
#define DLSYS_EXTERN_C extern "C"
#else
#define DLSYS_EXTERN_C
#endif

#include "dlarray.h"
#include <stddef.h>
#include <stdint.h>

DLSYS_EXTERN_C {
  /*! \brief type of array index. */
  typedef int64_t index_t;

  /*! \brief the array handle */
  typedef DLArray *DLArrayHandle;
  /*!
   * \brief The stream that is specific to device
   * can be NULL, which indicates the default one.
   */
  typedef void *DLStreamHandle;

  // Array related apis for quick proptying
  /*!
   * \brief Allocate a nd-array's memory,
   *  including space of shape, of given spec.
   *
   * \param shape The shape of the array, the data content will be copied to out
   * \param ndim The number of dimension of the array.
   * \param ctx The ctx this array sits on.
   * \param out The output handle.
   * \return 0 when success, -1 when failure happens
   */
  int DLArrayAlloc(const index_t *shape, index_t ndim, DLContext ctx,
                   DLArrayHandle *out);

  /*!
   * \brief Free the DL Array.
   * \param handle The array handle to be freed.
   * \return 0 when success, -1 when failure happens
   */
  int DLArrayFree(DLArrayHandle handle);

  /*!
   * \brief Copy the array, both from and to must be valid during the copy.
   * \param from The array to be copied from.
   * \param to The target space.
   * \param stream The stream where the copy happens, can be NULL.
   * \return 0 when success, -1 when failure happens
   */
  int DLArrayCopyFromTo(DLArrayHandle from, DLArrayHandle to,
                        DLStreamHandle stream);

  /*!
   * \brief Set all array elements to given value.
   * \param arr The array to be Set.
   * \param value The target value.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuArraySet(DLArrayHandle arr, float value);

  /*!
   * \brief Broadcast input array to output array.
   * \param input The input array.
   * \param output The output array.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output);

  /*!
   * \brief Reduce sum input array by axis=0 and store to output.
   * \param input The input array.
   * \param output The output array.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output);

  /*!
   * \brief Elementwise add two matrices and store to output.
   * \param matA The left input array.
   * \param matB The right input array.
   * \param output The output array.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                                const DLArrayHandle matB, DLArrayHandle output);

  /*!
   * \brief Add matrix by const and store to output.
   * \param input The input array.
   * \param val The constant.
   * \param output The output array.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                       DLArrayHandle output);

  /*!
   * \brief Elementwise multiply two matrices and store to output.
   * \param matA The left input array.
   * \param matB The right input array.
   * \param output The output array.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuMatrixElementwiseMultiply(
      const DLArrayHandle matA, const DLArrayHandle matB, DLArrayHandle output);

  /*!
   * \brief Multiply matrix by const and store to output.
   * \param input The input array.
   * \param val The constant.
   * \param output The output array.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                                 DLArrayHandle output);

  /*!
   * \brief Matrix multiply two matrices and store to output.
   * \param matA The left input array.
   * \param transposeA Whether matA needs to be transposed
   * \param matB The right input array.
   * \param transposeB Whether matB needs to be transposed
   * \param output The output array.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                          const DLArrayHandle matB, bool transposeB,
                          DLArrayHandle matC);

  /*!
   * \brief Compute relu on all array elements, and store to output.
   * \param input The input array.
   * \param output The output value.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output);

  /*!
   * \brief Compute relu gradient, and store to output.
   * \param input The input array.
   * \param in_grad The input gradients value.
   * \param output The output array.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                        DLArrayHandle output);

  /*!
   * \brief Compute softmax on matrix, and store to output.
   * \param input The input array.
   * \param output The output value.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output);

  /*!
   * \brief Compute softmax_cross_entropy.
   *  np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
   * \param input_a The y array.
   * \param input_b The y_ array.
   * \param output The output value.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                               const DLArrayHandle input_b,
                               DLArrayHandle output);

  /*!
   * \brief Naive matrix multiply (baseline for benchmarking).
   */
  int DLGpuMatrixMultiplyNaive(const DLArrayHandle matA, bool transposeA,
                               const DLArrayHandle matB, bool transposeB,
                               DLArrayHandle matC);

  /*!
   * \brief Shared-memory + register-tiled matrix multiply.
   */
  int DLGpuMatrixMultiplyShared(const DLArrayHandle matA, bool transposeA,
                                const DLArrayHandle matB, bool transposeB,
                                DLArrayHandle matC);

  /*!
   * \brief Fused Linear + Bias + Activation: Y = Act(X @ W + b)
   * \param input (M, K) input matrix X
   * \param weight (K, N) weight matrix W
   * \param bias (N,) or (1, N) bias vector b (can be NULL if no bias)
   * \param output (M, N) output matrix Y
   * \param act_type 0 = Identity, 1 = ReLU, 2 = GELU
   */
  int DLGpuFusedLinear(const DLArrayHandle input, const DLArrayHandle weight,
                       const DLArrayHandle bias, DLArrayHandle output,
                       int act_type);

  /*!
   * \brief Fused LayerNorm forward pass:
   *  y = gamma * (x - mean) / sqrt(var + eps) + beta
   * \param input (M, N) input tensor
   * \param gamma (N,) scale parameter
   * \param beta (N,) shift parameter
   * \param output (M, N) normalized output
   * \param mean optional (M,) per-row mean buffer (can be NULL)
   * \param rstd optional (M,) per-row reciprocal std buffer (can be NULL)
   * \param eps epsilon for numerical stability
   */
  int DLGpuLayerNorm(const DLArrayHandle input, const DLArrayHandle gamma,
                     const DLArrayHandle beta, DLArrayHandle output,
                     DLArrayHandle mean, DLArrayHandle rstd, float eps);

  /*!
   * \brief Fused LayerNorm backward pass:
   *  Computes dx (M, N), dgamma (N,), dbeta (N,)
   */
  int DLGpuLayerNormGrad(const DLArrayHandle input, const DLArrayHandle in_grad,
                         const DLArrayHandle gamma, DLArrayHandle grad_x,
                         DLArrayHandle grad_gamma, DLArrayHandle grad_beta,
                         float eps);

  /*!
   * \brief Fused Adaptive LayerNorm (AdaLN) forward pass:
   *  y = (1 + scale) * (x - mean) / sqrt(var + eps) + shift
   * \param input (M, N) input tensor
   * \param scale (M, N) or (N,) adaptive scale tensor
   * \param shift (M, N) or (N,) adaptive shift tensor
   * \param output (M, N) modulated output
   * \param eps epsilon for numerical stability
   */
  int DLGpuAdaLN(const DLArrayHandle input, const DLArrayHandle scale,
                 const DLArrayHandle shift, DLArrayHandle output, float eps);

  /*!
   * \brief Fused Adaptive LayerNorm (AdaLN) backward pass:
   *  Computes dx (M, N), dscale (M, N) or (N,), dshift (M, N) or (N,)
   */
  int DLGpuAdaLNGrad(const DLArrayHandle input, const DLArrayHandle in_grad,
                     const DLArrayHandle scale, DLArrayHandle grad_x,
                     DLArrayHandle grad_scale, DLArrayHandle grad_shift,
                     float eps);

  /*!
   * \brief GELU activation forward pass (tanh approximation).
   */
  int DLGpuGelu(const DLArrayHandle input, DLArrayHandle output);

  /*!
   * \brief GELU activation backward pass.
   */
  int DLGpuGeluGrad(const DLArrayHandle input, const DLArrayHandle in_grad,
                    DLArrayHandle output);

  /*!
   * \brief Fused Softmax Cross-Entropy backward pass:
   *  dx = (softmax(logits) - labels) * (grad_out[0] / nrow)
   */
  int DLGpuSoftmaxCrossEntropyGrad(const DLArrayHandle logits,
                                   const DLArrayHandle labels,
                                   const DLArrayHandle grad_out,
                                   DLArrayHandle output);

  /*!
   * \brief Broadcast a (1,) scalar GPU tensor to output tensor on device
   *  without host-device synchronization.
   */
  int DLGpuBroadcastScalar(const DLArrayHandle scalar, DLArrayHandle output);

  /*!
   * \brief Unfused LayerNorm components for 5-kernel baseline comparison:
   *  Row-wise mean/variance reduction, normalize, affine scale/shift.
   */
  int DLGpuReduceMeanAxisOne(const DLArrayHandle input, DLArrayHandle output);
  int DLGpuBroadcastAxisOne(const DLArrayHandle input, DLArrayHandle output);
  int DLGpuMatrixElementwiseSub(const DLArrayHandle matA,
                                const DLArrayHandle matB, DLArrayHandle output);
  int DLGpuMatrixElementwiseDiv(const DLArrayHandle matA,
                                const DLArrayHandle matB, DLArrayHandle output);
  int DLGpuSqrt(const DLArrayHandle input, DLArrayHandle output);

  /*!
   * \brief CUDA Memory Pool & Telemetry APIs
   */
  int DLGpuGetMemoryStats(size_t *active_bytes, size_t *peak_bytes,
                          size_t *reserved_bytes, size_t *malloc_count);
  int DLGpuResetPeakMemoryStats();
  int DLGpuEmptyCache();
  int DLGpuSetUseMemoryPool(bool use_pool);
} // DLSYS_EXTERN_C

#endif // DLSYS_RUNTIME_C_RUNTIME_API_H_
