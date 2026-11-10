from __future__ import absolute_import

import ctypes
from ._base import _LIB, check_call
from . import ndarray as _nd


_KERNEL_LAUNCH_COUNT = 0


def get_kernel_launch_count():
    """Return total number of GPU kernel launches since last reset."""
    return _KERNEL_LAUNCH_COUNT


def reset_kernel_launch_count():
    """Reset the GPU kernel launch counter to zero."""
    global _KERNEL_LAUNCH_COUNT
    _KERNEL_LAUNCH_COUNT = 0


def inc_kernel_launch_count(n=1):
    """Increment the GPU kernel launch counter."""
    global _KERNEL_LAUNCH_COUNT
    _KERNEL_LAUNCH_COUNT += int(n)


def get_memory_stats(dev_id=0):
    """Query the C++ caching GPU allocator telemetry for the given device."""
    active = ctypes.c_size_t(0)
    peak = ctypes.c_size_t(0)
    reserved = ctypes.c_size_t(0)
    malloc_cnt = ctypes.c_size_t(0)
    check_call(_LIB.DLGpuGetMemoryStats(
        ctypes.c_int(dev_id),
        ctypes.byref(active),
        ctypes.byref(peak),
        ctypes.byref(reserved),
        ctypes.byref(malloc_cnt),
    ))
    return {
        "active_bytes": int(active.value),
        "peak_bytes": int(peak.value),
        "reserved_bytes": int(reserved.value),
        "malloc_count": int(malloc_cnt.value),
    }


def reset_peak_memory_stats(dev_id=0):
    """Reset peak active memory bytes in the C++ GPU allocator."""
    check_call(_LIB.DLGpuResetPeakMemoryStats(ctypes.c_int(dev_id)))


def empty_cache(dev_id=0):
    """Release all cached free blocks in the C++ GPU memory pool."""
    check_call(_LIB.DLGpuEmptyCache(ctypes.c_int(dev_id)))


def set_use_memory_pool(use_pool, dev_id=0):
    """Enable or disable the C++ caching memory pool (vs raw cudaMalloc/cudaFree)."""
    check_call(_LIB.DLGpuSetUseMemoryPool(
        ctypes.c_int(dev_id), ctypes.c_int(1 if use_pool else 0)))


def array_set(arr, value):
    assert isinstance(arr, _nd.NDArray)
    inc_kernel_launch_count(1)
    check_call(_LIB.DLGpuArraySet(arr.handle, ctypes.c_float(value)))


def broadcast_to(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    inc_kernel_launch_count(1)
    check_call(_LIB.DLGpuBroadcastTo(in_arr.handle, out_arr.handle))


def broadcast_scalar(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    inc_kernel_launch_count(1)
    check_call(_LIB.DLGpuBroadcastScalar(in_arr.handle, out_arr.handle))


def broadcast_axis_one(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    inc_kernel_launch_count(1)
    check_call(_LIB.DLGpuBroadcastAxisOne(in_arr.handle, out_arr.handle))


def reduce_sum_axis_zero(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    inc_kernel_launch_count(1)
    check_call(_LIB.DLGpuReduceSumAxisZero(in_arr.handle, out_arr.handle))


def reduce_mean_axis_one(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    inc_kernel_launch_count(1)
    check_call(_LIB.DLGpuReduceMeanAxisOne(in_arr.handle, out_arr.handle))


def matrix_elementwise_add(matA, matB, matC):
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(matC, _nd.NDArray)
    inc_kernel_launch_count(1)
    check_call(_LIB.DLGpuMatrixElementwiseAdd(matA.handle, matB.handle, matC.handle))


def matrix_elementwise_sub(matA, matB, matC):
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(matC, _nd.NDArray)
    inc_kernel_launch_count(1)
    check_call(_LIB.DLGpuMatrixElementwiseSub(matA.handle, matB.handle, matC.handle))


def matrix_elementwise_add_by_const(in_mat, val, out_mat):
    assert isinstance(in_mat, _nd.NDArray)
    assert isinstance(out_mat, _nd.NDArray)
    inc_kernel_launch_count(1)
    check_call(_LIB.DLGpuMatrixElementwiseAddByConst(
        in_mat.handle, ctypes.c_float(val), out_mat.handle))


def matrix_elementwise_multiply(matA, matB, matC):
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(matC, _nd.NDArray)
    inc_kernel_launch_count(1)
    check_call(_LIB.DLGpuMatrixElementwiseMultiply(
        matA.handle, matB.handle, matC.handle))


def matrix_elementwise_div(matA, matB, matC):
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(matC, _nd.NDArray)
    inc_kernel_launch_count(1)
    check_call(_LIB.DLGpuMatrixElementwiseDiv(
        matA.handle, matB.handle, matC.handle))


def matrix_elementwise_multiply_by_const(in_mat, val, out_mat):
    assert isinstance(in_mat, _nd.NDArray)
    assert isinstance(out_mat, _nd.NDArray)
    inc_kernel_launch_count(1)
    check_call(_LIB.DLGpuMatrixMultiplyByConst(
        in_mat.handle, ctypes.c_float(val), out_mat.handle))


def sqrt(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    inc_kernel_launch_count(1)
    check_call(_LIB.DLGpuSqrt(in_arr.handle, out_arr.handle))



def matrix_multiply_naive(matA, transA, matB, transB, matC):
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(matC, _nd.NDArray)
    inc_kernel_launch_count(1)
    check_call(_LIB.DLGpuMatrixMultiplyNaive(
        matA.handle, ctypes.c_bool(transA), matB.handle, ctypes.c_bool(transB), matC.handle))


def matrix_multiply_shared(matA, transA, matB, transB, matC):
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(matC, _nd.NDArray)
    inc_kernel_launch_count(1)
    check_call(_LIB.DLGpuMatrixMultiplyShared(
        matA.handle, ctypes.c_bool(transA), matB.handle, ctypes.c_bool(transB), matC.handle))


def matrix_multiply(matA, transA, matB, transB, matC):
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(matC, _nd.NDArray)
    inc_kernel_launch_count(1)
    check_call(_LIB.DLGpuMatrixMultiply(
        matA.handle, ctypes.c_bool(transA), matB.handle, ctypes.c_bool(transB), matC.handle))


def fused_linear(matA, matB, bias, out_arr, act_type=0, backend="cuda", transA=False, transB=False):
    if isinstance(matB, bool):
        # Support 7-arg signature: (matA, transA, matB, transB, bias, out_arr, act_type=...)
        transA_val = matB
        real_matB = bias
        transB_val = out_arr
        real_bias = act_type
        real_out = backend
        real_act = 0 if isinstance(transA, bool) else int(transA)
        matA, transA, matB, transB, bias, out_arr, act_type, backend = (
            matA, transA_val, real_matB, transB_val, real_bias, real_out, real_act, "cuda"
        )
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    if backend == "triton" and not transA and not transB:
        from . import triton_ops
        inc_kernel_launch_count(1)
        triton_ops.triton_fused_linear(matA, matB, bias, out_arr, act_type=act_type)
        return
    if not transA and not transB and matA.shape[0] * matA.shape[1] * matB.shape[1] <= 512 * 512 * 512:
        bias_handle = bias.handle if bias is not None else None
        inc_kernel_launch_count(1)
        check_call(_LIB.DLGpuFusedLinear(
            matA.handle, matB.handle, bias_handle, out_arr.handle, ctypes.c_int(int(act_type))))
        return
    inc_kernel_launch_count(1)
    check_call(_LIB.DLGpuMatrixMultiply(
        matA.handle, ctypes.c_bool(transA), matB.handle, ctypes.c_bool(transB), out_arr.handle))
    if bias is not None:
        bcast_tmp = _nd.empty(out_arr.shape, ctx=out_arr.ctx)
        check_call(_LIB.DLGpuBroadcastTo(bias.handle, bcast_tmp.handle))
        check_call(_LIB.DLGpuMatrixElementwiseAdd(out_arr.handle, bcast_tmp.handle, out_arr.handle))
    if int(act_type) == 1:
        check_call(_LIB.DLGpuRelu(out_arr.handle, out_arr.handle))
    elif int(act_type) == 2:
        check_call(_LIB.DLGpuGelu(out_arr.handle, out_arr.handle))


def relu(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    inc_kernel_launch_count(1)
    check_call(_LIB.DLGpuRelu(in_arr.handle, out_arr.handle))


def relu_gradient(in_arr, in_grad_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(in_grad_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    inc_kernel_launch_count(1)
    check_call(_LIB.DLGpuReluGradient(in_arr.handle, in_grad_arr.handle, out_arr.handle))


def gelu(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    inc_kernel_launch_count(1)
    check_call(_LIB.DLGpuGelu(in_arr.handle, out_arr.handle))


def gelu_gradient(in_arr, in_grad_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(in_grad_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    inc_kernel_launch_count(1)
    check_call(_LIB.DLGpuGeluGrad(in_arr.handle, in_grad_arr.handle, out_arr.handle))



def layernorm(in_arr, gamma, beta, out_arr, mean_arr=None, rstd_arr=None, eps=1e-5, backend="cuda"):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    if backend == "triton":
        from . import triton_ops
        inc_kernel_launch_count(1)
        triton_ops.triton_layernorm(in_arr, gamma, beta, out_arr, eps=eps)
        return
    g_handle = gamma.handle if gamma is not None else None
    b_handle = beta.handle if beta is not None else None
    m_handle = mean_arr.handle if mean_arr is not None else None
    r_handle = rstd_arr.handle if rstd_arr is not None else None
    inc_kernel_launch_count(1)
    check_call(_LIB.DLGpuLayerNorm(
        in_arr.handle, g_handle, b_handle, out_arr.handle,
        m_handle, r_handle, ctypes.c_float(eps)))


def layernorm_gradient(in_grad_arr, in_arr, gamma, grad_x, grad_gamma=None, grad_beta=None, eps=1e-5):
    assert isinstance(in_grad_arr, _nd.NDArray)
    assert isinstance(in_arr, _nd.NDArray)
    g_handle = gamma.handle if gamma is not None else None
    dx_handle = grad_x.handle if grad_x is not None else None
    dg_handle = grad_gamma.handle if grad_gamma is not None else None
    db_handle = grad_beta.handle if grad_beta is not None else None
    launches = (1 if dx_handle is not None else 0) + (1 if (dg_handle is not None or db_handle is not None) else 0)
    inc_kernel_launch_count(max(launches, 1))
    check_call(_LIB.DLGpuLayerNormGrad(
        in_arr.handle, in_grad_arr.handle, g_handle,
        dx_handle, dg_handle, db_handle, ctypes.c_float(eps)))


def adaln(in_arr, scale, shift, out_arr, eps=1e-5, backend="cuda"):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(scale, _nd.NDArray)
    assert isinstance(shift, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    if backend == "triton":
        from . import triton_ops
        inc_kernel_launch_count(1)
        triton_ops.triton_adaln(in_arr, scale, shift, out_arr, eps=eps)
        return
    inc_kernel_launch_count(1)
    check_call(_LIB.DLGpuAdaLN(
        in_arr.handle, scale.handle, shift.handle, out_arr.handle, ctypes.c_float(eps)))


def adaln_gradient(in_grad_arr, in_arr, scale, grad_x, grad_scale=None, grad_shift=None, eps=1e-5):
    assert isinstance(in_grad_arr, _nd.NDArray)
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(scale, _nd.NDArray)
    assert isinstance(grad_x, _nd.NDArray)
    dsc_handle = grad_scale.handle if grad_scale is not None else None
    dsh_handle = grad_shift.handle if grad_shift is not None else None
    per_row = (scale.size == in_arr.size)
    launches = 1 if per_row or (dsc_handle is None and dsh_handle is None) else 2
    inc_kernel_launch_count(launches)
    check_call(_LIB.DLGpuAdaLNGrad(
        in_arr.handle, in_grad_arr.handle, scale.handle,
        grad_x.handle, dsc_handle, dsh_handle, ctypes.c_float(eps)))


def softmax(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    inc_kernel_launch_count(1)
    check_call(_LIB.DLGpuSoftmax(in_arr.handle, out_arr.handle))


def softmax_cross_entropy(in_arr_a, in_arr_b, out_arr):
    assert isinstance(in_arr_a, _nd.NDArray)
    assert isinstance(in_arr_b, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    inc_kernel_launch_count(1)
    check_call(_LIB.DLGpuSoftmaxCrossEntropy(
        in_arr_a.handle, in_arr_b.handle, out_arr.handle))


def softmax_cross_entropy_gradient(in_arr_a, in_arr_b, in_grad_arr, out_arr, backend="cuda"):
    assert isinstance(in_arr_a, _nd.NDArray)
    assert isinstance(in_arr_b, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    if backend == "triton":
        from . import triton_ops
        inc_kernel_launch_count(1)
        triton_ops.triton_softmax_ce_grad(in_arr_a, in_arr_b, in_grad_arr, out_arr)
        return
    grad_handle = in_grad_arr.handle if in_grad_arr is not None else None
    inc_kernel_launch_count(1)
    check_call(_LIB.DLGpuSoftmaxCrossEntropyGrad(
        in_arr_a.handle, in_arr_b.handle, grad_handle, out_arr.handle))


softmax_cross_entropy_grad = softmax_cross_entropy_gradient

