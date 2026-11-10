from __future__ import absolute_import

import ctypes
from ._base import _LIB, check_call
from . import ndarray as _nd


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
    _LIB.DLGpuArraySet(arr.handle, ctypes.c_float(value))


def broadcast_to(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuBroadcastTo(in_arr.handle, out_arr.handle)


def reduce_sum_axis_zero(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuReduceSumAxisZero(in_arr.handle, out_arr.handle)


def matrix_elementwise_add(matA, matB, matC):
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(matC, _nd.NDArray)
    _LIB.DLGpuMatrixElementwiseAdd(matA.handle, matB.handle, matC.handle)


def matrix_elementwise_add_by_const(in_mat, val, out_mat):
    assert isinstance(in_mat, _nd.NDArray)
    assert isinstance(out_mat, _nd.NDArray)
    _LIB.DLGpuMatrixElementwiseAddByConst(
        in_mat.handle, ctypes.c_float(val), out_mat.handle)


def matrix_elementwise_multiply(matA, matB, matC):
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(matC, _nd.NDArray)
    _LIB.DLGpuMatrixElementwiseMultiply(
        matA.handle, matB.handle, matC.handle)


def matrix_elementwise_multiply_by_const(in_mat, val, out_mat):
    assert isinstance(in_mat, _nd.NDArray)
    assert isinstance(out_mat, _nd.NDArray)
    _LIB.DLGpuMatrixMultiplyByConst(
        in_mat.handle, ctypes.c_float(val), out_mat.handle)


def matrix_multiply(matA, transA, matB, transB, matC):
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(matC, _nd.NDArray)
    _LIB.DLGpuMatrixMultiply(
        matA.handle, transA, matB.handle, transB, matC.handle)


def relu(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuRelu(in_arr.handle, out_arr.handle)


def relu_gradient(in_arr, in_grad_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(in_grad_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuReluGradient(in_arr.handle, in_grad_arr.handle, out_arr.handle)


def softmax(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuSoftmax(in_arr.handle, out_arr.handle)


def softmax_cross_entropy(in_arr_a, in_arr_b, out_arr):
    assert isinstance(in_arr_a, _nd.NDArray)
    assert isinstance(in_arr_b, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuSoftmaxCrossEntropy(
        in_arr_a.handle, in_arr_b.handle, out_arr.handle)
