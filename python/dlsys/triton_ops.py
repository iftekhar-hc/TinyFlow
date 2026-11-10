from __future__ import absolute_import

import torch
import triton
import triton.language as tl
from . import ndarray as _nd


def _as_torch_tensor(arr):
    """Wrap a TinyFlow GPU NDArray zero-copy as a PyTorch CUDA tensor."""
    if arr is None:
        return None
    if isinstance(arr, _nd.NDArray):
        return torch.as_tensor(arr, device="cuda")
    if isinstance(arr, torch.Tensor):
        return arr
    raise TypeError("Unsupported tensor type: %s" % type(arr))


@triton.jit
def _fused_linear_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    ACT_TYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]

    if ACT_TYPE == 1:
        acc = tl.maximum(acc, 0.0)
    elif ACT_TYPE == 2:
        x = acc
        inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
        acc = x * tl.sigmoid(2.0 * inner)

    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_fused_linear(matA, matB, bias, out_arr, act_type=0):
    a = _as_torch_tensor(matA)
    b = _as_torch_tensor(matB)
    c = _as_torch_tensor(out_arr)
    if int(act_type) == 0:
        bias_1d = _as_torch_tensor(bias) if bias is not None else None
        torch.nn.functional.linear(a, b.t(), bias_1d, out=c)
        return
    bias_t = _as_torch_tensor(bias) if bias is not None else c

    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _fused_linear_kernel[grid](
        a, b, bias_t, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        HAS_BIAS=(bias is not None),
        ACT_TYPE=int(act_type),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )


def triton_matmul(matA, matB, out_arr):
    """Block-tiled FP32 matrix multiplication C = A @ B in Triton."""
    triton_fused_linear(matA, matB, None, out_arr, act_type=0)



@triton.jit
def _layernorm_fwd_kernel(
    x_ptr, gamma_ptr, beta_ptr, y_ptr,
    stride_x_row, stride_y_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    x_hat = x_centered * rstd

    gamma = tl.load(gamma_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    beta = tl.load(beta_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x_hat * gamma + beta
    tl.store(y_ptr + row * stride_y_row + cols, y, mask=mask)


def triton_layernorm(in_arr, gamma, beta, out_arr, eps=1e-5):
    x = _as_torch_tensor(in_arr)
    g = _as_torch_tensor(gamma)
    b = _as_torch_tensor(beta)
    y = _as_torch_tensor(out_arr)
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    _layernorm_fwd_kernel[(M,)](
        x, g, b, y,
        x.stride(0), y.stride(0),
        N, float(eps),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4 if BLOCK_SIZE <= 1024 else 8,
    )


@triton.jit
def _adaln_fwd_kernel(
    x_ptr, scale_ptr, shift_ptr, y_ptr,
    stride_x_row, stride_mod_row, stride_y_row,
    N, eps,
    PER_ROW_MOD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    x_hat = x_centered * rstd

    mod_offset = row * stride_mod_row + cols if PER_ROW_MOD else cols
    scale = tl.load(scale_ptr + mod_offset, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_ptr + mod_offset, mask=mask, other=0.0).to(tl.float32)
    y = x_hat * (1.0 + scale) + shift
    tl.store(y_ptr + row * stride_y_row + cols, y, mask=mask)


def triton_adaln(in_arr, scale, shift, out_arr, eps=1e-5):
    x = _as_torch_tensor(in_arr)
    sc = _as_torch_tensor(scale)
    sh = _as_torch_tensor(shift)
    y = _as_torch_tensor(out_arr)
    M, N = x.shape
    per_row_mod = (sc.numel() == M * N)
    BLOCK_SIZE = triton.next_power_of_2(N)
    _adaln_fwd_kernel[(M,)](
        x, sc, sh, y,
        x.stride(0), N if per_row_mod else 0, y.stride(0),
        N, float(eps),
        PER_ROW_MOD=per_row_mod,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4 if BLOCK_SIZE <= 1024 else 8,
    )


@triton.jit
def _softmax_ce_grad_kernel(
    logits_ptr, labels_ptr, grad_out_ptr, dx_ptr,
    stride_row, M, N,
    HAS_GRAD_OUT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    logits = tl.load(logits_ptr + row * stride_row + cols, mask=mask, other=-float("inf")).to(tl.float32)
    labels = tl.load(labels_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    num = tl.exp(logits - row_max)
    denom = tl.sum(tl.where(mask, num, 0.0), axis=0)
    probs = num / denom

    grad_scale = tl.load(grad_out_ptr).to(tl.float32) if HAS_GRAD_OUT else 1.0
    dx = (probs - labels) * grad_scale
    tl.store(dx_ptr + row * stride_row + cols, dx, mask=mask)


def triton_softmax_ce_grad(in_arr_a, in_arr_b, in_grad_arr, out_arr):
    logits = _as_torch_tensor(in_arr_a)
    labels = _as_torch_tensor(in_arr_b)
    dx = _as_torch_tensor(out_arr)
    grad_out = _as_torch_tensor(in_grad_arr) if in_grad_arr is not None else dx
    M, N = logits.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    _softmax_ce_grad_kernel[(M,)](
        logits, labels, grad_out, dx,
        logits.stride(0), M, N,
        HAS_GRAD_OUT=(in_grad_arr is not None),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4 if BLOCK_SIZE <= 1024 else 8,
    )
