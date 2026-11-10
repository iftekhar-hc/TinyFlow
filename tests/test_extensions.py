"""Comprehensive verification tests for TinyFlow CUDA/Triton kernels, autodiff,
fusion passes, min-peak-memory scheduling, memory planning, and caching allocator.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import numpy as np
import torch
import torch.nn.functional as F

from dlsys import ndarray, gpu_op, autodiff as ad, triton_ops, fusion, scheduler


def test_sgemm_tiers_and_triton():
    ctx = ndarray.gpu(0)
    rng = np.random.default_rng(42)
    for m, k, n in [(64, 96, 80), (127, 253, 191), (256, 512, 256)]:
        a = rng.standard_normal((m, k), dtype=np.float32)
        b = rng.standard_normal((k, n), dtype=np.float32)
        ref = a @ b

        arr_a = ndarray.array(a, ctx=ctx)
        arr_b = ndarray.array(b, ctx=ctx)
        out_naive = ndarray.empty((m, n), ctx=ctx)
        out_shared = ndarray.empty((m, n), ctx=ctx)
        out_reg = ndarray.empty((m, n), ctx=ctx)
        out_triton = ndarray.empty((m, n), ctx=ctx)

        gpu_op.matrix_multiply_naive(arr_a, False, arr_b, False, out_naive)
        gpu_op.matrix_multiply_shared(arr_a, False, arr_b, False, out_shared)
        gpu_op.matrix_multiply(arr_a, False, arr_b, False, out_reg)
        triton_ops.triton_matmul(arr_a, arr_b, out_triton)

        np.testing.assert_allclose(out_naive.asnumpy(), ref, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(out_shared.asnumpy(), ref, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(out_reg.asnumpy(), ref, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(out_triton.asnumpy(), ref, rtol=1e-3, atol=1e-3)


def test_fused_linear_cuda_and_triton():
    ctx = ndarray.gpu(0)
    rng = np.random.default_rng(7)
    m, k, n = 128, 256, 192
    x = rng.standard_normal((m, k), dtype=np.float32)
    w = rng.standard_normal((k, n), dtype=np.float32)
    b = rng.standard_normal((n,), dtype=np.float32)

    tx = torch.tensor(x)
    tw = torch.tensor(w)
    tb = torch.tensor(b)
    z_ref = tx @ tw + tb
    refs = {
        0: z_ref.numpy(),
        1: F.relu(z_ref).numpy(),
        2: F.gelu(z_ref, approximate="tanh").numpy(),
    }

    arr_x = ndarray.array(x, ctx=ctx)
    arr_w = ndarray.array(w, ctx=ctx)
    arr_b = ndarray.array(b, ctx=ctx)

    for act_type, ref in refs.items():
        out_cuda = ndarray.empty((m, n), ctx=ctx)
        out_triton = ndarray.empty((m, n), ctx=ctx)
        gpu_op.fused_linear(arr_x, arr_w, arr_b, out_cuda, act_type=act_type)
        triton_ops.triton_fused_linear(arr_x, arr_w, arr_b, out_triton, act_type=act_type)
        np.testing.assert_allclose(out_cuda.asnumpy(), ref, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(out_triton.asnumpy(), ref, rtol=1e-3, atol=1e-3)


def test_layernorm_and_backward_vs_pytorch():
    ctx = ndarray.gpu(0)
    rng = np.random.default_rng(13)
    m, n = 64, 384
    x_np = rng.standard_normal((m, n), dtype=np.float32)
    g_np = rng.standard_normal((n,), dtype=np.float32)
    b_np = rng.standard_normal((n,), dtype=np.float32)
    dy_np = rng.standard_normal((m, n), dtype=np.float32)

    tx = torch.tensor(x_np, requires_grad=True)
    tg = torch.tensor(g_np, requires_grad=True)
    tb = torch.tensor(b_np, requires_grad=True)
    ty = F.layer_norm(tx, (n,), weight=tg, bias=tb, eps=1e-5)
    ty.backward(torch.tensor(dy_np))

    arr_x = ndarray.array(x_np, ctx=ctx)
    arr_g = ndarray.array(g_np, ctx=ctx)
    arr_b = ndarray.array(b_np, ctx=ctx)
    arr_dy = ndarray.array(dy_np, ctx=ctx)
    out_cuda = ndarray.empty((m, n), ctx=ctx)
    out_triton = ndarray.empty((m, n), ctx=ctx)

    gpu_op.layernorm(arr_x, arr_g, arr_b, out_cuda, eps=1e-5)
    triton_ops.triton_layernorm(arr_x, arr_g, arr_b, out_triton, eps=1e-5)
    np.testing.assert_allclose(out_cuda.asnumpy(), ty.detach().numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out_triton.asnumpy(), ty.detach().numpy(), rtol=1e-4, atol=1e-4)

    dx = ndarray.empty((m, n), ctx=ctx)
    dg = ndarray.empty((n,), ctx=ctx)
    db = ndarray.empty((n,), ctx=ctx)
    gpu_op.layernorm_gradient(arr_dy, arr_x, arr_g, dx, dg, db, eps=1e-5)
    np.testing.assert_allclose(dx.asnumpy(), tx.grad.numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(dg.asnumpy(), tg.grad.numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(db.asnumpy(), tb.grad.numpy(), rtol=1e-4, atol=1e-4)



def test_adaln_and_backward_vs_pytorch():
    ctx = ndarray.gpu(0)
    rng = np.random.default_rng(19)
    m, n = 64, 256
    x_np = rng.standard_normal((m, n), dtype=np.float32)
    s_np = rng.standard_normal((m, n), dtype=np.float32)
    b_np = rng.standard_normal((m, n), dtype=np.float32)
    dy_np = rng.standard_normal((m, n), dtype=np.float32)

    tx = torch.tensor(x_np, requires_grad=True)
    ts = torch.tensor(s_np, requires_grad=True)
    tb = torch.tensor(b_np, requires_grad=True)
    tx_hat = F.layer_norm(tx, (n,), eps=1e-5)
    ty = tx_hat * (1.0 + ts) + tb
    ty.backward(torch.tensor(dy_np))

    arr_x = ndarray.array(x_np, ctx=ctx)
    arr_s = ndarray.array(s_np, ctx=ctx)
    arr_b = ndarray.array(b_np, ctx=ctx)
    arr_dy = ndarray.array(dy_np, ctx=ctx)
    out_cuda = ndarray.empty((m, n), ctx=ctx)
    out_triton = ndarray.empty((m, n), ctx=ctx)

    gpu_op.adaln(arr_x, arr_s, arr_b, out_cuda, eps=1e-5)
    triton_ops.triton_adaln(arr_x, arr_s, arr_b, out_triton, eps=1e-5)
    np.testing.assert_allclose(out_cuda.asnumpy(), ty.detach().numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out_triton.asnumpy(), ty.detach().numpy(), rtol=1e-4, atol=1e-4)

    dx = ndarray.empty((m, n), ctx=ctx)
    ds = ndarray.empty((m, n), ctx=ctx)
    db = ndarray.empty((m, n), ctx=ctx)
    gpu_op.adaln_gradient(arr_dy, arr_x, arr_s, dx, ds, db, eps=1e-5)
    np.testing.assert_allclose(dx.asnumpy(), tx.grad.numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ds.asnumpy(), ts.grad.numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(db.asnumpy(), tb.grad.numpy(), rtol=1e-4, atol=1e-4)


def test_gelu_and_softmax_ce_grad():
    ctx = ndarray.gpu(0)
    rng = np.random.default_rng(23)
    m, n = 128, 256
    x_np = rng.standard_normal((m, n), dtype=np.float32)
    dy_np = rng.standard_normal((m, n), dtype=np.float32)

    tx = torch.tensor(x_np, requires_grad=True)
    ty = F.gelu(tx, approximate="tanh")
    ty.backward(torch.tensor(dy_np))

    arr_x = ndarray.array(x_np, ctx=ctx)
    arr_dy = ndarray.array(dy_np, ctx=ctx)
    arr_y = ndarray.empty((m, n), ctx=ctx)
    arr_dx = ndarray.empty((m, n), ctx=ctx)
    gpu_op.gelu(arr_x, arr_y)
    gpu_op.gelu_gradient(arr_x, arr_dy, arr_dx)
    np.testing.assert_allclose(arr_y.asnumpy(), ty.detach().numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(arr_dx.asnumpy(), tx.grad.numpy(), rtol=1e-4, atol=1e-4)

    labels_np = rng.uniform(0, 1, size=(m, n)).astype(np.float32)
    labels_np /= np.sum(labels_np, axis=1, keepdims=True)
    out_grad_np = np.array([1.7], dtype=np.float32)
    ref_grad = (ad.softmax_func(x_np) - labels_np) * out_grad_np[0]

    arr_labels = ndarray.array(labels_np, ctx=ctx)
    arr_og = ndarray.array(out_grad_np, ctx=ctx)
    grad_cuda = ndarray.empty((m, n), ctx=ctx)
    grad_triton = ndarray.empty((m, n), ctx=ctx)
    gpu_op.softmax_cross_entropy_grad(arr_x, arr_labels, arr_og, grad_cuda)
    triton_ops.triton_softmax_ce_grad(arr_x, arr_labels, arr_og, grad_triton)
    np.testing.assert_allclose(grad_cuda.asnumpy(), ref_grad, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(grad_triton.asnumpy(), ref_grad, rtol=1e-4, atol=1e-4)


def test_fusion_passes_and_end_to_end_autodiff():
    ctx = ndarray.gpu(0)
    rng = np.random.default_rng(101)
    batch, in_dim, hid_dim, num_classes = 32, 64, 96, 10

    x = ad.Variable("x")
    w1 = ad.Variable("w1")
    b1 = ad.Variable("b1")
    gamma = ad.Variable("gamma")
    beta = ad.Variable("beta")
    w2 = ad.Variable("w2")
    b2 = ad.Variable("b2")
    y_ = ad.Variable("y_")

    mm1 = ad.matmul_op(x, w1)
    h1_lin = mm1 + ad.broadcastto_op(b1, mm1)
    h1_act = ad.gelu_op(h1_lin)
    h1_ln = ad.layernorm_unfused_op(h1_act, gamma, beta, eps=1e-5)
    mm2 = ad.matmul_op(h1_ln, w2)
    logits = mm2 + ad.broadcastto_op(b2, mm2)
    loss = ad.softmaxcrossentropy_op(logits, y_)
    gw1, gb1, ggamma, gbeta, gw2, gb2 = ad.gradients(loss, [w1, b1, gamma, beta, w2, b2])

    eval_nodes = [loss, gw1, gb1, ggamma, gbeta, gw2, gb2]
    exec_unfused = ad.Executor(eval_nodes, ctx=ctx, enable_fusion=False)
    exec_fused_cuda = ad.Executor(
        eval_nodes, ctx=ctx, enable_fusion=True,
        scheduler_policy="min_peak_memory", enable_memory_planning=True, backend="cuda"
    )
    exec_fused_triton = ad.Executor(
        eval_nodes, ctx=ctx, enable_fusion=True,
        scheduler_policy="min_peak_memory", enable_memory_planning=True, backend="triton"
    )

    assert len(exec_fused_cuda.topo_order) < len(exec_unfused.topo_order)

    x_val = rng.standard_normal((batch, in_dim), dtype=np.float32)
    w1_val = rng.standard_normal((in_dim, hid_dim), dtype=np.float32) * 0.1
    b1_val = rng.standard_normal((hid_dim,), dtype=np.float32) * 0.1
    g_val = np.ones((hid_dim,), dtype=np.float32)
    beta_val = np.zeros((hid_dim,), dtype=np.float32)
    w2_val = rng.standard_normal((hid_dim, num_classes), dtype=np.float32) * 0.1
    b2_val = rng.standard_normal((num_classes,), dtype=np.float32) * 0.1
    y_val = np.eye(num_classes, dtype=np.float32)[rng.integers(0, num_classes, size=batch)]

    feed = {
        x: x_val, w1: w1_val, b1: b1_val,
        gamma: g_val, beta: beta_val,
        w2: w2_val, b2: b2_val, y_: y_val,
    }
    res_unfused = exec_unfused.run(feed, convert_to_numpy_ret_vals=True)
    res_fused_cuda = exec_fused_cuda.run(feed, convert_to_numpy_ret_vals=True)
    res_fused_triton = exec_fused_triton.run(feed, convert_to_numpy_ret_vals=True)

    for u, fc, ft in zip(res_unfused, res_fused_cuda, res_fused_triton):
        np.testing.assert_allclose(fc, u, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(ft, u, rtol=1e-2, atol=1e-2)


def test_scheduler_and_memory_planner_and_allocator():
    ctx = ndarray.gpu(0)
    ndarray.empty_cache(ctx)
    ndarray.reset_memory_stats(ctx)

    for _ in range(10):
        tmp = ndarray.empty((256, 256), ctx=ctx)
        del tmp
    stats = ndarray.memory_stats(ctx)
    assert stats["pool_Cache_hits"] >= 9
    assert stats["cuda_malloc_calls"] == 1

    x = ad.Variable("x")
    cur = x
    for _ in range(6):
        cur = ad.relu_op(cur + 1.0)
    shapes = {x: (128, 256)}
    exec_naive = ad.Executor([cur], ctx=ctx, enable_memory_planning=False)
    exec_planned = ad.Executor(
        [cur], ctx=ctx, scheduler_policy="min_peak_memory", enable_memory_planning=True
    )
    feed = {x: np.ones(shapes[x], dtype=np.float32)}
    out_naive = exec_naive.run(feed, convert_to_numpy_ret_vals=True)[0]
    out_planned = exec_planned.run(feed, convert_to_numpy_ret_vals=True)[0]
    np.testing.assert_allclose(out_planned, out_naive, rtol=1e-5)
    assert exec_planned.memory_plan_info["planned_pool_bytes"] < exec_naive.memory_plan_info["naive_total_bytes"]

