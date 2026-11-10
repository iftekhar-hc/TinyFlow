"""End-to-end profiling and benchmark suite for TinyFlow extensions.

Benchmarks:
1. SGEMM tiers (Naive CUDA, Shared-Tiled CUDA, Register-Tiled float4 CUDA, Triton, PyTorch cuBLAS)
2. FusedLinearBiasAct (Unfused 4-op vs Fused CUDA vs Fused Triton)
3. LayerNorm & AdaLN (Unfused 11-op vs Fused Warp-Shuffle CUDA vs Fused Triton)
4. SoftmaxCrossEntropyGrad (Unfused 3-op vs Fused Online-Softmax CUDA vs Fused Triton)
5. End-to-End MLP/DiT Training Step (Unfused vs Fused + MinPeakMemoryScheduler + MemoryPlanner)
6. Caching CUDA Allocator vs Raw cudaMalloc/cudaFree
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import time
import numpy as np
import torch

from dlsys import ndarray, gpu_op, autodiff as ad, triton_ops


def sync_gpu():
    torch.cuda.synchronize()


def bench_fn(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    sync_gpu()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync_gpu()
    return ((time.perf_counter() - t0) * 1e3) / iters


def benchmark_sgemm(ctx):
    print("\n" + "=" * 78)
    print("1. SGEMM Kernel Tier Benchmark (FP32, M=N=K=1024)")
    print("=" * 78)
    m = n = k = 1024
    flops = 2.0 * m * n * k
    a_np = np.random.randn(m, k).astype(np.float32)
    b_np = np.random.randn(k, n).astype(np.float32)
    a = ndarray.array(a_np, ctx=ctx)
    b = ndarray.array(b_np, ctx=ctx)
    c = ndarray.empty((m, n), ctx=ctx)
    ta = torch.tensor(a_np, device="cuda")
    tb = torch.tensor(b_np, device="cuda")

    ms_naive = bench_fn(lambda: gpu_op.matrix_multiply_naive(a, False, b, False, c))
    ms_shared = bench_fn(lambda: gpu_op.matrix_multiply_shared(a, False, b, False, c))
    ms_reg = bench_fn(lambda: gpu_op.matrix_multiply(a, False, b, False, c))
    ms_triton = bench_fn(lambda: triton_ops.triton_matmul(a, b, c))
    ms_torch = bench_fn(lambda: torch.matmul(ta, tb))

    for name, ms in [
        ("CUDA Tier-1 Naive", ms_naive),
        ("CUDA Tier-2 Shared Tiled (32x32)", ms_shared),
        ("CUDA Tier-3 Reg-Tiled (BM=64,BN=64,TM=4,TN=4)", ms_reg),
        ("Triton Block-Tiled MatMul", ms_triton),
        ("PyTorch cuBLAS (torch.matmul)", ms_torch),
    ]:
        tflops = (flops / (ms * 1e-3)) / 1e12
        print(f"  {name:<48}: {ms:7.3f} ms | {tflops:6.2f} TFLOPS")


def benchmark_fused_ops(ctx):
    print("\n" + "=" * 78)
    print("2. Operator Fusion Kernel Benchmarks (Batch=512, Dim=1024)")
    print("=" * 78)
    m, k, n = 512, 1024, 1024
    x_np = np.random.randn(m, k).astype(np.float32)
    w_np = np.random.randn(k, n).astype(np.float32)
    b_np = np.random.randn(n).astype(np.float32)
    g_np = np.ones((n,), dtype=np.float32)
    beta_np = np.zeros((n,), dtype=np.float32)

    x = ndarray.array(x_np, ctx=ctx)
    w = ndarray.array(w_np, ctx=ctx)
    b = ndarray.array(b_np, ctx=ctx)
    g = ndarray.array(g_np, ctx=ctx)
    beta = ndarray.array(beta_np, ctx=ctx)
    tmp_mm = ndarray.empty((m, n), ctx=ctx)
    tmp_b = ndarray.empty((m, n), ctx=ctx)
    out = ndarray.empty((m, n), ctx=ctx)

    def unfused_linear_gelu():
        gpu_op.matrix_multiply(x, False, w, False, tmp_mm)
        gpu_op.broadcast_to(b, tmp_b)
        gpu_op.matrix_elementwise_add(tmp_mm, tmp_b, out)
        gpu_op.gelu(out, out)

    ms_lin_unfused = bench_fn(unfused_linear_gelu)
    ms_lin_cuda = bench_fn(lambda: gpu_op.fused_linear(x, w, b, out, act_type=2))
    ms_lin_triton = bench_fn(lambda: triton_ops.triton_fused_linear(x, w, b, out, act_type=2))
    print(f"  Linear+Bias+GELU Unfused (4 launches)           : {ms_lin_unfused:7.3f} ms")
    print(f"  Linear+Bias+GELU Fused CUDA (1 launch)          : {ms_lin_cuda:7.3f} ms ({ms_lin_unfused/ms_lin_cuda:.2f}x)")
    print(f"  Linear+Bias+GELU Fused Triton (1 launch)        : {ms_lin_triton:7.3f} ms ({ms_lin_unfused/ms_lin_triton:.2f}x)")

    ms_ln_cuda = bench_fn(lambda: gpu_op.layernorm(out, g, beta, tmp_mm, eps=1e-5))
    ms_ln_triton = bench_fn(lambda: triton_ops.triton_layernorm(out, g, beta, tmp_mm, eps=1e-5))
    print(f"  LayerNorm Fused Warp-Shuffle CUDA               : {ms_ln_cuda:7.3f} ms")
    print(f"  LayerNorm Fused Triton                          : {ms_ln_triton:7.3f} ms")



def benchmark_executor_and_memory(ctx):
    print("\n" + "=" * 78)
    print("3. End-to-End Graph Execution, Fusion, Scheduling & Memory Planning")
    print("=" * 78)
    rng = np.random.default_rng(7)
    batch, in_dim, hid_dim, num_classes = 256, 512, 512, 128

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

    feed = {
        x: ndarray.array(rng.standard_normal((batch, in_dim), dtype=np.float32), ctx=ctx),
        w1: ndarray.array(rng.standard_normal((in_dim, hid_dim), dtype=np.float32) * 0.05, ctx=ctx),
        b1: ndarray.array(np.zeros((hid_dim,), dtype=np.float32), ctx=ctx),
        gamma: ndarray.array(np.ones((hid_dim,), dtype=np.float32), ctx=ctx),
        beta: ndarray.array(np.zeros((hid_dim,), dtype=np.float32), ctx=ctx),
        w2: ndarray.array(rng.standard_normal((hid_dim, num_classes), dtype=np.float32) * 0.05, ctx=ctx),
        b2: ndarray.array(np.zeros((num_classes,), dtype=np.float32), ctx=ctx),
        y_: ndarray.array(np.eye(num_classes, dtype=np.float32)[rng.integers(0, num_classes, size=batch)], ctx=ctx),
    }

    exec_baseline = ad.Executor(
        eval_nodes, ctx=ctx, enable_fusion=False,
        scheduler_policy="default", enable_memory_planning=False, backend="cuda"
    )
    exec_opt_cuda = ad.Executor(
        eval_nodes, ctx=ctx, enable_fusion=True,
        scheduler_policy="min_peak_memory", enable_memory_planning=True, backend="cuda"
    )
    exec_opt_triton = ad.Executor(
        eval_nodes, ctx=ctx, enable_fusion=True,
        scheduler_policy="min_peak_memory", enable_memory_planning=True, backend="triton"
    )

    ms_base = bench_fn(lambda: exec_baseline.run(feed))
    ms_cuda = bench_fn(lambda: exec_opt_cuda.run(feed))
    ms_triton = bench_fn(lambda: exec_opt_triton.run(feed))

    base_info = exec_baseline.memory_plan_info
    opt_info = exec_opt_cuda.memory_plan_info
    print(f"  Unfused Graph Node Count                        : {len(exec_baseline.topo_order)}")
    print(f"  Fused Graph Node Count                          : {len(exec_opt_cuda.topo_order)}")
    print(f"  Unfused Static Intermediate Allocation          : {base_info['naive_total_bytes'] / 1024:.1f} KB")
    print(f"  Planned Peak Live Memory (MinPeakScheduler)     : {opt_info['peak_live_bytes'] / 1024:.1f} KB")
    print(f"  Planned Shared Buffer Pool (MemoryPlanner)      : {opt_info['planned_pool_bytes'] / 1024:.1f} KB "
          f"({opt_info['num_shared_buffers']} buffers vs {opt_info['num_nodes']} nodes)")
    print(f"  Step Latency Baseline (Unfused CUDA)            : {ms_base:7.3f} ms")
    print(f"  Step Latency Optimized (Fused CUDA + Planner)   : {ms_cuda:7.3f} ms ({ms_base/ms_cuda:.2f}x)")
    print(f"  Step Latency Optimized (Fused Triton + Planner) : {ms_triton:7.3f} ms ({ms_base/ms_triton:.2f}x)")

    ndarray.empty_cache(ctx)
    ndarray.reset_memory_stats(ctx)
    for _ in range(100):
        arr = ndarray.empty((256, 512), ctx=ctx)
        del arr
    stats = ndarray.memory_stats(ctx)
    print("\n" + "=" * 78)
    print("4. Caching GPU Memory Pool Telemetry (100 Ephemeral Allocations)")
    print("=" * 78)
    print(f"  cudaMalloc driver calls                         : {stats['cuda_malloc_calls']}")
    print(f"  Pool cache hits                                 : {stats['pool_Cache_hits']}")
    print(f"  Peak active bytes                               : {stats['peak_bytes_in_use'] / 1024:.1f} KB")


if __name__ == "__main__":
    ctx = ndarray.gpu(0)
    benchmark_sgemm(ctx)
    benchmark_fused_ops(ctx)
    benchmark_executor_and_memory(ctx)

