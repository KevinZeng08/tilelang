# ruff: noqa

import torch
import time
import argparse
import tilelang
from tilelang import language as T
import tilelang.testing
from typing import Optional, Union
import triton
import triton.language as tl

from example_triton_nsa_bwd import parallel_nsa_fwd, parallel_nsa_bwd
from example_tilelang_nsa_bwd import tilelang_kernel_fwd, parallel_nsa_bwd as tilelang_parallel_nsa_bwd
from reference import naive_nsa, MemRecorder


def generate_block_indices(batch, seq_len, heads, selected_blocks, block_size):
    """Generate random block indices for the benchmark."""
    block_indices = torch.full((batch, seq_len, heads, selected_blocks),
                               seq_len,
                               dtype=torch.long,
                               device='cuda')

    for b in range(batch):
        for t in range(seq_len):
            for h in range(heads):
                i_i = torch.randperm(max(1, (t // block_size)))[:selected_blocks]
                block_indices[b, t, h, :len(i_i)] = i_i

    return block_indices.sort(-1)[0]


# ============================================================================
# Triton Kernels for Forward and Backward Pass
# ============================================================================


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None,
    'USE_BLOCK_COUNTS': lambda args: isinstance(args['block_counts'], torch.Tensor),
})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=['BS', 'BK', 'BV'],
)
@triton.jit
def parallel_nsa_fwd_kernel(q, k, v, o_slc, o_swa, lse_slc, lse_swa, scale, block_indices,
                            block_counts, offsets, token_indices, T, H: tl.constexpr,
                            HQ: tl.constexpr, G: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                            S: tl.constexpr, BS: tl.constexpr, WS: tl.constexpr, BK: tl.constexpr,
                            BV: tl.constexpr, USE_OFFSETS: tl.constexpr,
                            USE_BLOCK_COUNTS: tl.constexpr):
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if USE_OFFSETS:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(token_indices + i_t * 2 +
                                                                          1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    block_indices += (bos + i_t) * H * S + i_h * S

    if USE_BLOCK_COUNTS:
        NS = tl.load(block_counts + (bos + i_t) * H + i_h)
    else:
        NS = S

    p_q = tl.make_block_ptr(q + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK),
                            (1, 0))
    # the Q block is kept in the shared memory throughout the whole kernel
    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    p_o_slc = tl.make_block_ptr(o_slc + (bos + i_t) * HQ * V, (HQ, V), (V, 1), (i_h * G, i_v * BV),
                                (G, BV), (1, 0))
    p_lse_slc = lse_slc + (bos + i_t) * HQ + i_h * G + tl.arange(0, G)
    # [G, BV]
    b_o_slc = tl.zeros([G, BV], dtype=tl.float32)

    b_m_slc = tl.full([G], float('-inf'), dtype=tl.float32)
    b_acc_slc = tl.zeros([G], dtype=tl.float32)
    for i in range(NS):
        i_s = tl.load(block_indices + i).to(tl.int32) * BS
        if i_s <= i_t and i_s >= 0:
            p_k_slc = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
            p_v_slc = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
            # [BK, BS]
            b_k_slc = tl.load(p_k_slc, boundary_check=(0, 1))
            # [BS, BV]
            b_v_slc = tl.load(p_v_slc, boundary_check=(0, 1))
            # [G, BS]
            b_s_slc = tl.dot(b_q, b_k_slc)
            b_s_slc = tl.where((i_t >= (i_s + tl.arange(0, BS)))[None, :], b_s_slc, float('-inf'))

            # [G]
            b_m_slc, b_mp_slc = tl.maximum(b_m_slc, tl.max(b_s_slc, 1)), b_m_slc
            b_r_slc = tl.exp(b_mp_slc - b_m_slc)
            # [G, BS]
            b_p_slc = tl.exp(b_s_slc - b_m_slc[:, None])
            # [G]
            b_acc_slc = b_acc_slc * b_r_slc + tl.sum(b_p_slc, 1)
            # [G, BV]
            b_o_slc = b_o_slc * b_r_slc[:, None] + tl.dot(b_p_slc.to(b_q.dtype), b_v_slc)

            b_mp_slc = b_m_slc
    b_o_slc = b_o_slc / b_acc_slc[:, None]
    b_m_slc += tl.log(b_acc_slc)
    tl.store(p_o_slc, b_o_slc.to(p_o_slc.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_lse_slc, b_m_slc.to(p_lse_slc.dtype.element_ty))

    if WS > 0:
        p_o_swa = tl.make_block_ptr(o_swa + (bos + i_t) * HQ * V, (HQ, V), (V, 1),
                                    (i_h * G, i_v * BV), (G, BV), (1, 0))
        p_lse_swa = lse_swa + (bos + i_t) * HQ + i_h * G + tl.arange(0, G)
        # [G, BV]
        b_o_swa = tl.zeros([G, BV], dtype=tl.float32)

        b_m_swa = tl.full([G], float('-inf'), dtype=tl.float32)
        b_acc_swa = tl.zeros([G], dtype=tl.float32)
        for i_s in range(max(0, i_t - WS + 1), i_t + 1, BS):
            p_k_swa = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
            p_v_swa = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
            # [BK, BS]
            b_k_swa = tl.load(p_k_swa, boundary_check=(0, 1))
            # [BS, BV]
            b_v_swa = tl.load(p_v_swa, boundary_check=(0, 1))
            # [G, BS]
            b_s_swa = tl.dot(b_q, b_k_swa)
            b_s_swa = tl.where((i_t >= (i_s + tl.arange(0, BS)))[None, :], b_s_swa, float('-inf'))

            # [G]
            b_m_swa, b_mp_swa = tl.maximum(b_m_swa, tl.max(b_s_swa, 1)), b_m_swa
            b_r_swa = tl.exp(b_mp_swa - b_m_swa)
            # [G, BS]
            b_p_swa = tl.exp(b_s_swa - b_m_swa[:, None])
            # [G]
            b_acc_swa = b_acc_swa * b_r_swa + tl.sum(b_p_swa, 1)
            # [G, BV]
            b_o_swa = b_o_swa * b_r_swa[:, None] + tl.dot(b_p_swa.to(b_q.dtype), b_v_swa)

            b_mp_swa = b_m_swa
        b_o_swa = b_o_swa / b_acc_swa[:, None]
        b_m_swa += tl.log(b_acc_swa)
        tl.store(p_o_swa, b_o_swa.to(p_o_swa.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_lse_swa, b_m_swa.to(p_lse_swa.dtype.element_ty))


@triton.jit
def parallel_nsa_bwd_kernel_preprocess(o, do, delta, B: tl.constexpr, V: tl.constexpr):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, B)
    m_d = o_d < V

    b_o = tl.load(o + i_n * V + o_d, mask=m_d, other=0)
    b_do = tl.load(do + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)
    b_delta = tl.sum(b_o * b_do)

    tl.store(delta + i_n, b_delta.to(delta.dtype.element_ty))


def benchmark_triton_nsa(batch_size,
                         seq_len,
                         heads,
                         head_query,
                         dim,
                         selected_blocks,
                         block_size,
                         dtype,
                         scale,
                         warmup=10,
                         iterations=100,
                         validate=False):
    """Benchmark the Triton-based TileLang Sparse Attention implementation."""

    # Set random seed for reproducibility
    tilelang.testing.set_random_seed(0)
    torch.random.manual_seed(0)

    # Create input tensors
    Q = torch.randn((batch_size, seq_len, head_query, dim), dtype=dtype,
                    device='cuda').requires_grad_(True)
    K = torch.randn((batch_size, seq_len, heads, dim), dtype=dtype,
                    device='cuda').requires_grad_(True)
    V = torch.randn((batch_size, seq_len, heads, dim), dtype=dtype,
                    device='cuda').requires_grad_(True)
    g_slc = torch.ones((batch_size, seq_len, head_query), dtype=dtype,
                       device='cuda').requires_grad_(True)
    g_swa = torch.ones((batch_size, seq_len, head_query), dtype=dtype,
                       device='cuda').requires_grad_(True)
    do = torch.randn((batch_size, seq_len, head_query, dim), dtype=dtype, device='cuda')

    # Generate block indices
    block_indices = generate_block_indices(batch_size, seq_len, heads, selected_blocks, block_size)
    block_counts = torch.randint(
        1, selected_blocks + 1, (batch_size, seq_len, heads), device='cuda')
    # o_slc = torch.empty((batch_size, seq_len, head_query, dim), dtype=dtype, device='cuda')
    # lse_slc = torch.empty((batch_size, seq_len, head_query), dtype=torch.float, device='cuda')

    o_slc, lse_slc, o_swa, lse_swa = parallel_nsa_fwd(
        q=Q,
        k=K,
        v=V,
        # o_slc=o_slc,
        # o_swa=None,
        # lse_slc=lse_slc,
        # lse_swa=None,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
        window_size=0,
        scale=scale)
    # if window_size > 0:
    #     o = torch.addcmul(o_slc * g_slc.unsqueeze(-1), o_swa, g_swa.unsqueeze(-1))
    # else:
    o = o_slc * g_slc.unsqueeze(-1)

    # Warmup
    for _ in range(warmup):
        dq, dk, dv = parallel_nsa_bwd(
            q=Q,
            k=K,
            v=V,
            o_slc=o,
            o_swa=o_swa,
            lse_slc=lse_slc,
            lse_swa=lse_swa,
            do_slc=do,
            do_swa=None,
            block_indices=block_indices,
            block_counts=block_counts,
            block_size=block_size,
            window_size=0,
            scale=scale,
        )

    # Synchronize before timing
    torch.cuda.synchronize()

    # Benchmark
    mems = [0.0] * iterations
    start_time = time.time()
    for i in range(iterations):
        with MemRecorder(mode="peak") as mr:
            dq, dk, dv = parallel_nsa_bwd(
                q=Q,
                k=K,
                v=V,
                o_slc=o,
                o_swa=o_swa,
                lse_slc=lse_slc,
                lse_swa=lse_swa,
                do_slc=do,
                do_swa=None,
                block_indices=block_indices,
                block_counts=block_counts,
                block_size=block_size,
                window_size=0,
                scale=scale,
            )
        mems[i] = mr.memory

    torch.cuda.synchronize()
    end_time = time.time()

    # Calculate metrics
    elapsed_time = end_time - start_time
    avg_time = elapsed_time / iterations * 1000  # ms
    avg_memory = sum(mems) / iterations / (1024**3)  # GB

    flops_per_token = 4 * dim * selected_blocks * block_size
    flops_per_token = flops_per_token * 2.5  # for backward pass, x2.5 TFLOPS
    total_flops = batch_size * seq_len * head_query * flops_per_token
    flops_per_sec = total_flops / (elapsed_time / iterations)
    tflops = flops_per_sec / 1e12

    # TODO: validate correctness
    if validate:
        ref = naive_nsa(
            q=Q,
            k=K,
            v=V,
            g_slc=g_slc,
            g_swa=g_swa,
            block_indices=block_indices,
            block_counts=block_counts,
            block_size=block_size,
        )
        ref.backward(do)
        ref_dq, Q.grad = Q.grad.clone(), None
        ref_dk, K.grad = K.grad.clone(), None
        ref_dv, V.grad = V.grad.clone(), None
        ref_dg_slc, g_slc.grad = g_slc.grad.clone(), None

        # assert_close(" o", ref, tri, 0.004)
        torch.testing.assert_close(ref, o, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(ref_dq, dq, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(ref_dk, dk, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(ref_dv, dv, atol=1e-2, rtol=1e-2)
        # torch.testing.assert_close(ref_dg_slc, tri_dg_slc, atol=1e-2, rtol=1e-2)

    # Return benchmark results
    return {
        "avg_time_ms": avg_time,
        "tflops": tflops,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "heads": heads,
        "head_query": head_query,
        "dim": dim,
        "selected_blocks": selected_blocks,
        "block_size": block_size,
        "avg_memory_gb": avg_memory,
    }


def benchmark_nsa(batch_size,
                  seq_len,
                  heads,
                  head_query,
                  dim,
                  selected_blocks,
                  block_size,
                  dtype,
                  scale,
                  warmup=10,
                  iterations=100,
                  validate=False):
    """Benchmark the TileLang Sparse Attention implementation."""

    # Set random seed for reproducibility
    tilelang.testing.set_random_seed(0)
    torch.random.manual_seed(0)

    # Create input tensors
    Q = torch.randn((batch_size, seq_len, head_query, dim), dtype=dtype,
                    device='cuda').requires_grad_(True)
    K = torch.randn((batch_size, seq_len, heads, dim), dtype=dtype,
                    device='cuda').requires_grad_(True)
    V = torch.randn((batch_size, seq_len, heads, dim), dtype=dtype,
                    device='cuda').requires_grad_(True)
    g_slc = torch.ones((batch_size, seq_len, head_query), dtype=dtype,
                       device='cuda').requires_grad_(True)
    g_swa = torch.ones((batch_size, seq_len, head_query), dtype=dtype,
                       device='cuda').requires_grad_(True)
    do = torch.randn((batch_size, seq_len, head_query, dim), dtype=dtype, device='cuda')

    # Generate block indices
    block_indices = generate_block_indices(batch_size, seq_len, heads, selected_blocks, block_size)
    block_counts = torch.randint(
        1, selected_blocks + 1, (batch_size, seq_len, heads), device='cuda')
    o_slc = torch.empty((batch_size, seq_len, head_query, dim), dtype=dtype, device='cuda')
    lse_slc = torch.empty((batch_size, seq_len, head_query), dtype=torch.float, device='cuda')

    B, SEQLEN, HQ, D = Q.shape
    H = K.shape[2]
    G = HQ // H
    S = block_indices.shape[-1]
    kernel = tilelang_kernel_fwd(
        batch=B,
        heads=HQ,
        seq_len=SEQLEN,
        dim=D,
        is_causal=True,
        scale=scale,
        block_size=block_size,
        groups=G,
        selected_blocks=S,
    )
    kernel(Q, K, V, block_indices.to(torch.int32), o_slc, lse_slc)

    # Warmup
    for _ in range(warmup):
        # print(f"Warmup {warmup}")
        dq, dk, dv = tilelang_parallel_nsa_bwd(
            q=Q,
            k=K,
            v=V,
            o_slc=o_slc,
            o_swa=None,
            lse_slc=lse_slc,
            lse_swa=None,
            do_slc=do,
            do_swa=None,
            block_indices=block_indices,
            block_counts=block_counts,
            block_size=block_size,
            window_size=0,
            scale=scale,
        )

    # Synchronize before timing
    torch.cuda.synchronize()

    # Benchmark
    mems = [0.0] * iterations
    start_time = time.time()
    for j in range(iterations):
        # print(f"Iteration {j+1}/{iterations}")
        with MemRecorder(mode="peak") as mr:
            dq, dk, dv = tilelang_parallel_nsa_bwd(
                q=Q,
                k=K,
                v=V,
                o_slc=o_slc,
                o_swa=None,
                lse_slc=lse_slc,
                lse_swa=None,
                do_slc=do,
                do_swa=None,
                block_indices=block_indices,
                block_counts=block_counts,
                block_size=block_size,
                window_size=0,
                scale=scale,
            )
        mems[j] = mr.memory

    torch.cuda.synchronize()
    end_time = time.time()

    # Calculate metrics
    elapsed_time = end_time - start_time
    avg_time = elapsed_time / iterations * 1000  # ms
    avg_memory = sum(mems) / iterations / (1024**3)  # GB

    flops_per_token = 4 * dim * selected_blocks * block_size
    flops_per_token = flops_per_token * 2.5  # for backward pass, x2.5 TFLOPS
    total_flops = batch_size * seq_len * head_query * flops_per_token
    flops_per_sec = total_flops / (elapsed_time / iterations)
    tflops = flops_per_sec / 1e12

    # Return benchmark results
    return {
        "avg_time_ms": avg_time,
        "tflops": tflops,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "heads": heads,
        "head_query": head_query,
        "dim": dim,
        "selected_blocks": selected_blocks,
        "block_size": block_size,
        "avg_memory_gb": avg_memory,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark TileLang Sparse Attention")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--heads", type=int, default=1, help="Number of heads")
    parser.add_argument("--head_query", type=int, default=16, help="Number of query heads")
    parser.add_argument("--dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--selected_blocks", type=int, default=16, help="Number of selected blocks")
    parser.add_argument("--block_size", type=int, default=32, help="Block size")
    parser.add_argument(
        "--dtype", type=str, default="float16", help="Data type (float16 or float32)")
    parser.add_argument("--scale", type=float, default=0.1, help="Attention scale factor")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--validate", action="store_true", help="Validate against reference")
    parser.add_argument("--suite", action="store_true", help="Run benchmark suite")
    parser.add_argument(
        "--impl",
        type=str,
        default="all",
        choices=["tilelang", "triton", "all"],
        help="Implementation to benchmark (tilelang, triton, or all)")

    args = parser.parse_args()

    # For Triton impl, ensure head_query is a multiple of heads*16
    if args.impl in ["triton", "all"] and args.head_query % (args.heads * 16) != 0:
        # Adjust head_query to nearest valid value
        args.head_query = ((args.head_query // (args.heads * 16)) + 1) * (args.heads * 16)
        print(
            f"Adjusted head_query to {args.head_query} to be compatible with Triton implementation")

    if args.suite:
        run_benchmark_suite(impl=args.impl)
    else:
        dtype = torch.float16 if args.dtype == "float16" else torch.float32

        if args.impl in ["tilelang", "all"]:
            print("Benchmarking TileLang implementation:")
            result = benchmark_nsa(
                batch_size=args.batch,
                seq_len=args.seq_len,
                heads=args.heads,
                head_query=args.head_query,
                dim=args.dim,
                selected_blocks=args.selected_blocks,
                block_size=args.block_size,
                dtype=dtype,
                scale=args.scale,
                warmup=args.warmup,
                iterations=args.iterations,
                validate=args.validate)
            print("\nBenchmark Results (TileLang):")
            print(
                f"Configuration: batch={args.batch}, seq_len={args.seq_len}, heads={args.heads}, " +
                f"head_query={args.head_query}, dim={args.dim}, blocks={args.selected_blocks}, " +
                f"block_size={args.block_size}")
            print(f"Average time: {result['avg_time_ms']:.2f} ms")
            print(f"Performance: {result['tflops']:.2f} TFLOPs")
            print(f"Memory Usage: {result['avg_memory_gb']:.2f} GB")

        if args.impl in ["triton", "all"]:
            print("Benchmarking Triton implementation:")
            result = benchmark_triton_nsa(
                batch_size=args.batch,
                seq_len=args.seq_len,
                heads=args.heads,
                head_query=args.head_query,
                dim=args.dim,
                selected_blocks=args.selected_blocks,
                block_size=args.block_size,
                dtype=dtype,
                scale=args.scale,
                warmup=args.warmup,
                iterations=args.iterations,
                validate=args.validate)
            print("\nBenchmark Results (Triton):")
            print(
                f"Configuration: batch={args.batch}, seq_len={args.seq_len}, heads={args.heads}, " +
                f"head_query={args.head_query}, dim={args.dim}, blocks={args.selected_blocks}, " +
                f"block_size={args.block_size}")
            print(f"Average time: {result['avg_time_ms']:.2f} ms")
            print(f"Performance: {result['tflops']:.2f} TFLOPs")
            print(f"Memory Usage: {result['avg_memory_gb']:.2f} GB")
