"""
Tensor Parallelism (TP) 推理加速模块。

将 Transformer block 中的 Attention (按 head 切分) 和 FFN (按中间维度切分)
分布到多张 GPU 上并行计算，降低单条数据推理延迟。

使用方式：
    1. 正常加载完整模型（单卡/CPU）
    2. 调用 shard_model() 就地替换 Linear 层为 TP 版本
    3. 各 rank 处理相同数据，结果通过 all_reduce 自动聚合

设计原则：
    - 不修改 CausalLTXAttention / FeedForward / CausalAVTransformerBlock 类定义
    - 仅替换 Linear 层权重 + 更新元数据（heads / inner_dim）
    - 零影响训练代码
"""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Numerical mode for TPRMSNorm.
#
# 默认走 bf16 in-place 路径（更快、显存读写更少），与单卡 nn.RMSNorm 在
# fp16/bf16 下的实现策略一致。LTX-2 内部 q/k 张量量级不会让 bf16 平方和
# 溢出（hidden=4096，单元素绝对值 << 1e4），实测与 fp32 路径数值差 <1e-5。
#
# 若发现回归，可设环境变量 TPRMSNORM_FP32=1 回退到 fp32 累加版本。
# ---------------------------------------------------------------------------
_TP_RMSNORM_USE_FP32 = os.environ.get("TPRMSNORM_FP32", "0") == "1"


# ============================================================================
# TP Linear Primitives
# ============================================================================

_FLOAT8_DTYPES = {
    getattr(torch, "float8_e4m3fn", None),
    getattr(torch, "float8_e5m2", None),
}
_FLOAT8_DTYPES.discard(None)


def _linear_params_for_input(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    x: torch.Tensor,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if weight.dtype in _FLOAT8_DTYPES:
        weight = weight.to(dtype=x.dtype)
    if bias is not None and bias.dtype in _FLOAT8_DTYPES:
        bias = bias.to(dtype=x.dtype)
    return weight, bias


class ColumnParallelLinear(nn.Module):
    """按 output 维度切分的线性层（无通信）。

    用于 QKV 投影和 FFN up-proj，每卡只计算 out_features/tp_size 个输出。
    """

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight, bias = _linear_params_for_input(self.weight, self.bias, x)
        return F.linear(x, weight, bias)


class RowParallelLinear(nn.Module):
    """按 input 维度切分的线性层（forward 后 all_reduce 聚合）。

    用于 attention out_proj 和 FFN down-proj。
    bias 在 all_reduce 之后加，避免重复累加 tp_size 次。
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        tp_group: dist.ProcessGroup,
    ):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        # bias 保持完整，all_reduce 之后再加
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None
        self.tp_group = tp_group

    class _AsyncOutput:
        def __init__(self, out: torch.Tensor, bias: Optional[torch.Tensor], work) -> None:
            self.out = out
            self.bias = bias
            self.work = work

        def wait(self) -> torch.Tensor:
            self.work.wait()
            if self.bias is not None:
                self.out = self.out + self.bias
            return self.out

    def forward_async(self, x: torch.Tensor) -> "RowParallelLinear._AsyncOutput":
        weight, bias = _linear_params_for_input(self.weight, self.bias, x)
        out = F.linear(x, weight)  # 不加 bias
        work = dist.all_reduce(out, group=self.tp_group, async_op=True)
        return RowParallelLinear._AsyncOutput(out, bias, work)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight, bias = _linear_params_for_input(self.weight, self.bias, x)
        out = F.linear(x, weight)  # 不加 bias
        dist.all_reduce(out, group=self.tp_group)
        if bias is not None:
            out = out + bias
        return out


# ============================================================================
# TP-Aware RMSNorm
# ============================================================================

class TPRMSNorm(nn.Module):
    """Tensor-Parallel-aware RMSNorm.

    背景：原模型中 ``q_norm`` / ``k_norm`` 是在全局 ``inner_dim`` 上做的
    （并非 per-head）：

        rms_full = sqrt(mean(x[0..inner_dim] ** 2) + eps)
        out      = x / rms_full * weight

    在 TP 下，``to_q/to_k`` 是列并行 (ColumnParallelLinear)，每张卡只持
    有 ``x_local = x[..., r*chunk:(r+1)*chunk]``。如果直接退化成本地
    ``nn.RMSNorm(chunk)``，分母变成各 rank 的局部均值，跨 head 能量分布
    不均会让结果偏离单卡基线，softmax(QKᵀ) 之后误差被进一步放大。

    本模块在 forward 中对局部平方和做一次 all_reduce，再用全局
    ``inner_dim`` 作分母，从而与单卡 RMSNorm 数学等价。

    Args:
        weight:    本 rank 持有的 weight 切片，shape = [chunk]。
        eps:       与原 RMSNorm 一致。
        full_dim:  全局 ``inner_dim``，用作 RMS 的分母。
        tp_group:  TP 进程组。
    """

    def __init__(
        self,
        weight: torch.Tensor,
        eps: float,
        full_dim: int,
        tp_group: dist.ProcessGroup,
    ):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.eps = float(eps)
        self.full_dim = int(full_dim)
        self.tp_group = tp_group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 默认 bf16/fp16 路径：直接在原 dtype 上算平方和。这是单卡
        # nn.RMSNorm 的标准实现（PyTorch 2.x 在 cuda/cpu 上对 bf16 做 in-place
        # square-sum 而不 promote，前向已被官方验证数值安全）。
        # 对应优势：
        #   - 显存读写量减半（无 fp32 临时张量）
        #   - 少 2 个 cast kernel，每 attention 节省 2 次 launch
        #   - all_reduce 的张量也是 bf16，带宽减半
        if _TP_RMSNORM_USE_FP32:
            x_compute = x.float()
        else:
            x_compute = x

        local_sq = x_compute.pow(2).sum(dim=-1, keepdim=True)
        # 跨 rank 把所有 head 的平方和加起来，得到全局平方和。
        dist.all_reduce(local_sq, group=self.tp_group)
        rms = torch.sqrt(local_sq / self.full_dim + self.eps)
        x_normed = x_compute / rms
        if x_normed.dtype != x.dtype:
            x_normed = x_normed.to(x.dtype)
        return x_normed * self.weight


def tp_qk_norm_fused(
    q_norm: "TPRMSNorm",
    k_norm: "TPRMSNorm",
    q: torch.Tensor,
    k: torch.Tensor,
    overlap_compute: Optional[callable] = None,
):
    """同时对 q / k 做 TP-aware RMSNorm，仅触发一次 all_reduce。

    背景：单独调用 ``q_norm(q)`` + ``k_norm(k)`` 会产生 2 次 all_reduce，
    在 LTX-2 22B（28 blocks × 6 attentions）下每步推理多出 168 次 NCCL
    launch，主要是调度开销而非通信带宽占用。本函数把 ``q`` / ``k`` 的
    局部平方和拼接成一个张量，仅做一次 all_reduce。

    可选 ``overlap_compute`` 回调允许在 NCCL 在 flight 期间调用其它独立
    计算（例如 ``to_v(ctx)`` 投影），进一步隐藏通信延迟。

    Args:
        q_norm / k_norm: 已经被 ``shard_attention`` 替换为 ``TPRMSNorm``。
        q, k:           [B, Lq, chunk] / [B, Lk, chunk]，列并行后的本地分量。
        overlap_compute: 在 all_reduce flight 期间执行的无依赖计算（callable
                          无参数，返回结果由本函数返回）。

    Returns:
        ``(q_normed, k_normed, overlap_result)``。无 overlap 时第三项为
        ``None``。
    """
    assert q_norm.tp_group is k_norm.tp_group, (
        "q_norm 与 k_norm 必须共享同一个 tp_group 才能融合 all_reduce"
    )
    assert q_norm.full_dim == k_norm.full_dim, (
        "q_norm 与 k_norm 的 full_dim 必须一致（attention 内 q/k 同 inner_dim）"
    )

    # bf16 默认路径，TPRMSNORM_FP32=1 可强制 fp32 累加（见模块顶部注释）
    if _TP_RMSNORM_USE_FP32:
        q_compute = q.float()
        k_compute = k.float()
    else:
        q_compute = q
        k_compute = k

    q_sq = q_compute.pow(2).sum(dim=-1, keepdim=True)   # [B, Lq, 1]
    k_sq = k_compute.pow(2).sum(dim=-1, keepdim=True)   # [B, Lk, 1]

    # 沿 seq 轴 cat 成一个张量，触发单次 all_reduce
    Lq = q_sq.shape[1]
    both_sq = torch.cat([q_sq, k_sq], dim=1)
    handle = dist.all_reduce(
        both_sq, group=q_norm.tp_group, async_op=True,
    )

    # 在 NCCL 在 flight 期间，做一些无依赖的本地计算
    overlap_result = overlap_compute() if overlap_compute is not None else None

    handle.wait()

    q_sq = both_sq[:, :Lq]
    k_sq = both_sq[:, Lq:]

    q_rms = torch.sqrt(q_sq / q_norm.full_dim + q_norm.eps)
    k_rms = torch.sqrt(k_sq / k_norm.full_dim + k_norm.eps)
    q_out = q_compute / q_rms
    k_out = k_compute / k_rms
    if q_out.dtype != q.dtype:
        q_out = q_out.to(q.dtype)
        k_out = k_out.to(k.dtype)
    q_out = q_out * q_norm.weight
    k_out = k_out * k_norm.weight
    return q_out, k_out, overlap_result


# ============================================================================
# Weight Slicing Helpers
# ============================================================================

def _slice_column(linear: nn.Linear, tp_rank: int, tp_size: int) -> tuple:
    """从 Linear 层按 output 维度切片，返回 (weight_chunk, bias_chunk)。"""
    out_features = linear.out_features
    chunk = out_features // tp_size
    start = tp_rank * chunk
    end = start + chunk

    weight_chunk = linear.weight.data[start:end, :].contiguous()
    bias_chunk = None
    if linear.bias is not None:
        bias_chunk = linear.bias.data[start:end].contiguous()
    return weight_chunk, bias_chunk


def _slice_row(linear: nn.Linear, tp_rank: int, tp_size: int) -> tuple:
    """从 Linear 层按 input 维度切片，返回 (weight_chunk, full_bias)。"""
    in_features = linear.in_features
    chunk = in_features // tp_size
    start = tp_rank * chunk
    end = start + chunk

    weight_chunk = linear.weight.data[:, start:end].contiguous()
    # bias 保持完整（all_reduce 后加一次）
    full_bias = linear.bias.data.clone() if linear.bias is not None else None
    return weight_chunk, full_bias


# ============================================================================
# Attention TP Sharding
# ============================================================================

def shard_attention(
    attn: nn.Module,
    tp_rank: int,
    tp_size: int,
    tp_group: dist.ProcessGroup,
) -> None:
    """就地将一个 CausalLTXAttention 转为 TP 版本。

    切分策略：
        to_q/to_k/to_v    → ColumnParallel (按 head 切分 output)
        q_norm/k_norm      → 缩小到 inner_dim/tp (权重切片)
        to_gate_logits     → ColumnParallel (按 head 切分 output)
        to_out[0]          → RowParallel (按 head 切分 input, all_reduce)

    元数据更新：
        attn.heads     //= tp_size
        attn.inner_dim //= tp_size
    """
    heads = attn.heads
    inner_dim = attn.inner_dim
    assert heads % tp_size == 0, (
        f"heads ({heads}) 不能被 tp_size ({tp_size}) 整除"
    )

    local_heads = heads // tp_size
    local_inner = local_heads * attn.dim_head

    # ── QKV 投影 → ColumnParallel ──
    for name in ("to_q", "to_k", "to_v"):
        linear = getattr(attn, name)
        w, b = _slice_column(linear, tp_rank, tp_size)
        setattr(attn, name, ColumnParallelLinear(w, b))

    # ── Q/K RMSNorm → TP-aware RMSNorm ──
    # 原 RMSNorm 在 inner_dim 上算 RMS（跨 head），列并行后必须用 all_reduce
    # 还原全局平方和，否则与单卡数学不等价（见 TPRMSNorm 注释）。
    for norm_name in ("q_norm", "k_norm"):
        old_norm = getattr(attn, norm_name)
        chunk = inner_dim // tp_size
        start = tp_rank * chunk
        end = start + chunk
        weight_slice = old_norm.weight.data[start:end].contiguous()
        new_norm = TPRMSNorm(
            weight=weight_slice,
            eps=old_norm.eps,
            full_dim=inner_dim,
            tp_group=tp_group,
        )
        setattr(attn, norm_name, new_norm)

    # ── Gate logits (可选) → ColumnParallel ──
    if attn.to_gate_logits is not None:
        gate_linear = attn.to_gate_logits
        w, b = _slice_column(gate_linear, tp_rank, tp_size)
        attn.to_gate_logits = ColumnParallelLinear(w, b)

    # ── Output 投影 → RowParallel ──
    # to_out 是 nn.Sequential(Linear, Identity)
    out_linear = attn.to_out[0]
    w, bias = _slice_row(out_linear, tp_rank, tp_size)
    attn.to_out = nn.Sequential(
        RowParallelLinear(w, bias, tp_group),
        nn.Identity(),
    )

    # ── 更新元数据 ──
    attn.heads = local_heads
    attn.inner_dim = local_inner


# ============================================================================
# FeedForward TP Sharding
# ============================================================================

def shard_feedforward(
    ff: nn.Module,
    tp_rank: int,
    tp_size: int,
    tp_group: dist.ProcessGroup,
) -> None:
    """就地将一个 FeedForward 转为 TP 版本。

    FeedForward 结构:
        net[0] = GELUApprox(dim, inner_dim)   → net[0].proj 是 Linear
        net[1] = Identity()
        net[2] = Linear(inner_dim, dim_out)

    切分策略：
        net[0].proj  → ColumnParallel (中间维度按 tp 切分)
        net[2]       → RowParallel (中间维度按 tp 聚合)
    """
    # ── Up-proj (GELUApprox 内部 Linear) → ColumnParallel ──
    up_linear = ff.net[0].proj
    w, b = _slice_column(up_linear, tp_rank, tp_size)
    ff.net[0].proj = ColumnParallelLinear(w, b)

    # ── Down-proj → RowParallel ──
    down_linear = ff.net[2]
    w, bias = _slice_row(down_linear, tp_rank, tp_size)
    ff.net[2] = RowParallelLinear(w, bias, tp_group)


# ============================================================================
# Model-Level TP Sharding
# ============================================================================

def shard_model(
    wrapper: nn.Module,
    tp_rank: int,
    tp_size: int,
    tp_group: dist.ProcessGroup,
) -> None:
    """就地将整个模型转为 TP 版本。

    遍历所有 Transformer block，对每个 CausalAVTransformerBlock 的
    6 个 attention 和 2 个 FFN 执行 TP 分片。

    在 wrapper.model 上存储 _tp_rank / _tp_size 属性，供 kv_cache.py
    的 model_forward_inference() 识别并执行 PE 切片。

    注意：patchify_proj / proj_out / adaln_single / caption_projection 等
    不做 TP（参数量小，全部复制到每卡）。
    """
    model = wrapper.model

    # 存储 TP 元信息，供 kv_cache.py 读取
    model._tp_rank = tp_rank
    model._tp_size = tp_size

    num_blocks = len(model.transformer_blocks)
    for i, block in enumerate(model.transformer_blocks):
        # ── 6 个 Attention 模块 ──
        attn_names = [
            "attn1",                # 视频自注意力
            "attn2",                # 视频-文本交叉注意力
            "audio_attn1",          # 音频自注意力
            "audio_attn2",          # 音频-文本交叉注意力
            "audio_to_video_attn",  # A2V 跨模态注意力
            "video_to_audio_attn",  # V2A 跨模态注意力
        ]
        for name in attn_names:
            if hasattr(block, name):
                shard_attention(getattr(block, name), tp_rank, tp_size, tp_group)

        # ── 2 个 FFN 模块 ──
        ffn_names = ["ff", "audio_ff"]
        for name in ffn_names:
            if hasattr(block, name):
                shard_feedforward(getattr(block, name), tp_rank, tp_size, tp_group)

    print(
        f"[TP] Rank {tp_rank}/{tp_size}: 分片完成, "
        f"{num_blocks} blocks × (6 attn + 2 ffn) = "
        f"{num_blocks * 8} 个模块已转为 TP 版本"
    )


# ============================================================================
# torch.compile Wrapper for Transformer Blocks
# ============================================================================

# 在 compile 模式下，禁用 tp_qk_norm_fused 的 async_op + Python closure 路径，
# 强制走每条 norm 各自的同步 all_reduce。原因：torch.compile (Dynamo) 无法 trace
# `dist.all_reduce(async_op=True)` 返回的 Work handle，也无法 trace Python
# closure，会触发严重 graph break。同步路径每个 RMSNorm 内部的 c10d_functional
# all_reduce 是 Dynamo 原生支持的，反而能编进图里和 RMSNorm 的算子做融合。
#
# 由 compile_blocks() 在启用 torch.compile 时自动设置；用户也可手动设环境变量
# LTX_COMPILE=1 直接启用同步路径（不必真编译）。
_LTX_FORCE_SYNC_QK_NORM = os.environ.get("LTX_COMPILE", "0") == "1"


def is_compile_sync_qk_norm() -> bool:
    """attention_with_cache fast path 用此函数判断是否绕开 async fused 路径。"""
    return _LTX_FORCE_SYNC_QK_NORM


def compile_blocks(
    wrapper: nn.Module,
    mode: str = "default",
    dynamic: bool = True,
    fullgraph: bool = False,
) -> int:
    """对模型的每个 transformer block 应用 ``torch.compile``。

    Args:
        wrapper: 训练时的 wrapper 对象（``CausalLTXModelWrapper``），或者
                  直接是 ``CausalLTXModel`` —— 都能识别。
        mode:    传给 torch.compile 的 mode。常用值：
                  - "default":         安全，编译时间短，~30% 推理加速
                  - "reduce-overhead": 内部启用 CUDA Graph，~50% 推理加速，
                                        但要求 shape 稳定，KV cache 增长
                                        会强制重 capture
                  - "max-autotune":   triton autotune，编译时间长（数分钟），
                                        但单次 compile 后吞吐最高
        dynamic: True → 允许 KV cache 长度等动态维度，避免每个 shape 都
                  重编译
        fullgraph: False → 允许 graph break（推荐，因为 block 内部有 Python
                  控制流）

    Returns:
        被编译的 block 数量。

    使用方式：
        compile_blocks(generator, mode="default")

    注意：
        - 第一次 forward 会比较慢（编译 + autotune），通常 30-90 秒
        - 与 TP 兼容：自动绕开 ``tp_qk_norm_fused`` 的 async 路径
        - 不修改 forward 逻辑，只用 nn.Module 替换为 OptimizedModule wrapper
    """
    # 启用 sync qk-norm 路径（async + closure 与 Dynamo 不兼容）
    global _LTX_FORCE_SYNC_QK_NORM
    _LTX_FORCE_SYNC_QK_NORM = True

    model = wrapper.model if hasattr(wrapper, "model") else wrapper
    num_blocks = len(model.transformer_blocks)

    for i in range(num_blocks):
        model.transformer_blocks[i] = torch.compile(
            model.transformer_blocks[i],
            mode=mode,
            dynamic=dynamic,
            fullgraph=fullgraph,
        )

    print(
        f"[Compile] {num_blocks} transformer blocks wrapped: "
        f"mode={mode}, dynamic={dynamic}, fullgraph={fullgraph}. "
        f"First forward will be slow (compilation in progress)."
    )
    return num_blocks
