# Copyright (c) 2024, Albert Gu and Tri Dao.
"""Minimal implementation of SSD.

This is the same as Listing 1 from the paper.
"""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


def segsum_unstable(x):
    """Naive segment sum calculation."""
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum

def segsum(x):
    """
        More stable segment sum calculation.
        实现了每个chunk内部序列求累加和
        in : (b h c l)
        out : (b h c l l)
        在(l l)中 (i,j)表示从j到i的累计和
    """
    T = x.size(-1) # 获取最后一个维度大小l == t
    x = repeat(x, "... d -> ... d e", e=T) # (b h c t t)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1) # (t t)下三角矩阵对角线下为true
    x = x.masked_fill(~mask, 0) # (b h t t) 将~mask中为True的地方填充为0
    x_segsum = torch.cumsum(x, dim=-2) # (b h c t t) 沿倒数第二个维度(矩阵行)累积和
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0) # (t t)下三角且对角线及以下为true
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum # (b h c l l)

def ssd_minimal_discrete(X, A, B, C, block_len, initial_states=None):
    """
    Arguments:
        X: (batch, length, n_heads, d_head) b l h p
        A: (batch, length, n_heads) b l h
        B: (batch, length, n_heads, d_state) b l h n
        C: (batch, length, n_heads, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
    说明:
        M = L * CB^T
        L(i,j) = a(i) * ... * a(j + 1) with i > j 
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0

    # Rearrange into blocks/chunks 
    # 把length拆分成block_len个chunk，每个chunk长block_len 其他维度保持不变 (b c l h n)
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1) # (b h c l)求每个chunk内l的累积和

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A)) # (b h c l l) 返回矩阵中每个值的exp l l构成mask矩阵，来缩放注意力分数矩阵
    # 张量收缩操作
    # s == l 表示token序列间的关系
    #  C B在n上点积 接着与L在s点积 接着与X在s点积 (在->没有出现的维度上矩阵乘法（点积）)
    Y_diag  = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X) # (b c l h p)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    # (b h c l) 计算累积和A的当前位置距离最后一个位置的差值，也就是A中这段区间的和 
    # -1:取最后维度的最后一个值但保留维度 -1不保留维度
    # decay_states表示每个位置到最后一个位置的衰减因子
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum)) # (b h c l)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X) # (b c h p n)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1) # (b c h p n + 1)
    # F.pad(t,(1,0))表示在最后一个维度左侧填充1个值,右侧填充0个值
    # F.pad提取每个累计和chunk最后一个元素，相当于得到每个chunk的l的和
    # 再进行segsum,得到每个chunk之间的衰减因子，同样也是越远越大，影响越小
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)))) # (b h c) -> (b h c c)
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states) # (b z h p n)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum) # (b h c l)
    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out) # (b c l h p)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag+Y_off, "b c l h p -> b (c l) h p") # (b l h p)
    return Y, final_state


# Simple test
def test_correctness():
    torch.manual_seed(42)

    ## Dimensions
    # Denoted (B, T, Q, D, P) in the paper
    batch, seqlen, chunk_size, dim, headdim = 1, 2048, 64, 2048, 64
    nheads = dim // headdim  # (H) in the paper
    ngroups = 1 # (G) in the paper
    dstate = 64  # (N) in the paper
    dtype = torch.float32
    device = "cuda"

    x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
    dt = F.softplus(torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=device) - 4).requires_grad_()
    A = (-torch.exp(torch.rand(nheads, dtype=torch.float32, device=device))).requires_grad_()
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device)
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device)
    D = torch.randn(nheads, dtype=dtype, device=device)

    # Comparing fused version and minimal version
    y = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None)
    y_min, _ = ssd_minimal_discrete(x*dt.unsqueeze(-1), A*dt, B, C, chunk_size)
