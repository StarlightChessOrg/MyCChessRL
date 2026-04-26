"""推理：两阶段合法掩码 + 贪心 / 采样。"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from mycchess_rl.chess.features import encode_model_planes
from mycchess_rl.chess.rationale import (
    POLICY_GRID_NUMEL,
    STM_VALUE_TERMINAL_DRAW,
    STM_VALUE_TERMINAL_LOSS,
)
from mycchess_rl.model import SuccessorPolicy, policy_temperature_scalar
from mycchess_rl.xqwl_state import XqwlGameState


def _encode_state_current_nchw(state: XqwlGameState, flist: dict[str, list[str]], device: torch.device) -> torch.Tensor:
    raw = state.board_view()
    legs = state.legal_moves_iccs_str()
    cur_chw = encode_model_planes(
        raw,
        state.red_to_move,
        legal_iccs=legs,
        in_check=state.in_check(),
        last_move=state.last_move_iccs,
    )
    cur_hwc = np.transpose(cur_chw, (1, 2, 0))
    return (
        torch.from_numpy(np.ascontiguousarray(np.expand_dims(cur_hwc, 0)))
        .float()
        .permute(0, 3, 1, 2)
        .to(device)
    )


@torch.no_grad()
def infer_greedy_move_string(
    state: XqwlGameState,
    model: SuccessorPolicy,
    device: torch.device,
    flist: dict[str, list[str]],
) -> str:
    legals_t = sorted(state.legal_moves_iccs())
    if not legals_t:
        raise RuntimeError("无合法着法")
    x_cur = _encode_state_current_nchw(state, flist, device)
    model.eval()
    feat = model._trunk_flat(x_cur)
    ls = model.head_src(feat)[0]
    src_mask = torch.zeros(POLICY_GRID_NUMEL, dtype=torch.bool, device=device)
    for x1, y1, _, _ in legals_t:
        src_mask[y1 * 9 + x1] = True
    ls = ls.masked_fill(~src_mask, -1e9)
    src_i = int(torch.argmax(ls).item())
    oh = torch.zeros(1, POLICY_GRID_NUMEL, device=device, dtype=feat.dtype)
    oh[0, src_i] = 1.0
    ld = model.head_dst(torch.cat([feat, oh], dim=1))[0]
    sx, sy = src_i % 9, src_i // 9
    dst_mask = torch.zeros(POLICY_GRID_NUMEL, dtype=torch.bool, device=device)
    for x1, y1, x2, y2 in legals_t:
        if (x1, y1) == (sx, sy):
            dst_mask[y2 * 9 + x2] = True
    ld = ld.masked_fill(~dst_mask, -1e9)
    dst_i = int(torch.argmax(ld).item())
    dx, dy = dst_i % 9, dst_i // 9
    return f"{sx}{sy}-{dx}{dy}"


def _scalar_value_from_logits_v(logits_v: torch.Tensor) -> torch.Tensor:
    """(3,) 或 (B,3) → 标量或 (B,) 行棋方价值期望。"""
    pv = torch.softmax(logits_v.float(), dim=-1)
    w = torch.tensor([3.0, 1.0, -3.0], device=logits_v.device, dtype=pv.dtype)
    if pv.dim() == 1:
        return (pv * w).sum()
    return (pv * w).sum(dim=-1)


@torch.no_grad()
def eval_value_stm(
    state: XqwlGameState,
    model: SuccessorPolicy,
    device: torch.device,
    flist: dict[str, list[str]],
) -> float:
    if not state.legal_moves_iccs_str():
        t, r = state.terminal()
        if t and r == "checkmate":
            return float(STM_VALUE_TERMINAL_LOSS)
        return float(STM_VALUE_TERMINAL_DRAW)
    x_cur = _encode_state_current_nchw(state, flist, device)
    model.eval()
    feat = model._trunk_flat(x_cur)
    logits_v = model.value_head(feat)[0]
    return float(_scalar_value_from_logits_v(logits_v).item())


def batched_encode_roots(
    states: list[XqwlGameState],
    flist: dict[str, list[str]],
    device: torch.device,
    *,
    encode_workers: int = 1,
) -> torch.Tensor:
    if not states:
        return torch.zeros(0, device=device)
    from mycchess_rl.encode_parallel import encode_states_parallel, resolve_encode_workers

    w = resolve_encode_workers(encode_workers, len(states))
    if w <= 1:
        xs = [_encode_state_current_nchw(g, flist, device) for g in states]
        return torch.cat(xs, dim=0)
    chw = encode_states_parallel(states, w)
    return torch.from_numpy(np.ascontiguousarray(chw)).to(device, non_blocking=True)


@torch.no_grad()
def batched_sample_moves_masked(
    states: list[XqwlGameState],
    model: SuccessorPolicy,
    device: torch.device,
    flist: dict[str, list[str]],
    *,
    policy_temperature: float = 1.0,
    generator: torch.Generator | None = None,
    encode_workers: int = 1,
) -> list[str]:
    """整批 trunk + 批量 ``head_dst``，避免逐环境 B 次小矩阵乘。"""
    T = policy_temperature_scalar(policy_temperature)
    model.eval()
    B = len(states)
    if B == 0:
        return []

    has_legal_list = [bool(st.legal_moves_iccs_str()) for st in states]
    has_legal = torch.tensor(has_legal_list, dtype=torch.bool, device=device)

    xb = batched_encode_roots(states, flist, device, encode_workers=encode_workers)
    feat_b = model._trunk_flat(xb)
    ls_b = model.head_src(feat_b)
    scaled_s = ls_b / T

    src_ok = torch.zeros(B, POLICY_GRID_NUMEL, dtype=torch.bool, device=device)
    for bi, st in enumerate(states):
        if not has_legal_list[bi]:
            continue
        for x1, y1, _, _ in sorted(st.legal_moves_iccs()):
            src_ok[bi, y1 * 9 + x1] = True

    logits_s = scaled_s.masked_fill(~src_ok, -1e9)
    safe_s = torch.full_like(logits_s, -1e9)
    safe_s[:, 0] = 0.0
    logits_s = torch.where(has_legal.unsqueeze(1), logits_s, safe_s)

    p_src = torch.softmax(logits_s, dim=1)
    src_idx = torch.multinomial(p_src, 1, generator=generator).squeeze(1)

    oh_s = torch.zeros(B, POLICY_GRID_NUMEL, device=device, dtype=feat_b.dtype)
    oh_s.scatter_(1, src_idx.unsqueeze(1), 1.0)
    logits_d = model.head_dst(torch.cat([feat_b, oh_s], dim=1))
    scaled_d = logits_d / T

    dst_ok = torch.zeros(B, POLICY_GRID_NUMEL, dtype=torch.bool, device=device)
    sx_all = (src_idx % 9).tolist()
    sy_all = (src_idx // 9).tolist()
    for bi, st in enumerate(states):
        if not has_legal_list[bi]:
            continue
        sx, sy = int(sx_all[bi]), int(sy_all[bi])
        for x1, y1, x2, y2 in sorted(st.legal_moves_iccs()):
            if (x1, y1) == (sx, sy):
                dst_ok[bi, y2 * 9 + x2] = True

    logits_d_m = scaled_d.masked_fill(~dst_ok, -1e9)
    safe_d = torch.full_like(logits_d_m, -1e9)
    safe_d[:, 0] = 0.0
    logits_d_m = torch.where(has_legal.unsqueeze(1), logits_d_m, safe_d)

    p_dst = torch.softmax(logits_d_m, dim=1)
    dst_idx = torch.multinomial(p_dst, 1, generator=generator).squeeze(1)

    sx_np = (src_idx % 9).detach().cpu().numpy()
    sy_np = (src_idx // 9).detach().cpu().numpy()
    dx_np = (dst_idx % 9).detach().cpu().numpy()
    dy_np = (dst_idx // 9).detach().cpu().numpy()

    out_moves: list[str] = []
    for bi in range(B):
        if not has_legal_list[bi]:
            out_moves.append("")
        else:
            out_moves.append(
                f"{int(sx_np[bi])}{int(sy_np[bi])}-{int(dx_np[bi])}{int(dy_np[bi])}"
            )
    return out_moves


@torch.no_grad()
def two_stage_logprob_on_move(
    state: XqwlGameState,
    model: SuccessorPolicy,
    device: torch.device,
    flist: dict[str, list[str]],
    iccs: str,
    *,
    policy_temperature: float = 1.0,
    feat_1_row: torch.Tensor | None = None,
) -> torch.Tensor:
    T = policy_temperature_scalar(policy_temperature)
    x1, y1, x2, y2 = int(iccs[0]), int(iccs[1]), int(iccs[3]), int(iccs[4])
    src_i = y1 * 9 + x1
    dst_i = y2 * 9 + x2
    legals_t = sorted(state.legal_moves_iccs())
    src_mask = torch.zeros(POLICY_GRID_NUMEL, dtype=torch.bool, device=device)
    for a, b, _, _ in legals_t:
        src_mask[b * 9 + a] = True
    model.eval()
    if feat_1_row is None:
        x_cur = _encode_state_current_nchw(state, flist, device)
        feat = model._trunk_flat(x_cur)
    else:
        feat = feat_1_row
    ls = model.head_src(feat)[0]
    log_p_src = torch.log_softmax((ls / T).masked_fill(~src_mask, -1e9), dim=0)
    oh = torch.zeros(1, POLICY_GRID_NUMEL, device=device, dtype=feat.dtype)
    oh[0, src_i] = 1.0
    ld = model.head_dst(torch.cat([feat, oh], dim=1))[0]
    dst_mask = torch.zeros(POLICY_GRID_NUMEL, dtype=torch.bool, device=device)
    for a, b, c, d in legals_t:
        if (a, b) == (x1, y1):
            dst_mask[d * 9 + c] = True
    log_p_dst = torch.log_softmax((ld / T).masked_fill(~dst_mask, -1e9), dim=0)
    return log_p_src[src_i] + log_p_dst[dst_i]


@torch.no_grad()
def batched_two_stage_logprob_on_moves(
    obs_list: list[XqwlGameState],
    mv_list: list[str],
    feat_b: torch.Tensor,
    model: SuccessorPolicy,
    device: torch.device,
    *,
    policy_temperature: float = 1.0,
) -> torch.Tensor:
    """
    与逐条 ``two_stage_logprob_on_move`` 等价，但 ``head_src`` / ``head_dst`` 为整批矩阵算子，
    避免数万次 Python 循环导致长时间无日志、极慢。
    """
    T = policy_temperature_scalar(policy_temperature)
    B = feat_b.shape[0]
    if B == 0:
        return torch.zeros(0, device=device, dtype=torch.float32)

    logits_s = model.head_src(feat_b)
    src_idx = torch.zeros(B, dtype=torch.long, device=device)
    dst_idx = torch.zeros(B, dtype=torch.long, device=device)
    src_ok = torch.zeros(B, POLICY_GRID_NUMEL, dtype=torch.bool, device=device)
    dst_ok = torch.zeros(B, POLICY_GRID_NUMEL, dtype=torch.bool, device=device)
    oh_s = torch.zeros(B, POLICY_GRID_NUMEL, device=device, dtype=feat_b.dtype)

    for i, (g, mv) in enumerate(zip(obs_list, mv_list)):
        x1, y1, x2, y2 = int(mv[0]), int(mv[1]), int(mv[3]), int(mv[4])
        src_idx[i] = y1 * 9 + x1
        dst_idx[i] = y2 * 9 + x2
        oh_s[i, src_idx[i]] = 1.0
        for a, b, _, _ in sorted(g.legal_moves_iccs()):
            src_ok[i, b * 9 + a] = True
        for a, b, c, d in sorted(g.legal_moves_iccs()):
            if (a, b) == (x1, y1):
                dst_ok[i, d * 9 + c] = True

    scaled_s = logits_s / T
    log_p_s = F.log_softmax(scaled_s.masked_fill(~src_ok, -1e9), dim=1)
    log_p_s = log_p_s.gather(1, src_idx.unsqueeze(1)).squeeze(1)

    logits_d = model.head_dst(torch.cat([feat_b, oh_s], dim=1))
    scaled_d = logits_d / T
    log_p_d = F.log_softmax(scaled_d.masked_fill(~dst_ok, -1e9), dim=1)
    log_p_d = log_p_d.gather(1, dst_idx.unsqueeze(1)).squeeze(1)
    return log_p_s + log_p_d


@torch.no_grad()
def batched_value_expectation(
    states: list[XqwlGameState],
    model: SuccessorPolicy,
    device: torch.device,
    flist: dict[str, list[str]],
    *,
    encode_workers: int = 1,
) -> torch.Tensor:
    """单次 ``_trunk_flat`` 批前向，避免逐环境调用 ``eval_value_stm`` 导致 GPU 吃不饱。"""
    n = len(states)
    if n == 0:
        return torch.zeros(0, device=device, dtype=torch.float32)
    out = torch.empty(n, device=device, dtype=torch.float32)
    active: list[int] = []
    active_states: list[XqwlGameState] = []
    for i, g in enumerate(states):
        if not g.legal_moves_iccs_str():
            t, r = g.terminal()
            if t and r == "checkmate":
                out[i] = float(STM_VALUE_TERMINAL_LOSS)
            else:
                out[i] = float(STM_VALUE_TERMINAL_DRAW)
        else:
            active.append(i)
            active_states.append(g)
    if active_states:
        model.eval()
        xb = batched_encode_roots(active_states, flist, device, encode_workers=encode_workers)
        feat = model._trunk_flat(xb)
        logits_v = model.value_head(feat)
        vals = _scalar_value_from_logits_v(logits_v)
        for j, idx in enumerate(active):
            out[idx] = vals[j]
    return out
