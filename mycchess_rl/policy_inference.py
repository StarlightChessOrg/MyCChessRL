"""推理：两阶段合法掩码 + 贪心 / 采样。"""
from __future__ import annotations

import threading
import numpy as np
import torch
import torch.nn.functional as F

from mycchess_rl.chess.features import encode_model_planes
from mycchess_rl.iccs_util import parse_move_squares
from mycchess_rl.chess.rationale import (
    POLICY_GRID_NUMEL,
    STM_VALUE_TERMINAL_DRAW,
    STM_VALUE_TERMINAL_LOSS,
)
from mycchess_rl.model import SuccessorPolicy, policy_temperature_scalar
from mycchess_rl.xqwl_state import XqwlGameState


def _legal_moves_sorted_tuples(st: XqwlGameState) -> list[tuple[int, int, int, int]]:
    """单次 ``legal_moves_iccs_str`` + Python 解析，避免 ``legal_moves_iccs`` 再调 C++ 生成一遍。"""
    ms = st.legal_moves_iccs_str()
    if not ms:
        return []
    return sorted(parse_move_squares(m) for m in ms)


def _xb_from_states_two_group_encode(states: list[XqwlGameState], device: torch.device) -> torch.Tensor:
    """两组局面并行 CPU 编码（主线程一半 + 守护线程一半），``np.concatenate`` 后一次 H2D。

    ``model.eval()`` 下 trunk 按行独立，与整批一次编码再 trunk 数值一致；用于叠合 CPU 准备空档。
    """
    from mycchess_rl.encode_parallel import encode_states_inline

    b = len(states)
    if b <= 1:
        chw = encode_states_inline(states)
        return torch.from_numpy(np.ascontiguousarray(chw)).to(device, non_blocking=True)
    mid = (b + 1) // 2
    s0, s1 = states[:mid], states[mid:]
    chw1_box: list[np.ndarray] = []

    def _enc_second_half() -> None:
        chw1_box.append(encode_states_inline(s1))

    th = threading.Thread(target=_enc_second_half, daemon=True)
    th.start()
    chw0 = encode_states_inline(s0)
    th.join()
    chw = np.concatenate([chw0, chw1_box[0]], axis=0)
    return torch.from_numpy(np.ascontiguousarray(chw)).to(device, non_blocking=True)


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
    encode_backend: str = "inline",
) -> torch.Tensor:
    """根平面批编码。默认 ``inline``：主进程批量 numpy + **一次** H2D，最适合轻量 14 平面与 rollout 高频小批。"""
    if not states:
        return torch.zeros(0, device=device)
    from mycchess_rl.encode_parallel import (
        encode_states_inline,
        encode_states_process_pool,
        encode_states_thread_pool,
        resolve_encode_workers,
    )

    backend = (encode_backend or "inline").strip().lower()
    if backend not in ("inline", "thread", "process"):
        backend = "inline"

    if backend == "inline":
        chw = encode_states_inline(states)
        return torch.from_numpy(np.ascontiguousarray(chw)).to(device, non_blocking=True)

    w = resolve_encode_workers(encode_workers, len(states))
    if w <= 1:
        chw = encode_states_inline(states)
        return torch.from_numpy(np.ascontiguousarray(chw)).to(device, non_blocking=True)
    if backend == "thread":
        chw = encode_states_thread_pool(states, w)
    else:
        chw = encode_states_process_pool(states, w)
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
    encode_backend: str = "inline",
    rollout_pipeline_groups: int = 1,
) -> list[str]:
    """整批 trunk + 批量 ``head_dst``，避免逐环境 B 次小矩阵乘。

    ``rollout_pipeline_groups>=2`` 时启用两组并行 CPU 编码（与单次 trunk 叠合准备空档）；
    该路径固定使用主进程 ``encode_states_inline``，忽略 ``encode_backend`` 的 thread/process。
    """
    T = policy_temperature_scalar(policy_temperature)
    model.eval()
    B = len(states)
    if B == 0:
        return []

    legals_t = [_legal_moves_sorted_tuples(st) for st in states]
    has_legal_list = [bool(lt) for lt in legals_t]
    has_legal = torch.tensor(has_legal_list, dtype=torch.bool, device=device)

    if int(rollout_pipeline_groups) >= 2 and B >= 2:
        xb = _xb_from_states_two_group_encode(states, device)
    else:
        xb = batched_encode_roots(
            states,
            flist,
            device,
            encode_workers=encode_workers,
            encode_backend=encode_backend,
        )
    feat_b = model._trunk_flat(xb)
    ls_b = model.head_src(feat_b)
    scaled_s = ls_b / T

    src_ok = torch.zeros(B, POLICY_GRID_NUMEL, dtype=torch.bool, device=device)
    for bi, lt in enumerate(legals_t):
        if not lt:
            continue
        for x1, y1, _, _ in lt:
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
    for bi, lt in enumerate(legals_t):
        if not lt:
            continue
        sx, sy = int(sx_all[bi]), int(sy_all[bi])
        for x1, y1, x2, y2 in lt:
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


def policy_legal_masks_for_batch(
    obs_list: list[XqwlGameState],
    mv_list: list[str],
    device: torch.device,
    *,
    oh_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """整批两阶段合法掩码：与 ``batched_sample_moves_masked`` 一样用 ``_legal_moves_sorted_tuples``。

    返回 ``(src_idx, dst_idx, oh_s, src_ok, dst_ok)``；``src_ok``/``dst_ok`` 供 PPO 的 masked
    ``log_softmax`` 与 ``old_logp`` 一致（分母只在合法着法上归一化）。
    """
    b = len(obs_list)
    if b == 0:
        zl = torch.zeros(0, dtype=torch.long, device=device)
        zb = torch.zeros(0, POLICY_GRID_NUMEL, dtype=torch.bool, device=device)
        zh = torch.zeros(0, POLICY_GRID_NUMEL, dtype=oh_dtype, device=device)
        return zl, zl, zh, zb, zb
    src_idx = torch.zeros(b, dtype=torch.long, device=device)
    dst_idx = torch.zeros(b, dtype=torch.long, device=device)
    src_ok = torch.zeros(b, POLICY_GRID_NUMEL, dtype=torch.bool, device=device)
    dst_ok = torch.zeros(b, POLICY_GRID_NUMEL, dtype=torch.bool, device=device)
    oh_s = torch.zeros(b, POLICY_GRID_NUMEL, device=device, dtype=oh_dtype)

    for i, (g, mv) in enumerate(zip(obs_list, mv_list)):
        x1, y1, x2, y2 = int(mv[0]), int(mv[1]), int(mv[3]), int(mv[4])
        si = y1 * 9 + x1
        di = y2 * 9 + x2
        src_idx[i] = si
        dst_idx[i] = di
        oh_s[i, si] = 1.0
        lt = _legal_moves_sorted_tuples(g)
        for a, b, _, _ in lt:
            src_ok[i, b * 9 + a] = True
        for a, b, c, d in lt:
            if (a, b) == (x1, y1):
                dst_ok[i, d * 9 + c] = True
    return src_idx, dst_idx, oh_s, src_ok, dst_ok


@torch.no_grad()
def batched_two_stage_logprob_on_moves(
    obs_list: list[XqwlGameState],
    mv_list: list[str],
    feat_b: torch.Tensor,
    model: SuccessorPolicy,
    device: torch.device,
    *,
    policy_temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    与逐条 ``two_stage_logprob_on_move`` 等价，但 ``head_src`` / ``head_dst`` 为整批矩阵算子，
    避免数万次 Python 循环导致长时间无日志、极慢。

    返回 ``(log_p, src_ok, dst_ok)``，后两者与 PPO 损失里应使用的合法掩码一致。
    """
    T = policy_temperature_scalar(policy_temperature)
    B = feat_b.shape[0]
    if B == 0:
        zt = torch.zeros(0, device=device, dtype=torch.float32)
        zm = torch.zeros(0, POLICY_GRID_NUMEL, dtype=torch.bool, device=device)
        return zt, zm, zm

    src_idx, dst_idx, oh_s, src_ok, dst_ok = policy_legal_masks_for_batch(
        obs_list, mv_list, device, oh_dtype=feat_b.dtype
    )
    logits_s = model.head_src(feat_b)
    scaled_s = logits_s / T
    log_p_s = F.log_softmax(scaled_s.masked_fill(~src_ok, -1e9), dim=1)
    log_p_s = log_p_s.gather(1, src_idx.unsqueeze(1)).squeeze(1)

    logits_d = model.head_dst(torch.cat([feat_b, oh_s], dim=1))
    scaled_d = logits_d / T
    log_p_d = F.log_softmax(scaled_d.masked_fill(~dst_ok, -1e9), dim=1)
    log_p_d = log_p_d.gather(1, dst_idx.unsqueeze(1)).squeeze(1)
    return log_p_s + log_p_d, src_ok, dst_ok


@torch.no_grad()
def batched_value_expectation(
    states: list[XqwlGameState],
    model: SuccessorPolicy,
    device: torch.device,
    flist: dict[str, list[str]],
    *,
    encode_workers: int = 1,
    encode_backend: str = "inline",
    rollout_pipeline_groups: int = 1,
) -> torch.Tensor:
    """单次 ``_trunk_flat`` 批前向，避免逐环境调用 ``eval_value_stm`` 导致 GPU 吃不饱。

    ``rollout_pipeline_groups>=2`` 且存在需网络估值的活跃局面时，对活跃子批做两组并行编码（同采样路径）。
    """
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
        na = len(active_states)
        if int(rollout_pipeline_groups) >= 2 and na >= 2:
            xb = _xb_from_states_two_group_encode(active_states, device)
        else:
            xb = batched_encode_roots(
                active_states,
                flist,
                device,
                encode_workers=encode_workers,
                encode_backend=encode_backend,
            )
        feat = model._trunk_flat(xb)
        logits_v = model.value_head(feat)
        vals = _scalar_value_from_logits_v(logits_v)
        for j, idx in enumerate(active):
            out[idx] = vals[j]
    return out
