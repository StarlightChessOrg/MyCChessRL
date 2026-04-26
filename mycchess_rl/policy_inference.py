"""推理：两阶段合法掩码 + 贪心 / 采样。"""
from __future__ import annotations

import numpy as np
import torch

from mycchess_rl.chess.features import encode_model_planes
from mycchess_rl.chess.rationale import (
    POLICY_GRID_NUMEL,
    STM_VALUE_TERMINAL_DRAW,
    STM_VALUE_TERMINAL_LOSS,
    stm_value_expectation_from_win_draw_loss_probs,
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
    pv = torch.softmax(logits_v.float(), dim=0)
    return float(
        stm_value_expectation_from_win_draw_loss_probs(
            float(pv[0].item()), float(pv[1].item()), float(pv[2].item())
        )
    )


def batched_encode_roots(
    states: list[XqwlGameState],
    flist: dict[str, list[str]],
    device: torch.device,
) -> torch.Tensor:
    xs = [_encode_state_current_nchw(g, flist, device) for g in states]
    return torch.cat(xs, dim=0) if xs else torch.zeros(0, device=device)


@torch.no_grad()
def batched_sample_moves_masked(
    states: list[XqwlGameState],
    model: SuccessorPolicy,
    device: torch.device,
    flist: dict[str, list[str]],
    *,
    policy_temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> list[str]:
    T = policy_temperature_scalar(policy_temperature)
    model.eval()
    xb = batched_encode_roots(states, flist, device)
    if xb.shape[0] == 0:
        return []
    feat_b = model._trunk_flat(xb)
    ls_b = model.head_src(feat_b)
    out_moves: list[str] = []
    for bi, st in enumerate(states):
        legals_t = sorted(st.legal_moves_iccs())
        if not legals_t:
            out_moves.append("")
            continue
        ls = ls_b[bi]
        src_mask = torch.zeros(POLICY_GRID_NUMEL, dtype=torch.bool, device=device)
        for x1, y1, _, _ in legals_t:
            src_mask[y1 * 9 + x1] = True
        p_src = torch.softmax((ls / T).masked_fill(~src_mask, -1e9), dim=0)
        src_i = int(torch.multinomial(p_src, 1, generator=generator).item())
        sx, sy = src_i % 9, src_i // 9
        oh = torch.zeros(1, POLICY_GRID_NUMEL, device=device, dtype=feat_b.dtype)
        oh[0, src_i] = 1.0
        ld = model.head_dst(torch.cat([feat_b[bi : bi + 1], oh], dim=1))[0]
        dst_mask = torch.zeros(POLICY_GRID_NUMEL, dtype=torch.bool, device=device)
        for x1, y1, x2, y2 in legals_t:
            if (x1, y1) == (sx, sy):
                dst_mask[y2 * 9 + x2] = True
        p_dst = torch.softmax((ld / T).masked_fill(~dst_mask, -1e9), dim=0)
        dst_i = int(torch.multinomial(p_dst, 1, generator=generator).item())
        dx, dy = dst_i % 9, dst_i // 9
        out_moves.append(f"{sx}{sy}-{dx}{dy}")
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
) -> torch.Tensor:
    T = policy_temperature_scalar(policy_temperature)
    x1, y1, x2, y2 = int(iccs[0]), int(iccs[1]), int(iccs[3]), int(iccs[4])
    src_i = y1 * 9 + x1
    dst_i = y2 * 9 + x2
    legals_t = sorted(state.legal_moves_iccs())
    src_mask = torch.zeros(POLICY_GRID_NUMEL, dtype=torch.bool, device=device)
    for a, b, _, _ in legals_t:
        src_mask[b * 9 + a] = True
    x_cur = _encode_state_current_nchw(state, flist, device)
    model.eval()
    feat = model._trunk_flat(x_cur)
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
def batched_value_expectation(
    states: list[XqwlGameState],
    model: SuccessorPolicy,
    device: torch.device,
    flist: dict[str, list[str]],
) -> torch.Tensor:
    xs = [eval_value_stm(g, model, device, flist) for g in states]
    return torch.tensor(xs, device=device, dtype=torch.float32)
