"""推理：在有序合法着法列表上的联合 softmax + 标量价值。"""
from __future__ import annotations

import threading

import numpy as np
import torch
import torch.nn.functional as F

from mycchess_rl.chess.features import encode_model_planes
from mycchess_rl.chess.rationale import STM_VALUE_TERMINAL_DRAW, STM_VALUE_TERMINAL_LOSS
from mycchess_rl.model import JointPolicyValueNet, policy_temperature_scalar
from mycchess_rl.xqwl_state import XqwlGameState


def sorted_legal_iccs(state: XqwlGameState) -> list[str]:
    return sorted(state.legal_moves_iccs_str())


def joint_legal_mask_and_action_index(
    obs_list: list[XqwlGameState],
    mv_list: list[str],
    device: torch.device,
    policy_max_legal: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``legal_mask[b,k]=True`` 当且仅当 ``k <`` 该局面的合法着法数；``action_idx`` 为 ``mv`` 在 ``sorted_legal_iccs`` 中的下标。"""
    b = len(obs_list)
    if b == 0:
        z = torch.zeros(0, policy_max_legal, dtype=torch.bool, device=device)
        zi = torch.zeros(0, dtype=torch.long, device=device)
        return z, zi
    mask = torch.zeros(b, policy_max_legal, dtype=torch.bool, device=device)
    idx = torch.zeros(b, dtype=torch.long, device=device)
    for i, (g, mv) in enumerate(zip(obs_list, mv_list)):
        legs = sorted_legal_iccs(g)
        L = len(legs)
        if L == 0:
            continue
        if L > policy_max_legal:
            raise RuntimeError(
                f"合法着法数 {L} 超过 policy_max_legal={policy_max_legal}，请增大 rationale.POLICY_MAX_LEGAL_MOVES 并重训"
            )
        mask[i, :L] = True
        try:
            idx[i] = legs.index(mv)
        except ValueError:
            idx[i] = 0
    return mask, idx


def _xb_from_states_two_group_encode(states: list[XqwlGameState], device: torch.device) -> torch.Tensor:
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
def infer_joint_policy_prior_and_value(
    state: XqwlGameState,
    model: JointPolicyValueNet,
    device: torch.device,
    flist: dict[str, list[str]],
    *,
    policy_temperature: float = 1.0,
) -> tuple[list[str], np.ndarray, float]:
    """
    返回 ``(sorted_legal_iccs, prob[L], v_norm)``。
    ``prob`` 在合法着法上归一化和为 1；``v_norm = v_net / value_scale``，约 ``(-1,1)``，表示**当前行棋方**的网络价值（与 SL 训练一致）。
    """
    legs = sorted_legal_iccs(state)
    if not legs:
        return [], np.zeros((0,), dtype=np.float64), 0.0
    M = model.policy_max_legal
    if len(legs) > M:
        raise RuntimeError(f"合法着法数 {len(legs)} > policy_max_legal={M}")
    x_cur = _encode_state_current_nchw(state, flist, device)
    model.eval()
    T = policy_temperature_scalar(policy_temperature)
    logits_m, v = model(x_cur)
    mask = torch.zeros(1, M, dtype=torch.bool, device=device)
    mask[0, : len(legs)] = True
    scaled = (logits_m / T).masked_fill(~mask, -1e9)
    p = torch.softmax(scaled, dim=1)[0, : len(legs)].detach().float().cpu().numpy()
    s = float(p.sum())
    if s > 0:
        p = p / s
    else:
        p = np.full(len(legs), 1.0 / len(legs), dtype=np.float64)
    vs = max(float(model.value_scale), 1e-6)
    v_norm = float(v.item()) / vs
    v_norm = max(-1.0, min(1.0, v_norm))
    return legs, p.astype(np.float64, copy=False), v_norm


@torch.no_grad()
def infer_greedy_move_string(
    state: XqwlGameState,
    model: JointPolicyValueNet,
    device: torch.device,
    flist: dict[str, list[str]],
) -> str:
    legs = sorted_legal_iccs(state)
    if not legs:
        raise RuntimeError("无合法着法")
    M = model.policy_max_legal
    if len(legs) > M:
        raise RuntimeError(f"合法着法数 {len(legs)} > policy_max_legal={M}")
    x_cur = _encode_state_current_nchw(state, flist, device)
    model.eval()
    logits_m, _ = model(x_cur)
    T = policy_temperature_scalar(1.0)
    mask = torch.zeros(1, M, dtype=torch.bool, device=device)
    mask[0, : len(legs)] = True
    scaled = (logits_m / T).masked_fill(~mask, -1e9)
    k = int(torch.argmax(scaled, dim=1).item())
    return legs[k]


@torch.no_grad()
def eval_value_stm(
    state: XqwlGameState,
    model: JointPolicyValueNet,
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
    _, v = model(x_cur)
    return float(v.item())


def batched_encode_roots(
    states: list[XqwlGameState],
    flist: dict[str, list[str]],
    device: torch.device,
    *,
    encode_workers: int = 1,
    encode_backend: str = "inline",
) -> torch.Tensor:
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
    model: JointPolicyValueNet,
    device: torch.device,
    flist: dict[str, list[str]],
    *,
    policy_temperature: float = 1.0,
    generator: torch.Generator | None = None,
    encode_workers: int = 1,
    encode_backend: str = "inline",
    rollout_pipeline_groups: int = 1,
) -> list[str]:
    T = policy_temperature_scalar(policy_temperature)
    model.eval()
    B = len(states)
    if B == 0:
        return []

    legals_str: list[list[str]] = [sorted_legal_iccs(st) for st in states]
    has_legal_list = [bool(ls) for ls in legals_str]
    has_legal = torch.tensor(has_legal_list, dtype=torch.bool, device=device)
    M = model.policy_max_legal

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
    logits_m, _ = model.forward_heads_from_feat(feat_b)
    scaled = logits_m / T

    mask = torch.zeros(B, M, dtype=torch.bool, device=device)
    for bi, ls in enumerate(legals_str):
        L = len(ls)
        if L == 0:
            continue
        if L > M:
            raise RuntimeError(
                f"环境 {bi} 合法着法数 {L} 超过 policy_max_legal={M}"
            )
        mask[bi, :L] = True

    logits_m = scaled.masked_fill(~mask, -1e9)
    safe = torch.full_like(logits_m, -1e9)
    safe[:, 0] = 0.0
    logits_m = torch.where(has_legal.unsqueeze(1), logits_m, safe)
    p = torch.softmax(logits_m, dim=1)
    idx = torch.multinomial(p, 1, generator=generator).squeeze(1)

    out_moves: list[str] = []
    for bi in range(B):
        if not has_legal_list[bi]:
            out_moves.append("")
        else:
            k = int(idx[bi].item())
            out_moves.append(legals_str[bi][k])
    return out_moves


@torch.no_grad()
def batched_joint_logprob_on_moves(
    obs_list: list[XqwlGameState],
    mv_list: list[str],
    feat_b: torch.Tensor,
    model: JointPolicyValueNet,
    device: torch.device,
    *,
    policy_temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 ``(log_p, legal_mask, action_idx)``，与 PPO 损失输入一致。"""
    T = policy_temperature_scalar(policy_temperature)
    B = feat_b.shape[0]
    if B == 0:
        zt = torch.zeros(0, device=device, dtype=torch.float32)
        zm = torch.zeros(0, model.policy_max_legal, dtype=torch.bool, device=device)
        zi = torch.zeros(0, dtype=torch.long, device=device)
        return zt, zm, zi

    mask, action_idx = joint_legal_mask_and_action_index(
        obs_list, mv_list, device, model.policy_max_legal
    )
    logits_m, _ = model.forward_heads_from_feat(feat_b)
    scaled = logits_m / T
    scaled = scaled.masked_fill(~mask, -1e9)
    log_p = F.log_softmax(scaled, dim=1)
    log_p = log_p.gather(1, action_idx.unsqueeze(1)).squeeze(1)
    return log_p, mask, action_idx


@torch.no_grad()
def batched_value_expectation(
    states: list[XqwlGameState],
    model: JointPolicyValueNet,
    device: torch.device,
    flist: dict[str, list[str]],
    *,
    encode_workers: int = 1,
    encode_backend: str = "inline",
    rollout_pipeline_groups: int = 1,
) -> torch.Tensor:
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
        _, vals = model.forward_heads_from_feat(feat)
        for j, idx in enumerate(active):
            out[idx] = vals[j]
    return out
