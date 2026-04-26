"""并行象棋环境：仅 ``xqwl_core`` 规则（``XqwlGameState``）。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mycchess_rl.encode_parallel import default_encode_workers
from mycchess_rl.policy_inference import batched_sample_moves_masked, eval_value_stm
from mycchess_rl.reward_patterns import DEFAULT_TACTIC_SHAPING_COEFF, tactics_shaping_total
from mycchess_rl.xqwl_state import XqwlGameState

# 吃子塑形：以兵/卒为 1.0 的相对权重（与 ``reward_shaping_capture`` 基量相乘）
_CAPTURE_MULT: dict[str, float] = {
    "p": 1.0,
    "P": 1.0,
    "a": 1.35,
    "A": 1.35,
    "b": 1.35,
    "B": 1.35,
    "n": 1.85,
    "N": 1.85,
    "c": 2.6,
    "C": 2.6,
    "r": 3.2,
    "R": 3.2,
    "k": 4.0,
    "K": 4.0,
}


def _captured_piece_before_move(g: XqwlGameState, mv: str) -> str | None:
    """走子前终点格上的对方棋子字符；非吃子返回 ``None``。"""
    if len(mv) < 5 or mv[2] != "-":
        return None
    x2, y2 = int(mv[3]), int(mv[4])
    b = g.board_view()
    ch = str(b[y2, x2]).strip()
    if not ch:
        return None
    if g.red_to_move:
        return ch if ch.islower() else None
    return ch if ch.isupper() else None


def _capture_multiplier(piece: str) -> float:
    return float(_CAPTURE_MULT.get(piece, 1.0))


def _opponent_king_char_after_move(g: XqwlGameState) -> str:
    """刚走完一方后轮到对方走；对方将/帅在棋盘上的字符。"""
    return "K" if g.red_to_move else "k"


def _king_manhattan_proximity(
    g: XqwlGameState,
    x_land: int,
    y_land: int,
    *,
    d_max: int = 17,
) -> float:
    """落点 ``(x_land,y_land)`` 与对方将/帅的曼哈顿距离 → ``[0,1]``，越近越大。"""
    b = g.board_view()
    king_ch = _opponent_king_char_after_move(g)
    ys, xs = np.where(b == king_ch)
    if ys.size == 0:
        return 0.0
    ky, kx = int(ys[0]), int(xs[0])
    dist = abs(kx - x_land) + abs(ky - y_land)
    return max(0.0, 1.0 - float(dist) / float(max(1, d_max)))


@dataclass
class SlotState:
    game: XqwlGameState = field(default_factory=XqwlGameState)


class ParallelXiangqiVecEnv:
    def __init__(self, n_env: int) -> None:
        self.n_env = int(n_env)
        self.slots = [SlotState() for _ in range(self.n_env)]

    def reset_all(self) -> None:
        for s in self.slots:
            s.game.reset()


def collect_rollout_step(
    vec: ParallelXiangqiVecEnv,
    model: object,
    device: object,
    flist: dict[str, list[str]],
    *,
    policy_temperature: float = 1.0,
    generator: object | None = None,
    encode_workers: int | None = None,
    encode_backend: str = "inline",
    rollout_pipeline_groups: int = 1,
    reward_shaping_check: float = 0.0,
    reward_shaping_capture: float = 0.0,
    reward_shaping_king_prox: float = 0.0,
    reward_shaping_step: float = 0.0,
    reward_shaping_ae_shape: float = DEFAULT_TACTIC_SHAPING_COEFF,
    reward_shaping_double_cannon: float = DEFAULT_TACTIC_SHAPING_COEFF,
    reward_shaping_rook_pair: float = DEFAULT_TACTIC_SHAPING_COEFF,
    reward_shaping_cross_pawn: float = DEFAULT_TACTIC_SHAPING_COEFF,
    reward_shaping_knight_flex: float = DEFAULT_TACTIC_SHAPING_COEFF,
    reward_shaping_three_edge: float = DEFAULT_TACTIC_SHAPING_COEFF,
    reward_shaping_central_cannon: float = DEFAULT_TACTIC_SHAPING_COEFF,
    reward_shaping_open_cannon: float = DEFAULT_TACTIC_SHAPING_COEFF,
    reward_shaping_rook_pin_cannon: float = DEFAULT_TACTIC_SHAPING_COEFF,
    reward_shaping_opp_king_gate: float = DEFAULT_TACTIC_SHAPING_COEFF,
    reward_shaping_miss_adv_double_rook: float = DEFAULT_TACTIC_SHAPING_COEFF,
    reward_shaping_double_adv_king_center: float = DEFAULT_TACTIC_SHAPING_COEFF,
    reward_shaping_king_near_start: float = DEFAULT_TACTIC_SHAPING_COEFF,
) -> tuple[np.ndarray, np.ndarray, list[XqwlGameState], list[str]]:
    """``reward_shaping_*``：稀疏终局奖励下的轻量塑形（**0 关闭**）。

    - ``reward_shaping_step``：非终局每步常数（常用小负数）。
    - ``reward_shaping_capture``：吃子基量 × 按子种相对权重（兵=1，车马炮等更高）。
    - ``reward_shaping_check``：对手应将时基量 × ``(0.4 + 0.6×prox)``，``prox`` 为落点与对方将/帅接近度。
    - ``reward_shaping_king_prox``：每步（非终局）额外 ``×prox``，微弱鼓励压将。
    - 战术类 ``reward_shaping_*`` 默认 ``reward_patterns.DEFAULT_TACTIC_SHAPING_COEFF``；``0`` 关闭该项。
    - 详见 ``reward_patterns.tactics_shaping_total``（士象、担子炮、过河卒、将门等）。
    """
    n = vec.n_env
    rewards = np.zeros(n, dtype=np.float32)
    dones = np.zeros(n, dtype=np.bool_)
    moves_out = [""] * n

    active = list(range(n))
    gps = [vec.slots[i].game for i in active]
    ew = default_encode_workers() if encode_workers is None else int(encode_workers)
    sampled = batched_sample_moves_masked(
        gps,
        model,
        device,
        flist,
        policy_temperature=policy_temperature,
        generator=generator,
        encode_workers=ew,
        encode_backend=encode_backend,
        rollout_pipeline_groups=int(rollout_pipeline_groups),
    )
    for j, i in enumerate(active):
        moves_out[i] = sampled[j]

    for i in active:
        g = vec.slots[i].game
        mv = moves_out[i]
        leg = g.legal_moves_iccs_str()
        if not leg:
            dones[i] = True
            rewards[i] = 0.0
            continue
        if mv not in leg:
            mv = leg[0]
        x2, y2 = int(mv[3]), int(mv[4])
        cap_piece = (
            _captured_piece_before_move(g, mv) if reward_shaping_capture != 0.0 else None
        )
        g.make_move_iccs(mv)

        term, reason = g.terminal()
        if term:
            dones[i] = True
            if reason == "checkmate":
                rewards[i] = 1.0
            else:
                rewards[i] = 0.0
        else:
            dones[i] = False
            prox = _king_manhattan_proximity(g, x2, y2)
            r_shape = float(reward_shaping_step)
            if reward_shaping_king_prox != 0.0:
                r_shape += float(reward_shaping_king_prox) * prox
            if reward_shaping_check != 0.0 and g.in_check():
                check_scale = 0.4 + 0.6 * prox
                r_shape += float(reward_shaping_check) * float(check_scale)
            if reward_shaping_capture != 0.0 and cap_piece is not None:
                r_shape += float(reward_shaping_capture) * _capture_multiplier(cap_piece)
            bv = g.board_view()
            piece_dst = str(bv[y2, x2]).strip()
            last_mover_red = not g.red_to_move
            r_shape += tactics_shaping_total(
                bv,
                piece_dst,
                x2,
                y2,
                last_mover_red=last_mover_red,
                coeff_ae=float(reward_shaping_ae_shape),
                coeff_double_cannon=float(reward_shaping_double_cannon),
                coeff_rook_pair=float(reward_shaping_rook_pair),
                coeff_cross_pawn=float(reward_shaping_cross_pawn),
                coeff_knight_flex=float(reward_shaping_knight_flex),
                coeff_three_edge=float(reward_shaping_three_edge),
                coeff_central_cannon=float(reward_shaping_central_cannon),
                coeff_open_cannon=float(reward_shaping_open_cannon),
                coeff_rook_pin_cannon=float(reward_shaping_rook_pin_cannon),
                coeff_opp_king_gate=float(reward_shaping_opp_king_gate),
                coeff_miss_adv_double_rook=float(reward_shaping_miss_adv_double_rook),
                coeff_double_adv_king_center=float(reward_shaping_double_adv_king_center),
                coeff_king_near_start=float(reward_shaping_king_near_start),
            )
            rewards[i] = r_shape

    return rewards, dones, [s.game for s in vec.slots], moves_out


def reset_finished(vec: ParallelXiangqiVecEnv, dones: np.ndarray) -> None:
    for i, d in enumerate(dones):
        if d:
            vec.slots[i].game.reset()


def bootstrap_values(
    vec: ParallelXiangqiVecEnv, model: object, device: object, flist: dict[str, list[str]]
) -> np.ndarray:
    return np.asarray(
        [eval_value_stm(s.game, model, device, flist) for s in vec.slots],
        dtype=np.float32,
    )
