"""并行象棋环境：仅 ``xqwl_core`` 规则（``XqwlGameState``）。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mycchess_rl.encode_parallel import default_encode_workers
from mycchess_rl.policy_inference import batched_sample_moves_masked, eval_value_stm
from mycchess_rl.xqwl_state import REP_RULE_VALUE_DRAWISH_ABS, XqwlGameState

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


def _repetition_terminal_reward_for_last_mover(g: XqwlGameState) -> float:
    """重复终局（含长将判负）对上一步走子方的标量回报：胜 +1、负 -1、和 0。

    ``rep_value_if_any`` 为 XQWL 对**当前行棋方**（终局后应先走的一方）的重复局面分值：
    明显为正则该行棋方在判例中得利 → 刚走完的一方失利 ``-1``；明显为负则反之 ``+1``。
    """
    v = int(g.rep_value_if_any())
    if abs(v) <= REP_RULE_VALUE_DRAWISH_ABS:
        return 0.0
    if v > 0:
        return -1.0
    return 1.0


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


def _king_attack_shaping(g: XqwlGameState, x_land: int, y_land: int, coeff: float) -> float:
    """鼓励落点靠近对方将/帅；若走后对方处于应将，再叠加与接近度相关的奖励。"""
    if coeff == 0.0:
        return 0.0
    prox = _king_manhattan_proximity(g, x_land, y_land)
    out = float(coeff) * prox
    if g.in_check():
        out += float(coeff) * (0.4 + 0.6 * prox)
    return float(out)


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
    reward_shaping_capture: float = 0.0,
    reward_shaping_king: float = 0.0,
    random_action_prob: float = 0.0,
    explore_rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, list[XqwlGameState], list[str]]:
    """非终局塑形仅两项（**0 关闭**）：压对方将/帅、吃子（按子种加权）。

    - ``reward_shaping_king``：落点与对方将/帅的接近度 ``prox∈[0,1]`` 线性奖励；若走后对方应将，再加 ``×(0.4+0.6·prox)``。
    - ``reward_shaping_capture``：吃子时 ``基量 × 子种权重``（兵卒=1，象士、马、炮、车、将递增，见 ``_CAPTURE_MULT``）。
    - ``random_action_prob``：每环境每步以该概率用 **均匀随机合法 ICCS** 替换策略样本（需 ``explore_rng``）；利于离开开局记忆、覆盖乱战。
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

    p_rand = float(random_action_prob)
    if p_rand > 0.0 and explore_rng is not None:
        p_rand = min(1.0, max(0.0, p_rand))
        for i in active:
            if explore_rng.random() >= p_rand:
                continue
            g = vec.slots[i].game
            leg = g.legal_moves_iccs_str()
            if not leg:
                continue
            lex = sorted(leg)
            moves_out[i] = str(explore_rng.choice(lex))

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
            elif reason == "repetition_rule":
                rewards[i] = float(_repetition_terminal_reward_for_last_mover(g))
            else:
                rewards[i] = 0.0
        else:
            dones[i] = False
            r_shape = _king_attack_shaping(g, x2, y2, float(reward_shaping_king))
            if reward_shaping_capture != 0.0 and cap_piece is not None:
                r_shape += float(reward_shaping_capture) * _capture_multiplier(cap_piece)
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
