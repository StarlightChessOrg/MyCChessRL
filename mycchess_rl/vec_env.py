"""并行象棋环境：仅 ``xqwl_core`` 规则（``XqwlGameState``）。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mycchess_rl.encode_parallel import default_encode_workers
from mycchess_rl.policy_inference import batched_sample_moves_masked, eval_value_stm
from mycchess_rl.xqwl_state import XqwlGameState


def _is_capture_before_move(g: XqwlGameState, mv: str) -> bool:
    """走子前终点格是否有对方棋子（合法着法下即吃子）。"""
    if len(mv) < 5 or mv[2] != "-":
        return False
    x2, y2 = int(mv[3]), int(mv[4])
    b = g.board_view()
    ch = str(b[y2, x2]).strip()
    if not ch:
        return False
    if g.red_to_move:
        return ch.islower()
    return ch.isupper()


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
    reward_shaping_step: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[XqwlGameState], list[str]]:
    """``reward_shaping_*``：稀疏终局奖励下的轻量塑形（**0 关闭**）。

    - ``reward_shaping_step``：非终局且本步未将死时，每步加常数（常用小负数，抑制无目的长走）。
    - ``reward_shaping_check``：走子后若**行棋方**（即对手）处于应将，加小正奖（鼓励进攻性将杀压力）。
    - ``reward_shaping_capture``：走子前终点格有对方子时加小正奖（吃子塑形）。
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
        capture = _is_capture_before_move(g, mv) if reward_shaping_capture != 0.0 else False
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
            r_shape = float(reward_shaping_step)
            if reward_shaping_check != 0.0 and g.in_check():
                r_shape += float(reward_shaping_check)
            if reward_shaping_capture != 0.0 and capture:
                r_shape += float(reward_shaping_capture)
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
