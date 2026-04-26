"""并行象棋环境：仅 ``xqwl_core`` 规则（``XqwlGameState``）。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mycchess_rl.encode_parallel import default_encode_workers
from mycchess_rl.policy_inference import batched_sample_moves_masked, eval_value_stm
from mycchess_rl.xqwl_state import XqwlGameState


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
) -> tuple[np.ndarray, np.ndarray, list[XqwlGameState], list[str]]:
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
