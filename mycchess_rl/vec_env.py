"""并行象棋环境：每步对所有未结束槽位批采样、走一步；终局后下一节拍重置。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mycchess_rl.policy_inference import batched_sample_moves_masked, eval_value_stm
from mycchess_rl.rules_backend import make_rules_backend, sync_gameplay_from_fen
from mycchess_rl.chess.session import GamePlay


@dataclass
class SlotState:
    backend: object
    gp_mirror: GamePlay = field(default_factory=GamePlay)
    done: bool = False


class ParallelXiangqiVecEnv:
    def __init__(self, n_env: int, *, prefer_cpp: bool = True) -> None:
        self.n_env = int(n_env)
        self.slots = [SlotState(backend=make_rules_backend(prefer_cpp)) for _ in range(self.n_env)]

    def reset_all(self) -> None:
        for s in self.slots:
            s.backend.reset()
            sync_gameplay_from_fen(s.gp_mirror, s.backend.fen())
            s.gp_mirror._last_move_iccs = None  # noqa: SLF001
            s.done = False


def collect_rollout_step(
    vec: ParallelXiangqiVecEnv,
    model: torch.nn.Module,
    device: torch.device,
    flist: dict[str, list[str]],
    *,
    policy_temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, list[GamePlay], list[str]]:
    """
    对所有槽位执行一步（含已终局槽位会先被 ``reset_finished`` 在外部清掉）。
    返回 ``(reward_red_perspective, done, gameplays, moves)``。
    终局回报：将死时红方视角 +1（红胜）/ -1（红负）；其余终局 0。
    """
    n = vec.n_env
    rewards = np.zeros(n, dtype=np.float32)
    dones = np.zeros(n, dtype=np.bool_)
    moves_out = [""] * n

    active = [i for i in range(n) if not vec.slots[i].done]
    if not active:
        vec.reset_all()
        active = list(range(n))

    gps = [vec.slots[i].gp_mirror for i in active]
    sampled = batched_sample_moves_masked(
        gps, model, device, flist, policy_temperature=policy_temperature, generator=generator
    )
    for j, i in enumerate(active):
        moves_out[i] = sampled[j]

    for i in active:
        s = vec.slots[i]
        mv = moves_out[i]
        leg = s.backend.legal_iccs()
        if not leg:
            dones[i] = True
            rewards[i] = 0.0
            s.done = True
            continue
        if mv not in leg:
            mv = leg[0]
        s.backend.make_iccs(mv)
        sync_gameplay_from_fen(s.gp_mirror, s.backend.fen())
        s.gp_mirror._last_move_iccs = mv  # noqa: SLF001

        term, reason = s.backend.terminal()
        if term:
            dones[i] = True
            s.done = True
            if reason == "checkmate":
                # 刚刚走完的一方将死对方 → 对「走子前的行棋方」+1
                rewards[i] = 1.0
            else:
                rewards[i] = 0.0
        else:
            dones[i] = False

    return rewards, dones, [s.gp_mirror for s in vec.slots], moves_out


def reset_finished(vec: ParallelXiangqiVecEnv, dones: np.ndarray) -> None:
    for i, d in enumerate(dones):
        if d:
            s = vec.slots[i]
            s.backend.reset()
            sync_gameplay_from_fen(s.gp_mirror, s.backend.fen())
            s.gp_mirror._last_move_iccs = None  # noqa: SLF001
            s.done = False


def bootstrap_values(
    vec: ParallelXiangqiVecEnv, model: torch.nn.Module, device: torch.device, flist: dict[str, list[str]]
) -> np.ndarray:
    return np.asarray(
        [eval_value_stm(s.gp_mirror, model, device, flist) for s in vec.slots],
        dtype=np.float32,
    )
