"""AlphaZero / MyElephant 常见风格的 PUCT + MCTS：联合策略头 softmax 作先验，标量价值作叶子评估。"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mycchess_rl.policy_inference import infer_joint_policy_prior_and_value
from mycchess_rl.xqwl_state import XqwlGameState

if TYPE_CHECKING:
    from mycchess_rl.model import JointPolicyValueNet


def _terminal_value_stm_norm(state: XqwlGameState) -> float:
    t, r = state.terminal()
    if t and r == "checkmate":
        return -1.0
    return 0.0


def _is_terminal_or_dead(state: XqwlGameState) -> bool:
    if state.terminal()[0]:
        return True
    return not state.legal_moves_iccs_str()


class MCTSNode:
    __slots__ = ("state", "expanded", "_v_leaf", "_child_n", "_child_w", "_children", "_prior")

    def __init__(self, state: XqwlGameState) -> None:
        self.state = state
        self.expanded = False
        self._v_leaf = 0.0
        self._child_n: dict[str, int] = {}
        self._child_w: dict[str, float] = {}
        self._children: dict[str, MCTSNode] = {}
        self._prior: dict[str, float] = {}

    def expand(
        self,
        model: JointPolicyValueNet,
        device: torch.device,
        flist: dict,
        *,
        policy_temperature: float = 1.0,
    ) -> float:
        """首次访问展开；已展开则返回缓存的叶子 v（行棋方、[-1,1]）。"""
        if self.expanded:
            return self._v_leaf
        if _is_terminal_or_dead(self.state):
            self.expanded = True
            self._v_leaf = _terminal_value_stm_norm(self.state)
            return self._v_leaf
        legs, probs, v_norm = infer_joint_policy_prior_and_value(
            self.state, model, device, flist, policy_temperature=policy_temperature
        )
        self._v_leaf = v_norm
        for i, a in enumerate(legs):
            self._prior[a] = float(probs[i])
            st = self.state.copy()
            assert st.make_move_iccs(a)
            self._children[a] = MCTSNode(st)
            self._child_n[a] = 0
            self._child_w[a] = 0.0
        self.expanded = True
        return self._v_leaf

    def select_action(self, c_puct: float) -> str:
        total_n = sum(self._child_n.values())
        sqrt_n = math.sqrt(max(1, total_n))
        best_a, best = None, -1e99
        for a, p in self._prior.items():
            n = self._child_n[a]
            q = (self._child_w[a] / n) if n > 0 else 0.0
            u = c_puct * p * sqrt_n / (1.0 + n)
            score = q + u
            if score > best:
                best, best_a = score, a
        if best_a is None:
            raise RuntimeError("MCTS select: 无先验子边")
        return best_a


def _apply_backup(path: list[tuple[MCTSNode, str]], v_leaf_stm: float) -> None:
    v = float(v_leaf_stm)
    for par, a in reversed(path):
        v = -v
        par._child_n[a] = par._child_n.get(a, 0) + 1
        par._child_w[a] = par._child_w.get(a, 0.0) + v


def mcts_select_move_iccs(
    root_state: XqwlGameState,
    model: JointPolicyValueNet,
    device: torch.device,
    flist: dict,
    *,
    n_simulations: int,
    c_puct: float = 1.5,
    policy_temperature: float = 1.0,
) -> str:
    """
    在 ``root_state`` 上行 ``n_simulations`` 次 PUCT 模拟，返回访问次数最多的 ICCS 着法。
    与「纯网络一步 argmax」独立；规则与 ``xqwl_core`` 一致。
    """
    n_simulations = int(n_simulations)
    if n_simulations < 1:
        raise ValueError("n_simulations 须 >= 1")
    root = MCTSNode(root_state.copy())
    for _ in range(n_simulations):
        path: list[tuple[MCTSNode, str]] = []
        node = root
        while not _is_terminal_or_dead(node.state):
            if not node.expanded:
                break
            a = node.select_action(c_puct)
            path.append((node, a))
            node = node._children[a]
        v_leaf = node.expand(model, device, flist, policy_temperature=policy_temperature)
        _apply_backup(path, v_leaf)

    if not root._prior:
        raise RuntimeError("MCTS 根未产生子着法")
    # 访问数优先；全 0（例如仅 1 次模拟只做了根展开）时退化为先验最大
    ranked = sorted(
        root._prior.keys(),
        key=lambda a: (-root._child_n.get(a, 0), -root._prior.get(a, 0.0), a),
    )
    return ranked[0]
