"""AlphaZero / MyElephant 常见风格的 PUCT + MCTS：联合策略头 softmax 作先验，标量价值作叶子评估。"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mycchess_rl.policy_inference import infer_joint_policy_prior_and_value
from mycchess_rl.xqwl_state import XqwlGameState

if TYPE_CHECKING:
    from mycchess_rl.model import JointPolicyValueNet


@dataclass(slots=True)
class MCTSRunStats:
    """一次根搜索的汇总（便于 Web UI 日志）。"""

    n_simulations: int
    nn_evaluations: int
    terminal_leaf_evals: int
    tree_nodes: int
    root_branching: int
    root_edge_visits_total: int
    best_move: str


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
        device: Any,
        flist: dict,
        *,
        policy_temperature: float = 1.0,
        stats_ctx: dict[str, int] | None = None,
    ) -> float:
        """首次访问展开；已展开则返回缓存的叶子 v（行棋方、[-1,1]）。"""
        if self.expanded:
            return self._v_leaf
        if _is_terminal_or_dead(self.state):
            self.expanded = True
            self._v_leaf = _terminal_value_stm_norm(self.state)
            if stats_ctx is not None:
                stats_ctx["terminal_leaf_evals"] = stats_ctx.get("terminal_leaf_evals", 0) + 1
            return self._v_leaf
        if stats_ctx is not None:
            stats_ctx["nn_evaluations"] = stats_ctx.get("nn_evaluations", 0) + 1
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


def _tree_node_count(root: MCTSNode) -> int:
    n = 1
    for ch in root._children.values():
        n += _tree_node_count(ch)
    return n


def _apply_backup(path: list[tuple[MCTSNode, str]], v_leaf_stm: float) -> None:
    v = float(v_leaf_stm)
    for par, a in reversed(path):
        v = -v
        par._child_n[a] = par._child_n.get(a, 0) + 1
        par._child_w[a] = par._child_w.get(a, 0.0) + v


def mcts_select_move_iccs(
    root_state: XqwlGameState,
    model: JointPolicyValueNet,
    device: Any,
    flist: dict,
    *,
    n_simulations: int,
    c_puct: float = 1.5,
    policy_temperature: float = 1.0,
    progress: Callable[[int, int, int], Any] | None = None,
    progress_every: int = 25,
) -> tuple[str, MCTSRunStats]:
    """
    在 ``root_state`` 上行 ``n_simulations`` 次 PUCT 模拟，返回 ``(着法, 统计)``。
    ``progress(sim_done, nn_evals, tree_nodes)`` 可选，用于 UI 刷新（勿在回调里做重计算）。
    """
    n_simulations = int(n_simulations)
    if n_simulations < 1:
        raise ValueError("n_simulations 须 >= 1")
    pe = max(1, int(progress_every))
    stats_ctx: dict[str, int] = {"nn_evaluations": 0, "terminal_leaf_evals": 0}
    root = MCTSNode(root_state.copy())
    for si in range(n_simulations):
        path: list[tuple[MCTSNode, str]] = []
        node = root
        while not _is_terminal_or_dead(node.state):
            if not node.expanded:
                break
            a = node.select_action(c_puct)
            path.append((node, a))
            node = node._children[a]
        v_leaf = node.expand(
            model,
            device,
            flist,
            policy_temperature=policy_temperature,
            stats_ctx=stats_ctx,
        )
        _apply_backup(path, v_leaf)
        if progress is not None and ((si + 1) % pe == 0 or (si + 1) == n_simulations):
            progress(si + 1, int(stats_ctx.get("nn_evaluations", 0)), _tree_node_count(root))

    if not root._prior:
        raise RuntimeError("MCTS 根未产生子着法")
    ranked = sorted(
        root._prior.keys(),
        key=lambda a: (-root._child_n.get(a, 0), -root._prior.get(a, 0.0), a),
    )
    best = ranked[0]
    root_visits = int(sum(root._child_n.values()))
    st = MCTSRunStats(
        n_simulations=n_simulations,
        nn_evaluations=int(stats_ctx.get("nn_evaluations", 0)),
        terminal_leaf_evals=int(stats_ctx.get("terminal_leaf_evals", 0)),
        tree_nodes=_tree_node_count(root),
        root_branching=len(root._prior),
        root_edge_visits_total=root_visits,
        best_move=best,
    )
    return best, st
