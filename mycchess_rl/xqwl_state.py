"""唯一规则后端：``xqwl_core.Position`` + 从 FEN 派生的观测字段。"""
from __future__ import annotations

import numpy as np

from mycchess_rl.fen_parse import parse_fen_board
from mycchess_rl.iccs_util import parse_move_squares


def _require_xqwl():
    try:
        from xqwl_core import Position as _P  # noqa: F401

        return _P
    except ImportError as e:
        raise ImportError(
            "MyCChessRL 需要已编译安装的 ``xqwl_core`` 扩展（见 README 的 CMake 说明）。"
            "不允许回退到其他规则引擎。"
        ) from e


class XqwlGameState:
    """
    封装 ``xqwl_core.Position``；``board_view`` 与旧版 ``bb._board[::-1]`` 一致（红方在下）。
    """

    __slots__ = ("pos", "_last_move_iccs")

    def __init__(self) -> None:
        P = _require_xqwl()
        self.pos = P()
        self._last_move_iccs: str | None = None

    def reset(self) -> None:
        self.pos.reset()
        self._last_move_iccs = None

    @property
    def last_move_iccs(self) -> str | None:
        return self._last_move_iccs

    @property
    def red_to_move(self) -> bool:
        return int(self.pos.side_to_move()) == 0

    @property
    def red(self) -> bool:
        return self.red_to_move

    def get_side(self) -> str:
        return "red" if self.red_to_move else "black"

    def legal_moves_iccs(self) -> list[tuple[int, int, int, int]]:
        return [parse_move_squares(m) for m in self.pos.legal_moves_iccs()]

    def legal_moves_iccs_str(self) -> list[str]:
        return list(self.pos.legal_moves_iccs())

    def make_move_iccs(self, mv: str) -> bool:
        ok = bool(self.pos.make_move_iccs(mv))
        if ok:
            self._last_move_iccs = mv
        return ok

    def make_move(self, mv: str) -> None:
        """与旧 ``GamePlay.make_move`` 兼容：非法着法将触发断言失败。"""
        assert self.make_move_iccs(mv)

    def fen(self) -> str:
        return self.pos.fen()

    def board_view(self) -> np.ndarray:
        """(10,9) 与 ``encode_model_planes`` 使用的红下视角一致。"""
        board, _ = parse_fen_board(self.pos.fen())
        return np.flip(board, axis=0).astype("<U1", copy=False)

    def terminal(self) -> tuple[bool, str]:
        k = int(self.pos.terminal_kind())
        if k == 0:
            return False, ""
        if k == 1:
            return True, "checkmate"
        if k == 2:
            return True, "repetition_rule"
        if k == 3:
            return True, "move_limit_draw"
        return True, "unknown"

    def in_check(self) -> bool:
        return bool(self.pos.in_check())

    def copy(self) -> XqwlGameState:
        o = object.__new__(XqwlGameState)
        o.pos = self.pos.copy()
        o._last_move_iccs = self._last_move_iccs
        return o
