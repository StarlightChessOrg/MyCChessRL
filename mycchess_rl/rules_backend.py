"""规则后端：优先 ``xqwl_core``（XQWL06），否则回退 ``cchess`` GamePlay。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from cchess.board import FULL_INIT_FEN, BaseChessBoard
from cchess.piece import ChessSide

from mycchess_rl.chess.session import GamePlay


class RulesBackend(Protocol):
    def reset(self) -> None: ...
    def fen(self) -> str: ...
    def legal_iccs(self) -> list[str]: ...
    def make_iccs(self, mv: str) -> bool: ...
    def side_red_to_move(self) -> bool: ...
    def terminal(self) -> tuple[bool, str]: ...


@dataclass
class CppXQWLBackend:
    """象棋小巫师 C++ 核心（需先编译 ``xqwl_core`` 扩展）。"""

    _p: object  # xqwl_core.Position

    def __init__(self) -> None:
        from xqwl_core import Position

        self._p = Position()

    def reset(self) -> None:
        self._p.reset()

    def fen(self) -> str:
        return self._p.fen()

    def legal_iccs(self) -> list[str]:
        return list(self._p.legal_moves_iccs())

    def make_iccs(self, mv: str) -> bool:
        return bool(self._p.make_move_iccs(mv))

    def side_red_to_move(self) -> bool:
        return int(self._p.side_to_move()) == 0

    def terminal(self) -> tuple[bool, str]:
        k = int(self._p.terminal_kind())
        if k == 0:
            return False, ""
        if k == 1:
            return True, "checkmate"
        if k == 2:
            return True, "repetition_rule"
        if k == 3:
            return True, "move_limit_draw"
        return True, "unknown"


@dataclass
class PythonGamePlayBackend:
    """Python 规则（与训练特征同源）；重复/限着规则弱于 XQWL。"""

    gp: GamePlay

    def __init__(self) -> None:
        self.gp = GamePlay()

    def reset(self) -> None:
        self.gp = GamePlay()

    def fen(self) -> str:
        return self.gp.bb.to_fen()

    def legal_iccs(self) -> list[str]:
        return [f"{a}{b}-{c}{d}" for (a, b, c, d) in self.gp.legal_moves_iccs()]

    def make_iccs(self, mv: str) -> bool:
        if mv not in set(self.legal_iccs()):
            return False
        self.gp.make_move(mv)
        return True

    def side_red_to_move(self) -> bool:
        return self.gp.red

    def terminal(self) -> tuple[bool, str]:
        if self.gp.legal_moves_iccs():
            return False, ""
        from mycchess_rl.chess.board_utils import chess_board_from_base

        cb = chess_board_from_base(self.gp.bb)
        if cb.is_checkmate():
            return True, "checkmate"
        return True, "stalemate"


def make_rules_backend(prefer_cpp: bool = True) -> RulesBackend:
    if prefer_cpp:
        try:
            return CppXQWLBackend()
        except ImportError:
            pass
    return PythonGamePlayBackend()


def sync_gameplay_from_fen(gp: GamePlay, fen: str) -> None:
    """用 FEN 覆盖 ``GamePlay`` 棋盘并同步 ``red`` 与上一手（用于观测编码）。"""
    bb = BaseChessBoard(fen)
    gp.bb = bb
    gp.red = bb.move_side is not ChessSide.BLACK
