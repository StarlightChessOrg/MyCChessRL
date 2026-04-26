"""棋子类型与 FEN 字符解析（替代原 cchess.piece 子集）。"""
from __future__ import annotations

from enum import Enum, auto


class ChessSide(Enum):
    RED = auto()
    BLACK = auto()

    @staticmethod
    def next_side(s: ChessSide) -> ChessSide:
        return ChessSide.BLACK if s is ChessSide.RED else ChessSide.RED


class PieceT(Enum):
    KING = auto()
    ADVISOR = auto()
    BISHOP = auto()
    KNIGHT = auto()
    ROOK = auto()
    CANNON = auto()
    PAWN = auto()


_fench_species_dict: dict[str, tuple[PieceT, ChessSide]] = {
    "k": (PieceT.KING, ChessSide.BLACK),
    "a": (PieceT.ADVISOR, ChessSide.BLACK),
    "b": (PieceT.BISHOP, ChessSide.BLACK),
    "n": (PieceT.KNIGHT, ChessSide.BLACK),
    "r": (PieceT.ROOK, ChessSide.BLACK),
    "c": (PieceT.CANNON, ChessSide.BLACK),
    "p": (PieceT.PAWN, ChessSide.BLACK),
    "K": (PieceT.KING, ChessSide.RED),
    "A": (PieceT.ADVISOR, ChessSide.RED),
    "B": (PieceT.BISHOP, ChessSide.RED),
    "N": (PieceT.KNIGHT, ChessSide.RED),
    "R": (PieceT.ROOK, ChessSide.RED),
    "C": (PieceT.CANNON, ChessSide.RED),
    "P": (PieceT.PAWN, ChessSide.RED),
}


def fench_to_species(fench: str) -> tuple[PieceT, ChessSide]:
    if not fench or fench not in _fench_species_dict:
        raise KeyError(fench)
    return _fench_species_dict[fench]
