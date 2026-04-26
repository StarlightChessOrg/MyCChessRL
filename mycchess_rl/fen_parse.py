"""中国象棋 FEN 解析（不依赖外部棋库）。"""
from __future__ import annotations

import numpy as np

FULL_INIT_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"


def parse_fen_board(fen: str) -> tuple[np.ndarray, bool]:
    """
    返回 ``(boardarr, red_to_move)``。
    ``boardarr`` 形状 ``(10, 9)``，``dtype='<U1'``；``boardarr[0]`` 为 FEN 第一行（黑方底线），``boardarr[9]`` 为红方底线。
    """

    parts = fen.strip().split()
    rows = parts[0].split("/")
    if len(rows) != 10:
        raise ValueError(f"期望 10 行棋盘 FEN 行，得到 {len(rows)}: {fen!r}")
    board = np.full((10, 9), "", dtype="<U1")
    for y, row in enumerate(rows):
        x = 0
        for ch in row:
            if ch.isdigit():
                x += int(ch)
            else:
                if x >= 9:
                    raise ValueError(f"FEN 行溢出: {row!r}")
                board[y, x] = ch
                x += 1
        if x != 9:
            raise ValueError(f"FEN 行宽度不为 9: {row!r}")
    stm = parts[1] if len(parts) > 1 else "w"
    red_to_move = stm != "b"
    return board, red_to_move


def boardarr_to_fen(boardarr: np.ndarray, red_to_move: bool) -> str:
    """与 ``parse_fen_board`` 互逆（简化，仅用于调试）。"""
    rows: list[str] = []
    for y in range(10):
        s = ""
        empty = 0
        for x in range(9):
            ch = str(boardarr[y, x])
            if not ch:
                empty += 1
            else:
                if empty:
                    s += str(empty)
                    empty = 0
                s += ch
        if empty:
            s += str(empty)
        rows.append(s)
    stm = "w" if red_to_move else "b"
    return "/".join(rows) + f" {stm} - - 0 1"
