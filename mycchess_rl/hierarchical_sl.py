"""B 方案：子种 → 源格(90) → 目标格(90)；仅用于监督学习标签与维度常量。"""
from __future__ import annotations

from typing import Any

import numpy as np

from mycchess_rl.iccs_util import iccs_y_to_board_view_row, parse_move_squares
from mycchess_rl.piece_types import ChessSide, PieceT, fench_to_species

HIER_PIECE_HEAD_DIM = 7
HIER_SQUARE_HEAD_DIM = 90

_PIECE_ORDER: tuple[PieceT, ...] = (
    PieceT.KING,
    PieceT.ADVISOR,
    PieceT.BISHOP,
    PieceT.KNIGHT,
    PieceT.ROOK,
    PieceT.CANNON,
    PieceT.PAWN,
)


def piece_t_to_head_index(pt: PieceT) -> int:
    return int(_PIECE_ORDER.index(pt))


def square_iccs_to_index(y_iccs: int, x: int) -> int:
    """ICCS 引擎 (y,x) → ``board_view`` 行主序展平下标 0..89。"""
    return iccs_y_to_board_view_row(y_iccs) * 9 + int(x)


def build_hierarchical_sl_labels(
    st: Any,
    expert_mv: str,
) -> tuple[np.ndarray, np.int64, np.ndarray, np.int64, np.ndarray, np.int64] | None:
    """
    教师分解：``mt/mf/m2`` 为各步合法掩码；``tt/tf/t2`` 为专家着法对应下标。

    - 子种：行棋方尚有合法走法的子种槽 0..6。
    - 源格：仅 **专家子种** 下能作为某合法着法起点的格子。
    - 目标格：仅 **专家源格** 出发的合法落点。
    """
    legs = sorted(st.legal_moves_iccs_str())
    if not legs or expert_mv not in legs:
        return None
    try:
        x1, y1, x2, y2 = parse_move_squares(expert_mv)
    except ValueError:
        return None
    board = st.board_view()
    stm = ChessSide.RED if st.red_to_move else ChessSide.BLACK
    ch0 = str(board[iccs_y_to_board_view_row(y1), x1])
    if not ch0:
        return None
    try:
        sp0, sd0 = fench_to_species(ch0)
    except KeyError:
        return None
    if sd0 is not stm:
        return None
    t_type = np.int64(piece_t_to_head_index(sp0))
    t_from = np.int64(square_iccs_to_index(y1, x1))
    t_to = np.int64(square_iccs_to_index(y2, x2))

    mt = np.zeros((HIER_PIECE_HEAD_DIM,), dtype=np.bool_)
    mf = np.zeros((HIER_SQUARE_HEAD_DIM,), dtype=np.bool_)
    m2 = np.zeros((HIER_SQUARE_HEAD_DIM,), dtype=np.bool_)

    for mv in legs:
        ax1, ay1, ax2, ay2 = parse_move_squares(mv)
        ach = str(board[iccs_y_to_board_view_row(ay1), ax1])
        if not ach:
            continue
        try:
            asp, asd = fench_to_species(ach)
        except KeyError:
            continue
        if asd is not stm:
            continue
        ti = piece_t_to_head_index(asp)
        mt[ti] = True
        fi = square_iccs_to_index(ay1, ax1)
        if ti == int(t_type):
            mf[fi] = True
        if fi == int(t_from):
            m2[square_iccs_to_index(ay2, ax2)] = True

    if not (bool(mt[int(t_type)]) and bool(mf[int(t_from)]) and bool(m2[int(t_to)])):
        return None
    return mt, t_type, mf, t_from, m2, t_to
