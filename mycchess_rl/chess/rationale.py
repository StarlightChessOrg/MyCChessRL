"""
理据平面与常量（不依赖 cchess；着法统计来自 ``legal_iccs`` 列表）。
"""
from __future__ import annotations

import numpy as np

from mycchess_rl.iccs_util import parse_move_squares
from mycchess_rl.piece_types import ChessSide, PieceT, fench_to_species
from mycchess_rl.chess.plane_extras import EXTRA_HINT_PLANE_COUNT

PIECE_VALUE_BY_FENCH: dict[str, float] = {
    "r": 9.0,
    "n": 4.0,
    "c": 4.5,
    "b": 2.0,
    "a": 2.0,
    "p": 1.0,
    "k": 0.0,
}

PIECE_PLANE_COUNT = 14
RATIONALE_PLANE_COUNT = 11
PIECE_SIGNED_PLANE_COUNT = 7
POLICY_SELECT_IN_CHANNELS = PIECE_SIGNED_PLANE_COUNT + RATIONALE_PLANE_COUNT + EXTRA_HINT_PLANE_COUNT
POLICY_MAX_LEGAL_MOVES = 96
POLICY_GRID_NUMEL = 90

RED_OUTCOME_WIN = 0
RED_OUTCOME_DRAW = 1
RED_OUTCOME_LOSS = 2
STM_OUTCOME_WIN = 0
STM_OUTCOME_DRAW = 1
STM_OUTCOME_LOSS = 2
VALUE_LABEL_IGNORE = -100

STM_VALUE_EXPECT_WIN_COEF = 3.0
STM_VALUE_EXPECT_DRAW_COEF = 1.0
STM_VALUE_EXPECT_LOSS_COEF = 3.0
STM_VALUE_TERMINAL_LOSS = -STM_VALUE_EXPECT_LOSS_COEF
STM_VALUE_TERMINAL_DRAW = STM_VALUE_EXPECT_DRAW_COEF


def stm_value_expectation_from_win_draw_loss_probs(p_win: float, p_draw: float, p_loss: float) -> float:
    return (
        STM_VALUE_EXPECT_WIN_COEF * p_win
        + STM_VALUE_EXPECT_DRAW_COEF * p_draw
        - STM_VALUE_EXPECT_LOSS_COEF * p_loss
    )


def stm_outcome_class_from_red_outcome(red_cls: int, red_to_move: bool) -> int:
    if red_cls == VALUE_LABEL_IGNORE:
        return VALUE_LABEL_IGNORE
    if red_to_move:
        return int(red_cls)
    if red_cls == RED_OUTCOME_WIN:
        return STM_OUTCOME_LOSS
    if red_cls == RED_OUTCOME_LOSS:
        return STM_OUTCOME_WIN
    return STM_OUTCOME_DRAW


def _fench_material_value(fench: str) -> float:
    ch = fench.lower()
    return float(PIECE_VALUE_BY_FENCH.get(ch, 0.0))


def _palace_red_mask() -> np.ndarray:
    m = np.zeros((10, 9), dtype=np.float32)
    for iy in (0, 1, 2):
        for x in (3, 4, 5):
            m[9 - iy, x] = 1.0
    return m


def _palace_black_mask() -> np.ndarray:
    m = np.zeros((10, 9), dtype=np.float32)
    for iy in (7, 8, 9):
        for x in (3, 4, 5):
            m[9 - iy, x] = 1.0
    return m


def _black_territory_mask() -> np.ndarray:
    m = np.zeros((10, 9), dtype=np.float32)
    for iy in range(5, 10):
        m[9 - iy, :] = 1.0
    return m


def _king_planes(boardarr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kr = np.zeros((10, 9), dtype=np.float32)
    kb = np.zeros((10, 9), dtype=np.float32)
    for r in range(10):
        for c in range(9):
            ch = boardarr[r, c]
            if ch == "K":
                kr[r, c] = 1.0
            elif ch == "k":
                kb[r, c] = 1.0
    return kr, kb


def _side_to_move_plane(red_to_move: bool) -> np.ndarray:
    v = 1.0 if red_to_move else -1.0
    return np.full((10, 9), v, dtype=np.float32)


def _in_check_plane(in_check: bool) -> np.ndarray:
    v = 1.0 if in_check else 0.0
    return np.full((10, 9), v, dtype=np.float32)


def _signed_material_plane(boardarr: np.ndarray, red_to_move: bool) -> np.ndarray:
    out = np.zeros((10, 9), dtype=np.float32)
    stm = ChessSide.RED if red_to_move else ChessSide.BLACK
    denom = 9.0
    for iy in range(10):
        for ix in range(9):
            fench = str(boardarr[iy, ix])
            if not fench:
                continue
            val = _fench_material_value(fench)
            if val <= 0.0:
                continue
            _, side = fench_to_species(fench)
            ar, ac = 9 - iy, ix
            sign = 1.0 if side == stm else -1.0
            out[ar, ac] = float(np.clip(sign * val / denom, -1.0, 1.0))
    return out


def _mobility_from_legals(boardarr: np.ndarray, legal_iccs: list[str]) -> np.ndarray:
    out = np.zeros((10, 9), dtype=np.float32)
    cnt: dict[tuple[int, int], int] = {}
    for mv in legal_iccs:
        x1, y1, x2, y2 = parse_move_squares(mv)
        ch = str(boardarr[y1, x1])
        if not ch:
            continue
        cnt[(x1, y1)] = cnt.get((x1, y1), 0) + 1
    for (x1, y1), c in cnt.items():
        ar, ac = 9 - y1, x1
        out[ar, ac] = min(1.0, c / 25.0)
    return out


def _mobility_quality_heuristic(boardarr: np.ndarray, legal_iccs: list[str], mover_red: bool) -> np.ndarray:
    out = np.zeros((10, 9), dtype=np.float32)
    quals: dict[tuple[int, int], list[float]] = {}
    for mv in legal_iccs:
        x1, y1, x2, y2 = parse_move_squares(mv)
        ch = str(boardarr[y1, x1])
        if not ch:
            continue
        sp, side = fench_to_species(ch)
        if (side == ChessSide.RED) != mover_red:
            continue
        q = 0.92
        if sp == PieceT.KNIGHT:
            s = 1.0
            if x2 in (0, 8):
                s -= 0.45
            q = float(np.clip(s, 0.08, 1.0))
        elif sp == PieceT.ROOK:
            q = 0.95
        elif sp == PieceT.CANNON:
            q = 0.93
        elif sp == PieceT.PAWN:
            q = 0.9
        quals.setdefault((x1, y1), []).append(q)
    for (x1, y1), qs in quals.items():
        ar, ac = 9 - y1, x1
        out[ar, ac] = float(np.clip(float(np.mean(qs)), 0.0, 1.0))
    return out


def encode_rationale_planes(
    boardarr: np.ndarray, red_to_move: bool, in_check: bool, legal_iccs: list[str]
) -> np.ndarray:
    pr = _palace_red_mask()
    pb = _palace_black_mask()
    terr_b = _black_territory_mask()
    stm = _side_to_move_plane(red_to_move)
    kr, kb = _king_planes(boardarr)
    chk = _in_check_plane(in_check)
    mat = _signed_material_plane(boardarr, red_to_move)
    mob_self = _mobility_from_legals(boardarr, legal_iccs)
    mob_opp = np.zeros((10, 9), dtype=np.float32)
    mob_q_self = _mobility_quality_heuristic(boardarr, legal_iccs, red_to_move)
    return np.stack(
        [pr, pb, terr_b, stm, kr, kb, chk, mat, mob_self, mob_opp, mob_q_self],
        axis=0,
    ).astype(np.float32)
