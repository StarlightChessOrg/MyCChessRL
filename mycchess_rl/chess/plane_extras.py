"""额外提示平面（仅用棋盘矩阵 + 当前方合法着 ICCS 列表，不依赖 cchess）。"""
from __future__ import annotations

import numpy as np

from mycchess_rl.iccs_util import parse_move_squares
from mycchess_rl.piece_types import ChessSide, PieceT, fench_to_species

EXTRA_HINT_PLANE_COUNT = 47
_MAT_SUM_DENOM = 55.0


def _mat_val(ch: str) -> float:
    if not ch:
        return 0.0
    key = ch.lower()
    m = {"r": 9.0, "n": 4.0, "c": 4.5, "b": 2.0, "a": 2.0, "p": 1.0, "k": 0.0}
    return float(m.get(key, 0.0))


def _coord_planes() -> np.ndarray:
    out = np.zeros((2, 10, 9), dtype=np.float32)
    for iy in range(10):
        for ix in range(9):
            out[0, iy, ix] = ix / 8.0
            out[1, iy, ix] = iy / 9.0
    return out


_COORD_CACHE: np.ndarray | None = None


def _coord_planes_cached() -> np.ndarray:
    global _COORD_CACHE
    if _COORD_CACHE is None:
        _COORD_CACHE = _coord_planes()
    return _COORD_CACHE


def _ply_broadcast_plane(move_index: int | None) -> np.ndarray:
    if move_index is None or move_index < 0:
        v = 0.0
    else:
        v = min(1.0, float(move_index) / 150.0)
    return np.full((10, 9), v, dtype=np.float32)


def _kings_face_plane(boardarr: np.ndarray) -> np.ndarray:
    k_red: tuple[int, int] | None = None
    k_blk: tuple[int, int] | None = None
    for iy in range(10):
        for ix in range(9):
            ch = boardarr[iy, ix]
            if ch == "K":
                k_red = (ix, iy)
            elif ch == "k":
                k_blk = (ix, iy)
    if k_red is None or k_blk is None:
        return np.zeros((10, 9), dtype=np.float32)
    xr, yr = k_red
    xb, yb = k_blk
    if xr != xb:
        return np.zeros((10, 9), dtype=np.float32)
    y_lo, y_hi = (yr, yb) if yr < yb else (yb, yr)
    for y in range(y_lo + 1, y_hi):
        if boardarr[y, xr]:
            return np.zeros((10, 9), dtype=np.float32)
    return np.full((10, 9), 1.0, dtype=np.float32)


def _union_move_destinations(legal_iccs: list[str]) -> np.ndarray:
    out = np.zeros((10, 9), dtype=np.float32)
    for mv in legal_iccs:
        _x1, _y1, x2, y2 = parse_move_squares(mv)
        out[y2, x2] = 1.0
    return out


def _last_move_planes(last_move: str | None) -> tuple[np.ndarray, np.ndarray]:
    a = np.zeros((10, 9), dtype=np.float32)
    b = np.zeros((10, 9), dtype=np.float32)
    if not last_move or len(last_move) < 5 or last_move[2] != "-":
        return a, b
    try:
        x1, y1, x2, y2 = parse_move_squares(last_move)
    except ValueError:
        return a, b
    a[y1, x1] = 1.0
    b[y2, x2] = 1.0
    return a, b


def _material_broadcast(boardarr: np.ndarray, stm: ChessSide) -> tuple[np.ndarray, np.ndarray]:
    s_stm = 0.0
    s_opp = 0.0
    for iy in range(10):
        for ix in range(9):
            ch = str(boardarr[iy, ix])
            if not ch:
                continue
            v = _mat_val(ch)
            if v <= 0.0:
                continue
            _sp, side = fench_to_species(ch)
            if side == stm:
                s_stm += v
            else:
                s_opp += v
    v_stm = min(1.0, s_stm / _MAT_SUM_DENOM)
    v_opp = min(1.0, s_opp / _MAT_SUM_DENOM)
    return np.full((10, 9), v_stm, dtype=np.float32), np.full((10, 9), v_opp, dtype=np.float32)


def _king_geometry(boardarr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k_red: tuple[int, int] | None = None
    k_blk: tuple[int, int] | None = None
    for iy in range(10):
        for ix in range(9):
            ch = boardarr[iy, ix]
            if ch == "K":
                k_red = (ix, iy)
            elif ch == "k":
                k_blk = (ix, iy)
    if k_red is None or k_blk is None:
        z = np.zeros((10, 9), dtype=np.float32)
        return z, z
    xr, yr = k_red
    xb, yb = k_blk
    dist = abs(xr - xb) + abs(yr - yb)
    dn = min(1.0, float(dist) / 17.0)
    same_rank = 1.0 if yr == yb else 0.0
    return np.full((10, 9), dn, dtype=np.float32), np.full((10, 9), same_rank, dtype=np.float32)


def _board_density(boardarr: np.ndarray) -> np.ndarray:
    n = sum(1 for iy in range(10) for ix in range(9) if boardarr[iy, ix])
    return np.full((10, 9), float(n) / 90.0, dtype=np.float32)


def _major_ratio(boardarr: np.ndarray, side: ChessSide) -> np.ndarray:
    majors = (PieceT.ROOK, PieceT.KNIGHT, PieceT.CANNON)
    c = 0
    for iy in range(10):
        for ix in range(9):
            ch = str(boardarr[iy, ix])
            if not ch:
                continue
            sp, sd = fench_to_species(ch)
            if sd == side and sp in majors:
                c += 1
    return np.full((10, 9), min(1.0, c / 6.0), dtype=np.float32)


def _capture_dst_union(boardarr: np.ndarray, legal_iccs: list[str], mover_red: bool) -> np.ndarray:
    out = np.zeros((10, 9), dtype=np.float32)
    stm = ChessSide.RED if mover_red else ChessSide.BLACK
    opp = ChessSide.next_side(stm)
    for mv in legal_iccs:
        x1, y1, x2, y2 = parse_move_squares(mv)
        dest = str(boardarr[y2, x2])
        if not dest:
            continue
        _, sd_src = fench_to_species(str(boardarr[y1, x1]))
        _, sd_dst = fench_to_species(dest)
        if sd_src == stm and sd_dst == opp:
            out[y2, x2] = 1.0
    return out


def _pawn_progress_plane(boardarr: np.ndarray) -> np.ndarray:
    out = np.zeros((10, 9), dtype=np.float32)
    for iy in range(10):
        for ix in range(9):
            ch = boardarr[iy, ix]
            if ch == "P":
                out[iy, ix] = (9.0 - float(iy)) / 9.0
            elif ch == "p":
                out[iy, ix] = float(iy) / 9.0
    return out


def _species_dst_union(boardarr: np.ndarray, legal_iccs: list[str], species: PieceT) -> np.ndarray:
    out = np.zeros((10, 9), dtype=np.float32)
    for mv in legal_iccs:
        x1, y1, x2, y2 = parse_move_squares(mv)
        ch = str(boardarr[y1, x1])
        if not ch:
            continue
        sp, _ = fench_to_species(ch)
        if sp != species:
            continue
        out[y2, x2] = 1.0
    return out


def _river_band_plane() -> np.ndarray:
    out = np.zeros((10, 9), dtype=np.float32)
    out[4, :] = 1.0
    out[5, :] = 1.0
    return out


def _half_board_planes() -> tuple[np.ndarray, np.ndarray]:
    red = np.zeros((10, 9), dtype=np.float32)
    blk = np.zeros((10, 9), dtype=np.float32)
    red[5:, :] = 1.0
    blk[:5, :] = 1.0
    return red, blk


def _find_king_iccs(boardarr: np.ndarray, side: ChessSide) -> tuple[int, int] | None:
    target = "K" if side == ChessSide.RED else "k"
    for iy in range(10):
        for ix in range(9):
            if boardarr[iy, ix] == target:
                return ix, iy
    return None


def _king_ortho_neighbor_density(boardarr: np.ndarray, side: ChessSide) -> np.ndarray:
    pos = _find_king_iccs(boardarr, side)
    if pos is None:
        return np.zeros((10, 9), dtype=np.float32)
    kx, ky = pos
    n = 0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = kx + dx, ky + dy
        if 0 <= nx < 9 and 0 <= ny < 10 and boardarr[ny, nx]:
            n += 1
    v = n / 4.0
    return np.full((10, 9), v, dtype=np.float32)


def _king_cross_empty_rays(boardarr: np.ndarray, side: ChessSide) -> np.ndarray:
    out = np.zeros((10, 9), dtype=np.float32)
    pos = _find_king_iccs(boardarr, side)
    if pos is None:
        return out
    kx, ky = pos
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        s = 1
        while True:
            nx, ny = kx + dx * s, ky + dy * s
            if not (0 <= nx < 9 and 0 <= ny < 10):
                break
            if boardarr[ny, nx]:
                break
            out[ny, nx] = 1.0
            s += 1
    return out


def _pawn_capture_union(boardarr: np.ndarray, legal_iccs: list[str], mover_red: bool) -> np.ndarray:
    out = np.zeros((10, 9), dtype=np.float32)
    stm = ChessSide.RED if mover_red else ChessSide.BLACK
    opp = ChessSide.next_side(stm)
    for mv in legal_iccs:
        x1, y1, x2, y2 = parse_move_squares(mv)
        ch = str(boardarr[y1, x1])
        if not ch:
            continue
        sp, sd = fench_to_species(ch)
        if sp != PieceT.PAWN or sd != stm:
            continue
        dest = str(boardarr[y2, x2])
        if not dest:
            continue
        _, sd2 = fench_to_species(dest)
        if sd2 == opp:
            out[y2, x2] = 1.0
    return out


def _bishop_advisor_union(boardarr: np.ndarray, legal_iccs: list[str], mover_red: bool) -> np.ndarray:
    out = np.zeros((10, 9), dtype=np.float32)
    stm = ChessSide.RED if mover_red else ChessSide.BLACK
    for mv in legal_iccs:
        x1, y1, x2, y2 = parse_move_squares(mv)
        ch = str(boardarr[y1, x1])
        if not ch:
            continue
        sp, sd = fench_to_species(ch)
        if sd != stm or sp not in (PieceT.BISHOP, PieceT.ADVISOR):
            continue
        out[y2, x2] = 1.0
    return out


def _count_species(boardarr: np.ndarray, side: ChessSide, species: PieceT) -> int:
    c = 0
    for iy in range(10):
        for ix in range(9):
            ch = str(boardarr[iy, ix])
            if not ch:
                continue
            sp, sd = fench_to_species(ch)
            if sd == side and sp == species:
                c += 1
    return c


def _species_count_broadcast(boardarr: np.ndarray, side: ChessSide, species: PieceT, denom: float) -> np.ndarray:
    c = _count_species(boardarr, side, species)
    return np.full((10, 9), min(1.0, float(c) / denom), dtype=np.float32)


def encode_extra_hint_planes(
    boardarr: np.ndarray,
    red_to_move: bool,
    *,
    legal_iccs: list[str],
    in_check: bool,
    move_index: int | None = None,
    last_move: str | None = None,
) -> np.ndarray:
    _ = in_check
    coord = _coord_planes_cached()
    ply = _ply_broadcast_plane(move_index)[np.newaxis, ...]
    face = _kings_face_plane(boardarr)[np.newaxis, ...]
    stm_side = ChessSide.RED if red_to_move else ChessSide.BLACK
    opp_side = ChessSide.next_side(stm_side)

    stm_d = _union_move_destinations(legal_iccs)[np.newaxis, ...]
    opp_d = np.zeros((1, 10, 9), dtype=np.float32)

    lf = np.zeros((1, 10, 9), dtype=np.float32)
    lt = np.zeros((1, 10, 9), dtype=np.float32)
    lfa, ltb = _last_move_planes(last_move)
    lf[0] = lfa
    lt[0] = ltb

    ms, mo = _material_broadcast(boardarr, stm_side)
    mstm, mopp = ms[np.newaxis, ...], mo[np.newaxis, ...]
    maj_s = _major_ratio(boardarr, stm_side)[np.newaxis, ...]
    maj_o = _major_ratio(boardarr, opp_side)[np.newaxis, ...]
    cap_s = _capture_dst_union(boardarr, legal_iccs, red_to_move)[np.newaxis, ...]
    cap_o = np.zeros((1, 10, 9), dtype=np.float32)
    pwn = _pawn_progress_plane(boardarr)[np.newaxis, ...]
    kd, kr = _king_geometry(boardarr)
    kdist, krank = kd[np.newaxis, ...], kr[np.newaxis, ...]
    dens = _board_density(boardarr)[np.newaxis, ...]

    zatk = np.zeros((1, 10, 9), dtype=np.float32)
    orook = ocann = oknight = zatk
    srook = _species_dst_union(boardarr, legal_iccs, PieceT.ROOK)[np.newaxis, ...]
    scann = _species_dst_union(boardarr, legal_iccs, PieceT.CANNON)[np.newaxis, ...]
    sknight = _species_dst_union(boardarr, legal_iccs, PieceT.KNIGHT)[np.newaxis, ...]

    river = _river_band_plane()[np.newaxis, ...]
    red_h, blk_h = _half_board_planes()
    rh, bh = red_h[np.newaxis, ...], blk_h[np.newaxis, ...]
    knei_s = _king_ortho_neighbor_density(boardarr, stm_side)[np.newaxis, ...]
    knei_o = _king_ortho_neighbor_density(boardarr, opp_side)[np.newaxis, ...]
    pcap_s = _pawn_capture_union(boardarr, legal_iccs, red_to_move)[np.newaxis, ...]
    pcap_o = np.zeros((1, 10, 9), dtype=np.float32)
    ba_s = _bishop_advisor_union(boardarr, legal_iccs, red_to_move)[np.newaxis, ...]
    ba_o = np.zeros((1, 10, 9), dtype=np.float32)
    kray_s = _king_cross_empty_rays(boardarr, stm_side)[np.newaxis, ...]
    kray_o = _king_cross_empty_rays(boardarr, opp_side)[np.newaxis, ...]

    den = (2.0, 2.0, 2.0, 2.0, 2.0, 5.0)
    specs = (
        PieceT.ROOK,
        PieceT.KNIGHT,
        PieceT.CANNON,
        PieceT.BISHOP,
        PieceT.ADVISOR,
        PieceT.PAWN,
    )
    stm_counts = [_species_count_broadcast(boardarr, stm_side, sp, d)[np.newaxis, ...] for sp, d in zip(specs, den)]
    opp_counts = [_species_count_broadcast(boardarr, opp_side, sp, d)[np.newaxis, ...] for sp, d in zip(specs, den)]

    return np.concatenate(
        [
            coord,
            ply,
            face,
            stm_d,
            opp_d,
            lf,
            lt,
            mstm,
            mopp,
            kdist,
            krank,
            dens,
            maj_s,
            maj_o,
            cap_s,
            cap_o,
            pwn,
            orook,
            ocann,
            oknight,
            srook,
            scann,
            sknight,
            river,
            rh,
            bh,
            knei_s,
            knei_o,
            pcap_s,
            pcap_o,
            ba_s,
            ba_o,
            kray_s,
            kray_o,
            *stm_counts,
            *opp_counts,
        ],
        axis=0,
    ).astype(np.float32, copy=False)
