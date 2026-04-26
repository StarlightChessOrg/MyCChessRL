"""非终局战术类奖励塑形：士象、担子炮、双车、过河卒、马、三子归边、中炮、空头炮、车牵炮、将门等（微弱标量）。"""
from __future__ import annotations

import numpy as np

from mycchess_rl.fen_parse import FULL_INIT_FEN, parse_fen_board

# ``board_view`` 与 ``np.flip(parse_fen_board(FEN)[0], 0)`` 一致：红在较小 y、黑在较大 y（见 FULL_INIT_FEN）
_KING_START_XY: tuple[tuple[int, int], tuple[int, int]] | None = None


def _king_start_xy_cached() -> tuple[tuple[int, int], tuple[int, int]]:
    """``(x,y)`` 红帅、黑将的初始格（与当前仓库 ``board_view`` 约定一致）。"""
    global _KING_START_XY
    if _KING_START_XY is None:
        arr, _ = parse_fen_board(FULL_INIT_FEN)
        v = np.flip(np.asarray(arr), axis=0)
        rk = np.argwhere(v == "K")
        bk = np.argwhere(v == "k")
        _KING_START_XY = (
            (int(rk[0][1]), int(rk[0][0])),
            (int(bk[0][1]), int(bk[0][0])),
        )
    return _KING_START_XY


def _in_red_palace(x: int, y: int) -> bool:
    return 3 <= x <= 5 and 0 <= y <= 2


def _in_black_palace(x: int, y: int) -> bool:
    return 3 <= x <= 5 and 7 <= y <= 9


def _palace_geom_center(*, red_king: bool) -> tuple[int, int]:
    """九宫几何中心（非 ``_own_palace_center`` 马用坐标）。"""
    return (4, 1) if red_king else (4, 8)


def _own_palace_center(red: bool) -> tuple[int, int]:
    """己方九宫近似中心（board_view：y=0 黑底，y=9 红底）。"""
    return (4, 8) if red else (4, 1)


def _knight_dest_good_for_tactics(dx: int, dy: int, sx: int, sy: int, mover_red: bool) -> bool:
    """马的一步：落点更靠近己方宫心，或落在棋盘边线，视为「较好」。"""
    oc_x, oc_y = _own_palace_center(mover_red)
    dist_before = abs(sx - oc_x) + abs(sy - oc_y)
    dist_after = abs(dx - oc_x) + abs(dy - oc_y)
    toward_own = dist_after < dist_before
    on_edge = dx in (0, 8) or dy in (0, 9)
    return bool(toward_own or on_edge)


_KNIGHT_STEPS: tuple[tuple[int, int, int, int], ...] = (
    (2, 1, 1, 0),
    (2, -1, 1, 0),
    (-2, 1, -1, 0),
    (-2, -1, -1, 0),
    (1, 2, 0, 1),
    (1, -2, 0, -1),
    (-1, 2, 0, 1),
    (-1, -2, 0, -1),
)


def _knight_pseudo_reachable(board: np.ndarray, sx: int, sy: int, mover_red: bool) -> list[tuple[int, int]]:
    """从 ``(sx,sy)`` 的马（不计将军）可达落点；蹩马、吃己子排除。"""
    out: list[tuple[int, int]] = []
    for ddx, ddy, lx, ly in _KNIGHT_STEPS:
        fx, fy = sx + lx, sy + ly
        if not (0 <= fx < 9 and 0 <= fy < 10):
            continue
        if str(board[fy, fx]).strip():
            continue
        dx, dy = sx + ddx, sy + ddy
        if not (0 <= dx < 9 and 0 <= dy < 10):
            continue
        ch = str(board[dy, dx]).strip()
        if not ch:
            out.append((dx, dy))
            continue
        is_own = ch.isupper() if mover_red else ch.islower()
        if not is_own:
            out.append((dx, dy))
    return out


def knight_flex_bonus(
    board: np.ndarray,
    x2: int,
    y2: int,
    piece: str,
    *,
    coeff: float,
) -> float:
    """刚走动的是马时：根据「较好」马步数量给微弱分（用棋盘伪合法马步，与当前行棋方无关）。"""
    if coeff == 0.0 or piece not in ("N", "n"):
        return 0.0
    mover_red = piece == "N"
    dests = _knight_pseudo_reachable(board, x2, y2, mover_red)
    if not dests:
        return 0.0
    n_good = sum(1 for dx, dy in dests if _knight_dest_good_for_tactics(dx, dy, x2, y2, mover_red))
    if n_good >= 2:
        return float(coeff * 1.0)
    if n_good == 1:
        return float(coeff * 0.45)
    return 0.0


def cross_river_pawn_bonus(piece: str, y2: int, *, coeff: float) -> float:
    """过河卒：走子后落在对方半场一侧（河界 y=4/5）。"""
    if coeff == 0.0:
        return 0.0
    if piece == "P" and y2 <= 4:
        return float(coeff)
    if piece == "p" and y2 >= 5:
        return float(coeff)
    return 0.0


def advisor_elephant_shape_bonus(board: np.ndarray, mover_red: bool, *, coeff: float) -> float:
    """士在九宫、象在己方半场的近似阵型奖励。"""
    if coeff == 0.0:
        return 0.0
    yy, xx = np.indices(board.shape)
    if mover_red:
        a_zone = (board == "A") & (yy >= 7) & (xx >= 3) & (xx <= 5)
        b_zone = (board == "B") & (yy >= 5)
    else:
        a_zone = (board == "a") & (yy <= 2) & (xx >= 3) & (xx <= 5)
        b_zone = (board == "b") & (yy <= 4)
    na = int(a_zone.sum())
    nb = int(b_zone.sum())
    if na < 1 or nb < 1:
        return 0.0
    return float(coeff * min(1.0, na / 2.0) * min(1.0, nb / 2.0))


def _line_between_occupied(board: np.ndarray, y1: int, x1: int, y2: int, x2: int) -> int:
    pieces = 0
    if x1 == x2:
        lo, hi = sorted((y1, y2))
        for y in range(lo + 1, hi):
            if str(board[y, x1]).strip():
                pieces += 1
    elif y1 == y2:
        lo, hi = sorted((x1, x2))
        for x in range(lo + 1, hi):
            if str(board[y1, x]).strip():
                pieces += 1
    return pieces


def double_cannon_bonus(board: np.ndarray, mover_red: bool, *, coeff: float) -> float:
    """两枚己方炮共线且线间恰好一枚子（担子炮近似）。"""
    if coeff == 0.0:
        return 0.0
    ch = "C" if mover_red else "c"
    ys, xs = np.where(board == ch)
    if ys.size < 2:
        return 0.0
    for i in range(ys.size):
        for j in range(i + 1, ys.size):
            y1, x1 = int(ys[i]), int(xs[i])
            y2, x2 = int(ys[j]), int(xs[j])
            if x1 != x2 and y1 != y2:
                continue
            if _line_between_occupied(board, y1, x1, y2, x2) == 1:
                return float(coeff)
    return 0.0


def rook_pair_bonus(board: np.ndarray, mover_red: bool, *, coeff: float) -> float:
    """双车同横线且间距较大（霸王车/双车压制近似）。"""
    if coeff == 0.0:
        return 0.0
    ch = "R" if mover_red else "r"
    ys, xs = np.where(board == ch)
    if ys.size < 2:
        return 0.0
    for i in range(ys.size):
        for j in range(i + 1, ys.size):
            y1, x1 = int(ys[i]), int(xs[i])
            y2, x2 = int(ys[j]), int(xs[j])
            if y1 != y2:
                continue
            if abs(x1 - x2) >= 3:
                return float(coeff)
    return 0.0


def _piece_sets(mover_red: bool) -> tuple[str, str, str, str, str]:
    """己方车/炮、对方车/炮/将（字符）。"""
    if mover_red:
        return "R", "C", "r", "c", "k"
    return "r", "c", "R", "C", "K"


def _opponent_palace_center_xqwl(mover_red: bool) -> tuple[int, int]:
    """与对方半场划分一致（同 ``central_cannon``）：黑九宫心 ``(4,1)``，红九宫心 ``(4,8)``。"""
    return (4, 1) if mover_red else (4, 8)


def three_to_edge_bonus(board: np.ndarray, mover_red: bool, *, coeff: float) -> float:
    """三子归边近似：车/马/炮在对方半场一侧翼（x≤2 或 x≥6）上至少 3 枚。"""
    if coeff == 0.0:
        return 0.0
    my_r, my_c, _, _, _ = _piece_sets(mover_red)
    yy, xx = np.indices(board.shape)
    if mover_red:
        opp_half = yy <= 4
    else:
        opp_half = yy >= 5
    left = opp_half & (xx <= 2)
    right = opp_half & (xx >= 6)
    my_n = "N" if mover_red else "n"
    majors = (board == my_r) | (board == my_c) | (board == my_n)
    nl = int((majors & left).sum())
    nr = int((majors & right).sum())
    m = max(nl, nr)
    if m < 3:
        return 0.0
    return float(coeff * min(1.0, (m - 2) / 2.0))


def central_cannon_bonus(board: np.ndarray, mover_red: bool, *, coeff: float) -> float:
    """中炮近似：己方炮在纵线 x=4 且仍在己方河界一侧。
    若对方马恰在对方九宫心且与中炮同列（x=4），视为中炮镇马，**奖励加倍**。
    """
    if coeff == 0.0:
        return 0.0
    _, my_c, _, _, _ = _piece_sets(mover_red)
    yy, xx = np.indices(board.shape)
    on_file = (board == my_c) & (xx == 4)
    if mover_red:
        on_file = on_file & (yy >= 5)
    else:
        on_file = on_file & (yy <= 4)
    if not bool(on_file.any()):
        return 0.0
    mult = 1.0
    px, py = _opponent_palace_center_xqwl(mover_red)
    opp_n = "n" if mover_red else "N"
    if str(board[py, px]).strip() == opp_n:
        mult = 2.0
    return float(coeff * mult)


def open_file_cannon_bonus(board: np.ndarray, mover_red: bool, *, coeff: float) -> float:
    """空头炮近似：己方炮与对方将在同一纵线，其间无子或恰有一枚对方子（炮架）。"""
    if coeff == 0.0:
        return 0.0
    _, my_c, _, _, opp_k = _piece_sets(mover_red)
    kys, kxs = np.where(board == opp_k)
    if kys.size == 0:
        return 0.0
    ky, kx = int(kys[0]), int(kxs[0])
    cys, cxs = np.where(board == my_c)
    best = 0.0
    for i in range(cys.size):
        cy, cx = int(cys[i]), int(cxs[i])
        if cx != kx:
            continue
        lo, hi = sorted((cy, ky))
        mids: list[str] = []
        for y in range(lo + 1, hi):
            ch = str(board[y, kx]).strip()
            if ch:
                mids.append(ch)
        if len(mids) == 0:
            best = max(best, float(coeff * 0.25))
        elif len(mids) == 1:
            ch = mids[0]
            is_opp = ch.islower() if mover_red else ch.isupper()
            if is_opp and ch != opp_k:
                best = max(best, float(coeff))
    return float(best)


def _occupied_on_line_segment(
    board: np.ndarray, y0: int, x0: int, y1: int, x1: int, y2: int, x2: int
) -> int:
    """与三点共线的整条横线或纵线上的占用格数（含端点）。"""
    if y0 == y1 == y2:
        xa, xb = min(x0, x1, x2), max(x0, x1, x2)
        n = 0
        for x in range(xa, xb + 1):
            if str(board[y0, x]).strip():
                n += 1
        return n
    if x0 == x1 == x2:
        ya, yb = min(y0, y1, y2), max(y0, y1, y2)
        n = 0
        for y in range(ya, yb + 1):
            if str(board[y, x0]).strip():
                n += 1
        return n
    return 999


def rook_pin_opposite_cannon_bonus(board: np.ndarray, mover_red: bool, *, coeff: float) -> float:
    """车牵炮近似：己方车与对方车、炮共线，且该线上仅有这三枚子（对方车炮无第三子在根线）。"""
    if coeff == 0.0:
        return 0.0
    my_R, _, opp_r, opp_c, _ = _piece_sets(mover_red)
    my_ys, my_xs = np.where(board == my_R)
    rys, rxs = np.where(board == opp_r)
    cys, cxs = np.where(board == opp_c)
    if my_ys.size == 0 or rys.size == 0 or cys.size == 0:
        return 0.0
    for mi in range(my_ys.size):
        my_y, my_x = int(my_ys[mi]), int(my_xs[mi])
        for ri in range(rys.size):
            ry, rx = int(rys[ri]), int(rxs[ri])
            for ci in range(cys.size):
                cy, cx = int(cys[ci]), int(cxs[ci])
                if not (my_y == ry == cy or my_x == rx == cx):
                    continue
                if _occupied_on_line_segment(board, my_y, my_x, ry, rx, cy, cx) != 3:
                    continue
                return float(coeff)
    return 0.0


def opponent_king_gate_block_bonus(board: np.ndarray, mover_red: bool, *, coeff: float) -> float:
    """对方将在九宫内「门」被堵：宫内正交空格步很少时给走子方微弱奖。"""
    if coeff == 0.0:
        return 0.0
    opp_k = "k" if mover_red else "K"
    pos = np.argwhere(board == opp_k)
    if pos.size == 0:
        return 0.0
    ky, kx = int(pos[0][0]), int(pos[0][1])
    in_pal = _in_black_palace if mover_red else _in_red_palace
    if not in_pal(kx, ky):
        return 0.0
    n_empty = 0
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny, nx = ky + dy, kx + dx
        if not in_pal(nx, ny):
            continue
        if not str(board[ny, nx]).strip():
            n_empty += 1
    if n_empty > 1:
        return 0.0
    return float(coeff * (0.55 + 0.45 * (1 - n_empty)))


def missing_advisor_vs_double_rook_penalty(board: np.ndarray, mover_red: bool, *, coeff: float) -> float:
    """己方缺士（士少于 2）且对方有两车：惩罚走子方（系数为正则奖励减去该值）。"""
    if coeff == 0.0:
        return 0.0
    if mover_red:
        n_a = int(np.sum(board == "A"))
        n_opp_r = int(np.sum(board == "r"))
    else:
        n_a = int(np.sum(board == "a"))
        n_opp_r = int(np.sum(board == "R"))
    if n_a >= 2 or n_opp_r < 2:
        return 0.0
    return float(-coeff)


def double_advisor_king_center_bonus(board: np.ndarray, mover_red: bool, *, coeff: float) -> float:
    """己方恰双士且将/帅在九宫几何中心。"""
    if coeff == 0.0:
        return 0.0
    my_k = "K" if mover_red else "k"
    my_a = "A" if mover_red else "a"
    if int(np.sum(board == my_a)) != 2:
        return 0.0
    pos = np.argwhere(board == my_k)
    if pos.size == 0:
        return 0.0
    ky, kx = int(pos[0][0]), int(pos[0][1])
    cx, cy = _palace_geom_center(red_king=mover_red)
    if kx == cx and ky == cy:
        return float(coeff)
    return 0.0


def king_near_start_bonus(board: np.ndarray, mover_red: bool, *, coeff: float) -> float:
    """己将/帅距初始位曼哈顿越近奖越大：``coeff * max(0, 1 - dist/dmax)``。"""
    if coeff == 0.0:
        return 0.0
    my_k = "K" if mover_red else "k"
    pos = np.argwhere(board == my_k)
    if pos.size == 0:
        return 0.0
    ky, kx = int(pos[0][0]), int(pos[0][1])
    (sx_r, sy_r), (sx_b, sy_b) = _king_start_xy_cached()
    sx, sy = (sx_r, sy_r) if mover_red else (sx_b, sy_b)
    dist = abs(kx - sx) + abs(ky - sy)
    dmax = 10.0
    return float(coeff * max(0.0, 1.0 - float(dist) / dmax))


def tactics_shaping_total(
    board: np.ndarray,
    piece_at_dst: str,
    x2: int,
    y2: int,
    *,
    last_mover_red: bool,
    coeff_ae: float,
    coeff_double_cannon: float,
    coeff_rook_pair: float,
    coeff_cross_pawn: float,
    coeff_knight_flex: float,
    coeff_three_edge: float,
    coeff_central_cannon: float,
    coeff_open_cannon: float,
    coeff_rook_pin_cannon: float,
    coeff_opp_king_gate: float,
    coeff_miss_adv_double_rook: float,
    coeff_double_adv_king_center: float,
    coeff_king_near_start: float,
) -> float:
    """走子后局面上的战术塑形总和（仅当对应 coeff 非 0 时计算）。"""
    if all(
        c == 0.0
        for c in (
            coeff_ae,
            coeff_double_cannon,
            coeff_rook_pair,
            coeff_cross_pawn,
            coeff_knight_flex,
            coeff_three_edge,
            coeff_central_cannon,
            coeff_open_cannon,
            coeff_rook_pin_cannon,
            coeff_opp_king_gate,
            coeff_miss_adv_double_rook,
            coeff_double_adv_king_center,
            coeff_king_near_start,
        )
    ):
        return 0.0
    total = 0.0
    if coeff_ae != 0.0:
        total += advisor_elephant_shape_bonus(board, last_mover_red, coeff=coeff_ae)
    if coeff_double_cannon != 0.0:
        total += double_cannon_bonus(board, last_mover_red, coeff=coeff_double_cannon)
    if coeff_rook_pair != 0.0:
        total += rook_pair_bonus(board, last_mover_red, coeff=coeff_rook_pair)
    if coeff_cross_pawn != 0.0:
        total += cross_river_pawn_bonus(piece_at_dst, y2, coeff=coeff_cross_pawn)
    if coeff_knight_flex != 0.0:
        total += knight_flex_bonus(board, x2, y2, piece_at_dst, coeff=coeff_knight_flex)
    if coeff_three_edge != 0.0:
        total += three_to_edge_bonus(board, last_mover_red, coeff=coeff_three_edge)
    if coeff_central_cannon != 0.0:
        total += central_cannon_bonus(board, last_mover_red, coeff=coeff_central_cannon)
    if coeff_open_cannon != 0.0:
        total += open_file_cannon_bonus(board, last_mover_red, coeff=coeff_open_cannon)
    if coeff_rook_pin_cannon != 0.0:
        total += rook_pin_opposite_cannon_bonus(board, last_mover_red, coeff=coeff_rook_pin_cannon)
    if coeff_opp_king_gate != 0.0:
        total += opponent_king_gate_block_bonus(board, last_mover_red, coeff=coeff_opp_king_gate)
    if coeff_miss_adv_double_rook != 0.0:
        total += missing_advisor_vs_double_rook_penalty(
            board, last_mover_red, coeff=coeff_miss_adv_double_rook
        )
    if coeff_double_adv_king_center != 0.0:
        total += double_advisor_king_center_bonus(
            board, last_mover_red, coeff=coeff_double_adv_king_center
        )
    if coeff_king_near_start != 0.0:
        total += king_near_start_bonus(board, last_mover_red, coeff=coeff_king_near_start)
    return float(total)
