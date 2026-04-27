"""ICCS 坐标串解析。"""


def iccs_y_to_board_view_row(y_iccs: int) -> int:
    """ICCS 引擎纵坐标 0..9 → ``board_view`` 行（与 ``np.flip(FEN,0)`` 后编码平面一致）。"""
    return 9 - int(y_iccs)


def parse_move_squares(move: str) -> tuple[int, int, int, int]:
    if len(move) < 5 or move[2] != "-":
        raise ValueError(f"无法解析走法: {move!r}")
    x1, y1, x2, y2 = int(move[0]), int(move[1]), int(move[3]), int(move[4])
    return x1, y1, x2, y2
