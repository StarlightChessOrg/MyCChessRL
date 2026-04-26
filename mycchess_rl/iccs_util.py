"""ICCS 坐标串解析。"""


def parse_move_squares(move: str) -> tuple[int, int, int, int]:
    if len(move) < 5 or move[2] != "-":
        raise ValueError(f"无法解析走法: {move!r}")
    x1, y1, x2, y2 = int(move[0]), int(move[1]), int(move[3]), int(move[4])
    return x1, y1, x2, y2
