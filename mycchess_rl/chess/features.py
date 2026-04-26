"""盘面到模型输入的平面特征编码。

根张量与 `icyElephant` 的 ``game_convert`` / ``gameplay.get_board_arr`` 一致：14 路二值
棋子平面（行棋方子类在前），黑方行棋时对行维做 ``[:,::-1,:]`` 翻转，见
https://github.com/bupticybee/icyElephant/blob/master/game_convert.py
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

FEATURE_LIST: dict[str, list[str]] = {
    "red": ["A", "B", "C", "K", "N", "P", "R"],
    "black": ["a", "b", "c", "k", "n", "p", "r"],
}


def encode_picker_planes(
    boardarr: np.ndarray,
    red_to_move: bool,
    feature_list: Mapping[str, list[str]] | None = None,
) -> np.ndarray:
    fl = FEATURE_LIST if feature_list is None else feature_list
    planes: list[np.ndarray] = []
    if red_to_move:
        for ch in fl["red"]:
            planes.append(np.asarray(boardarr == ch, dtype=np.uint8))
        for ch in fl["black"]:
            planes.append(np.asarray(boardarr == ch, dtype=np.uint8))
    else:
        for ch in fl["black"]:
            planes.append(np.asarray(boardarr == ch, dtype=np.uint8))
        for ch in fl["red"]:
            planes.append(np.asarray(boardarr == ch, dtype=np.uint8))
    return np.asarray(planes, dtype=np.uint8)


def orient_planes_for_model(planes: np.ndarray, red_to_move: bool) -> np.ndarray:
    if red_to_move:
        return planes
    return planes[:, ::-1, :]


def encode_signed_seven_planes(boardarr: np.ndarray) -> np.ndarray:
    pairs = [
        ("A", "a"),
        ("B", "b"),
        ("C", "c"),
        ("K", "k"),
        ("N", "n"),
        ("P", "p"),
        ("R", "r"),
    ]
    out = np.zeros((7, 10, 9), dtype=np.float32)
    for i, (ru, bk) in enumerate(pairs):
        out[i] = (boardarr == ru).astype(np.float32) - (boardarr == bk).astype(np.float32)
    return out


def encode_model_planes(
    boardarr: np.ndarray,
    red_to_move: bool,
    *,
    legal_iccs: list[str],
    in_check: bool,
    last_move: str | None = None,
    move_index: int | None = None,
    feature_list: Mapping[str, list[str]] | None = None,
) -> np.ndarray:
    """icyElephant 风格 14 路根特征；其余参数仅为 API 兼容保留。"""
    _ = (legal_iccs, in_check, last_move, move_index)
    fl = FEATURE_LIST if feature_list is None else feature_list
    picker_u8 = encode_picker_planes(boardarr, red_to_move, fl)
    return orient_planes_for_model(picker_u8, red_to_move).astype(np.float32, copy=False)
