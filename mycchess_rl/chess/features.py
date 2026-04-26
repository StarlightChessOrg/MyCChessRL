"""盘面到模型输入的平面特征编码。"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from mycchess_rl.chess.plane_extras import encode_extra_hint_planes
from mycchess_rl.chess.rationale import encode_rationale_planes
from mycchess_rl.iccs_util import parse_move_squares

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
    _ = feature_list
    pieces = encode_signed_seven_planes(boardarr)
    rationale = encode_rationale_planes(boardarr, red_to_move, in_check, legal_iccs)
    extra = encode_extra_hint_planes(
        boardarr,
        red_to_move,
        legal_iccs=legal_iccs,
        in_check=in_check,
        move_index=move_index,
        last_move=last_move,
    )
    return np.concatenate([pieces, rationale, extra], axis=0)
