"""icyElephant / MyElephant 风格：``xmltodict`` 读 ``.cbf``，主进程无限打乱 + ``xqwl_core`` 回放（与 ``train_policy_torch`` 数据流一致，无 DataLoader）。"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import xmltodict

from mycchess_rl.chess.rationale import (
    POLICY_MAX_LEGAL_MOVES,
    RED_OUTCOME_DRAW,
    RED_OUTCOME_LOSS,
    RED_OUTCOME_WIN,
    STM_OUTCOME_LOSS,
    STM_OUTCOME_WIN,
    VALUE_LABEL_IGNORE,
    stm_outcome_class_from_red_outcome,
)
from mycchess_rl.encode_parallel import encode_states_inline
from mycchess_rl.fen_parse import FULL_INIT_FEN, parse_fen_board
from mycchess_rl.xqwl_state import XqwlGameState


def discover_cbf_files(root: Path | str, *, recursive: bool = True) -> list[str]:
    r = Path(root).expanduser().resolve()
    if not r.is_dir():
        raise NotADirectoryError(str(r))
    paths: list[Path] = []
    if recursive:
        for p in r.rglob("*"):
            if p.is_file() and p.suffix.lower() == ".cbf":
                paths.append(p)
    else:
        for p in r.iterdir():
            if p.is_file() and p.suffix.lower() == ".cbf":
                paths.append(p)
    out = sorted(str(p.resolve()) for p in paths)
    if not out:
        scope = "（含子目录）" if recursive else "（仅一层）"
        raise FileNotFoundError(f"在 {r}{scope} 未找到 .cbf 文件")
    return out


def split_paths_train_test(
    paths: list[str], train_ratio: float, *, seed: int = 42
) -> tuple[list[str], list[str]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio 应在 (0,1) 内，收到 {train_ratio}")
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    gap = int(len(shuffled) * train_ratio)
    if gap <= 0 or gap >= len(shuffled):
        raise ValueError(f"划分后 train 或 test 为空（n={len(shuffled)}, train_ratio={train_ratio}）")
    return shuffled[:gap], shuffled[gap:]


def _fen_matches_standard_start(fen: str) -> bool:
    """当前 ``xqwl_core`` 无 ``set_fen``，仅支持从标准初始局面开始的棋谱。"""
    try:
        b1, r1 = parse_fen_board(fen.strip())
        b2, r2 = parse_fen_board(FULL_INIT_FEN)
    except ValueError:
        return False
    return bool(np.array_equal(b1, b2) and r1 == r2)


def _xml_text(node: Any) -> str:
    if node is None:
        return ""
    if isinstance(node, dict):
        return str(node.get("#text", node.get("@value", ""))).strip()
    return str(node).strip()


def _normalize_move_entries(move_node: Any) -> list[dict[str, Any]]:
    """与 MyElephant ``xml_samples`` 一致：单条 Move 时 xmltodict 返回 dict。"""
    if isinstance(move_node, list):
        return move_node
    if isinstance(move_node, dict):
        return [move_node]
    return []


def red_outcome_class_from_head_dict(head: Any) -> int:
    """与 MyElephant ``red_outcome_class_from_head`` 一致（dict Head）。"""
    if not isinstance(head, dict):
        return VALUE_LABEL_IGNORE
    rr = head.get("RecordResult")
    if rr is None:
        return VALUE_LABEL_IGNORE
    s = _xml_text(rr)
    if not s:
        return VALUE_LABEL_IGNORE
    try:
        code = int(s)
    except ValueError:
        return VALUE_LABEL_IGNORE
    if code == 1:
        return RED_OUTCOME_WIN
    if code == 2:
        return RED_OUTCOME_LOSS
    if code in (3, 4):
        return RED_OUTCOME_DRAW
    return VALUE_LABEL_IGNORE


def _load_cbf_dict(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    return xmltodict.parse(text)


def iter_joint_sl_samples_from_cbf(
    path: str | Path,
    *,
    policy_max_legal: int = POLICY_MAX_LEGAL_MOVES,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.int64, np.float32, bool]]:
    """
    在 **标准起始 FEN** 上逐步回放一局；每步 yield 走子**前**的样本。
    解析路径与 MyElephant ``convert_game`` / icyElephant 棋谱结构一致（``xmltodict``）。
    """
    doc: dict[str, Any] | None = None
    try:
        try:
            doc = _load_cbf_dict(path)
        except Exception:
            return
        rec = doc.get("ChineseChessRecord")
        if not isinstance(rec, dict):
            return
        head = rec.get("Head")
        if not isinstance(head, dict):
            return
        fen = _xml_text(head.get("FEN"))
        if not _fen_matches_standard_start(fen):
            return
        red_cls = red_outcome_class_from_head_dict(head)
        ml = rec.get("MoveList") or {}
        moves_raw = ml.get("Move")
        moves = [
            str(m["@value"])
            for m in _normalize_move_entries(moves_raw)
            if m.get("@value") not in (None, "00-00")
        ]
        st = XqwlGameState()
        st.reset()
        for mv in moves:
            legs = sorted(st.legal_moves_iccs_str())
            if not legs:
                return
            if mv not in legs:
                return
            if len(legs) > policy_max_legal:
                raise ValueError(
                    f"合法着法数 {len(legs)} 超过 policy_max_legal={policy_max_legal} file={path!r}"
                )
            mask = np.zeros((policy_max_legal,), dtype=np.bool_)
            mask[: len(legs)] = True
            idx = int(legs.index(mv))
            chw = np.array(encode_states_inline([st])[0], dtype=np.float32, copy=True)
            stm_cls = int(stm_outcome_class_from_red_outcome(red_cls, bool(st.red_to_move)))
            if stm_cls == VALUE_LABEL_IGNORE:
                has_v = False
                vs = np.float32(0.0)
            else:
                has_v = True
                if stm_cls == STM_OUTCOME_WIN:
                    vs = np.float32(1.0)
                elif stm_cls == STM_OUTCOME_LOSS:
                    vs = np.float32(-1.0)
                else:
                    vs = np.float32(0.0)
            yield (chw, mask, np.int64(idx), vs, has_v)
            st.make_move_iccs(mv)
    finally:
        if doc is not None:
            doc.clear()


def infinite_shuffled_joint_samples(
    filelist: list[str],
    *,
    policy_max_legal: int = POLICY_MAX_LEGAL_MOVES,
    rng: random.Random | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.int64, np.float32, bool]]:
    """与 MyElephant ``SuccessorPolicyIterableDataset`` 同构：无限打乱文件列表后逐局 yield 样本（主进程，无 DataLoader）。"""
    rnd = rng if rng is not None else random.Random()
    fl = [str(x) for x in filelist]
    if not fl:
        raise ValueError("棋谱文件列表为空")
    while True:
        rnd.shuffle(fl)
        yielded_round = False
        for path in fl:
            try:
                for sample in iter_joint_sl_samples_from_cbf(path, policy_max_legal=policy_max_legal):
                    yielded_round = True
                    yield sample
            except Exception:
                continue
        if not yielded_round:
            raise RuntimeError(
                "整轮打乱后未产生任何训练样本：请确认 .cbf 为 icy 格式且 Head/FEN 与标准开局一致（当前 xqwl 无法 set_fen）"
            )


def collate_joint_sl_batch(
    batch: list[tuple[np.ndarray, np.ndarray, np.int64, np.float32, bool]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.stack([np.ascontiguousarray(np.array(b[0], dtype=np.float32, copy=True)) for b in batch], axis=0)
    m = np.stack([np.ascontiguousarray(np.array(b[1], dtype=np.bool_, copy=True)) for b in batch], axis=0)
    yi = np.stack([np.int64(b[2]) for b in batch], axis=0)
    vs = np.stack([np.float32(b[3]) for b in batch], axis=0)
    hv = np.stack([np.bool_(b[4]) for b in batch], axis=0)
    return x, m, yi, vs, hv


def next_collated_batch(
    gen: Iterator[tuple[np.ndarray, np.ndarray, np.int64, np.float32, bool]],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """从生成器凑满 ``batch_size`` 条后 ``collate``（主线程，无 DataLoader）。"""
    buf: list[tuple[np.ndarray, np.ndarray, np.int64, np.float32, bool]] = []
    while len(buf) < batch_size:
        buf.append(next(gen))
    return collate_joint_sl_batch(buf)
