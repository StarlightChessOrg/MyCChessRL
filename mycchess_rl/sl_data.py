"""icyElephant / MyElephant 风格 XML ``.cbf`` → ``xqwl_core`` 回放，供联合策略头监督学习。"""
from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterator
from xml.etree import ElementTree as ET

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

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

PolicySources = str | Path | list[str]


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


def _red_outcome_class_from_head(head: Any) -> int:
    if head is None:
        return VALUE_LABEL_IGNORE
    rr = head.find("RecordResult")
    if rr is None or rr.text is None:
        return VALUE_LABEL_IGNORE
    s = str(rr.text).strip()
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


def _move_values_from_movelist(root: ET.Element) -> list[str]:
    ml = root.find("MoveList")
    if ml is None:
        return []
    out: list[str] = []
    for node in ml.findall("Move"):
        v = node.get("value")
        if v and v != "00-00":
            out.append(v)
    return out


def iter_joint_sl_samples_from_cbf(
    path: str | Path,
    *,
    policy_max_legal: int = POLICY_MAX_LEGAL_MOVES,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.int64, np.float32, bool]]:
    """
    在 **标准起始 FEN** 上逐步回放一局；每步 yield 走子**前**的样本。

    Yields:
        chw: (C,10,9) float32
        legal_mask: (policy_max_legal,) bool，前 L 格为 True
        action_idx: 棋谱着法在 ``sorted(legal)`` 中的下标
        value_sign: -1 / 0 / +1，终局未知时为 0 且 has_value=False
        has_value: 是否参与价值 MSE
    """
    tree = ET.parse(str(path))
    root = tree.getroot()
    head = root.find("Head")
    if head is None:
        del tree
        return
    fen_el = head.find("FEN")
    fen = (fen_el.text or "").strip() if fen_el is not None else ""
    if not _fen_matches_standard_start(fen):
        del tree
        return
    red_cls = _red_outcome_class_from_head(head)
    moves = _move_values_from_movelist(root)
    del tree, root
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
        # 独立拷贝，避免与编码/下一局缓冲区共享底层存储导致未定义行为或堆损坏
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


def _shard_filelist_for_worker(filelist: list[str]) -> list[str]:
    wi = get_worker_info()
    paths = list(filelist)
    if not paths:
        return paths
    if wi is None:
        return paths
    n = len(paths)
    per = int(math.ceil(n / float(wi.num_workers)))
    start = wi.id * per
    end = min(start + per, n)
    if start < end:
        return paths[start:end]
    sub = [paths[i] for i in range(wi.id, n, wi.num_workers)]
    return sub if sub else paths


def _dataloader_worker_init(_worker_id: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def collate_joint_sl_batch(
    batch: list[tuple[np.ndarray, np.ndarray, np.int64, np.float32, bool]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.stack([np.ascontiguousarray(np.array(b[0], dtype=np.float32, copy=True)) for b in batch], axis=0)
    m = np.stack([np.ascontiguousarray(np.array(b[1], dtype=np.bool_, copy=True)) for b in batch], axis=0)
    yi = np.stack([np.int64(b[2]) for b in batch], axis=0)
    vs = np.stack([np.float32(b[3]) for b in batch], axis=0)
    hv = np.stack([np.bool_(b[4]) for b in batch], axis=0)
    return x, m, yi, vs, hv


class JointCBFIterableDataset(IterableDataset):
    """无限打乱遍历棋谱；每 worker 分片文件列表。"""

    def __init__(self, sources: PolicySources, *, policy_max_legal: int) -> None:
        super().__init__()
        if isinstance(sources, list):
            self.filelist = [str(x) for x in sources]
        else:
            p = Path(sources)
            lines = p.read_text(encoding="utf-8").splitlines()
            self.filelist = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
        if not self.filelist:
            raise ValueError("棋谱清单为空")
        self.policy_max_legal = int(policy_max_legal)

    def __iter__(self) -> Any:
        my_files = _shard_filelist_for_worker(self.filelist)
        while True:
            rnd = random.Random()
            rnd.shuffle(my_files)
            for path in my_files:
                try:
                    for sample in iter_joint_sl_samples_from_cbf(
                        path, policy_max_legal=self.policy_max_legal
                    ):
                        yield sample
                except Exception:
                    continue


def make_joint_sl_dataloader(
    sources: PolicySources,
    batch_size: int,
    *,
    policy_max_legal: int = POLICY_MAX_LEGAL_MOVES,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int = 2,
    drop_last: bool = True,
) -> DataLoader:
    ds = JointCBFIterableDataset(sources, policy_max_legal=policy_max_legal)
    kw: dict[str, Any] = {
        "dataset": ds,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_joint_sl_batch,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = max(2, int(prefetch_factor))
        kw["worker_init_fn"] = _dataloader_worker_init
        if sys.platform == "win32":
            kw["multiprocessing_context"] = "spawn"
    return DataLoader(**kw)


def build_joint_sl_train_val_loaders(
    train_sources: PolicySources,
    val_sources: PolicySources,
    batch_size: int,
    *,
    policy_max_legal: int = POLICY_MAX_LEGAL_MOVES,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    train_loader = make_joint_sl_dataloader(
        train_sources,
        batch_size,
        policy_max_legal=policy_max_legal,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        drop_last=True,
    )
    val_loader = make_joint_sl_dataloader(
        val_sources,
        batch_size,
        policy_max_legal=policy_max_legal,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        drop_last=False,
    )
    return train_loader, val_loader
