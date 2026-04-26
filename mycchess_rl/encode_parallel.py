"""常驻进程池 CPU 特征编码，加速 ``batched_encode_roots`` 数据准备。

使用 ``multiprocessing`` 的 **spawn** 上下文创建子进程，避免主进程已初始化 CUDA 时 **fork** 带来的未定义行为。纯 NumPy/棋面编码受 GIL 限制，多线程难以提速；多进程可并行占用多核。
"""
from __future__ import annotations

import atexit
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

# 与已初始化 CUDA 的训练主进程共存时，spawn 比默认 fork 更安全。
_mp_ctx = mp.get_context("spawn")
_pool: ProcessPoolExecutor | None = None
_pool_workers: int = 0


def default_encode_workers() -> int:
    """训练脚本与 ``collect_rollout_step`` 的默认并行编码进程数。"""
    return max(2, min(8, os.cpu_count() or 4))


def pack_planes_state(state: Any) -> tuple:
    """从 ``XqwlGameState`` 拆出纯数据副本，供子进程编码（避免与主进程共享可变棋盘视图）。"""
    return (
        np.array(state.board_view(), dtype="<U1", copy=True),
        bool(state.red_to_move),
        tuple(state.legal_moves_iccs_str()),
        bool(state.in_check()),
        state.last_move_iccs,
    )


def encode_packed_planes(packed: tuple) -> np.ndarray:
    boardarr, red_to_move, legal_iccs, in_check, last_move = packed
    from mycchess_rl.chess.features import encode_model_planes

    cur_chw = encode_model_planes(
        boardarr,
        red_to_move,
        legal_iccs=list(legal_iccs),
        in_check=in_check,
        last_move=last_move,
    )
    return np.ascontiguousarray(cur_chw.astype(np.float32, copy=False))


def _map_chunksize(n_items: int, n_workers: int) -> int:
    """加大 chunksize 减少进程间 pickle/IPC 次数（大批次训练很重要）。"""
    if n_items <= 0:
        return 1
    w = max(1, n_workers)
    return max(1, n_items // (w * 8))


def _ensure_pool(workers: int) -> ProcessPoolExecutor:
    global _pool, _pool_workers
    w = max(1, int(workers))
    if _pool is not None and _pool_workers == w:
        return _pool
    if _pool is not None:
        _pool.shutdown(wait=True, cancel_futures=False)
        _pool = None
    _pool = _mp_ctx.ProcessPoolExecutor(max_workers=w)
    _pool_workers = w
    return _pool


def encode_states_parallel(states: list[Any], workers: int) -> np.ndarray:
    """返回 ``(B, C, H, W)`` float32 numpy，调用方再 ``torch.from_numpy(...).to(device)``。"""
    if not states:
        return np.zeros((0, 0, 0, 0), dtype=np.float32)
    packed = [pack_planes_state(g) for g in states]
    w_pool = max(1, min(int(workers), len(packed)))
    pool = _ensure_pool(w_pool)
    cs = _map_chunksize(len(packed), w_pool)
    planes = list(pool.map(encode_packed_planes, packed, chunksize=cs))
    return np.stack(planes, axis=0)


def encode_states_threaded(states: list[Any], workers: int) -> np.ndarray:
    """兼容旧名：已改为进程池实现，请优先使用 ``encode_states_parallel``。"""
    return encode_states_parallel(states, workers)


def shutdown_encode_pool() -> None:
    global _pool, _pool_workers
    if _pool is not None:
        _pool.shutdown(wait=True, cancel_futures=False)
        _pool = None
        _pool_workers = 0


atexit.register(shutdown_encode_pool)


def resolve_encode_workers(requested: int, n_states: int) -> int:
    """``requested<=1`` 或样本过少时返回 1；否则取 ``min(requested, n, cpu)``。"""
    if requested <= 1 or n_states <= 1:
        return 1
    return max(1, min(int(requested), n_states, os.cpu_count() or 4))
