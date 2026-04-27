"""根平面 CPU 编码：支持 inline / 线程池 / 进程池。

``encode_model_planes`` 本身很轻（少量 NumPy），**默认 inline** 在主进程内批量
``np.stack`` 后一次拷到 GPU，避免进程池在「每步 × 两路前向」场景下的 **pickle/IPC**
反压 GPU。需要压榨多核且编码变重时可用 ``thread``；与 CUDA 共存且坚持子进程隔离时用
``process``（``spawn`` + ``ProcessPoolExecutor``）。
"""
from __future__ import annotations

import atexit
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any

import numpy as np

_mp_ctx = mp.get_context("spawn")
_proc_pool: ProcessPoolExecutor | None = None
_proc_pool_workers: int = 0

_thread_pool: ThreadPoolExecutor | None = None
_thread_pool_workers: int = 0


def default_encode_workers() -> int:
    """并行编码（仅 thread/process 后端）的默认 worker 数。"""
    return max(2, min(8, os.cpu_count() or 4))


def pack_planes_state(state: Any) -> tuple:
    """从 ``XqwlGameState`` 拆出纯数据副本，供子线程/子进程编码。"""
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


def encode_states_inline(states: list[Any]) -> np.ndarray:
    """主进程顺序编码；``np.stack`` 后由调用方一次 ``.to(device)``。"""
    if not states:
        return np.zeros((0, 0, 0, 0), dtype=np.float32)
    planes = [encode_packed_planes(pack_planes_state(g)) for g in states]
    return np.stack(planes, axis=0)


def _map_chunksize(n_items: int, n_workers: int) -> int:
    if n_items <= 0:
        return 1
    w = max(1, n_workers)
    return max(1, n_items // (w * 8))


def _ensure_proc_pool(workers: int) -> ProcessPoolExecutor:
    global _proc_pool, _proc_pool_workers
    w = max(1, int(workers))
    if _proc_pool is not None and _proc_pool_workers == w:
        return _proc_pool
    if _proc_pool is not None:
        _proc_pool.shutdown(wait=True, cancel_futures=False)
        _proc_pool = None
    _proc_pool = ProcessPoolExecutor(max_workers=w, mp_context=_mp_ctx)
    _proc_pool_workers = w
    return _proc_pool


def _ensure_thread_pool(workers: int) -> ThreadPoolExecutor:
    global _thread_pool, _thread_pool_workers
    w = max(1, int(workers))
    if _thread_pool is not None and _thread_pool_workers == w:
        return _thread_pool
    if _thread_pool is not None:
        _thread_pool.shutdown(wait=False, cancel_futures=False)
        _thread_pool = None
    _thread_pool = ThreadPoolExecutor(max_workers=w, thread_name_prefix="xqp_enc")
    _thread_pool_workers = w
    return _thread_pool


def encode_states_process_pool(states: list[Any], workers: int) -> np.ndarray:
    """多进程编码（高 IPC 成本；仅在大批量、编码明显变重时考虑）。"""
    if not states:
        return np.zeros((0, 0, 0, 0), dtype=np.float32)
    packed = [pack_planes_state(g) for g in states]
    w_pool = max(1, min(int(workers), len(packed)))
    pool = _ensure_proc_pool(w_pool)
    cs = _map_chunksize(len(packed), w_pool)
    planes = list(pool.map(encode_packed_planes, packed, chunksize=cs))
    return np.stack(planes, axis=0)


def encode_states_thread_pool(states: list[Any], workers: int) -> np.ndarray:
    """多线程编码（适合 NumPy 在 C 层释放 GIL 的片段；IPC 低于进程池）。"""
    if not states:
        return np.zeros((0, 0, 0, 0), dtype=np.float32)
    packed = [pack_planes_state(g) for g in states]
    w_pool = max(1, min(int(workers), len(packed)))
    pool = _ensure_thread_pool(w_pool)
    planes = list(pool.map(encode_packed_planes, packed))
    return np.stack(planes, axis=0)


def encode_packed_list_thread_pool(packed_list: list[tuple], workers: int) -> np.ndarray:
    """对已是 ``pack_planes_state`` 结果的列表做多线程 ``encode_packed_planes``（供 SL collate 等复用）。"""
    if not packed_list:
        return np.zeros((0, 0, 0, 0), dtype=np.float32)
    w_pool = resolve_encode_workers(int(workers), len(packed_list))
    if w_pool <= 1:
        planes = [encode_packed_planes(p) for p in packed_list]
    else:
        pool = _ensure_thread_pool(w_pool)
        planes = list(pool.map(encode_packed_planes, packed_list))
    return np.stack(planes, axis=0)


def encode_states_parallel(states: list[Any], workers: int) -> np.ndarray:
    """兼容旧名：等同 ``encode_states_process_pool``。"""
    return encode_states_process_pool(states, workers)


def encode_states_threaded(states: list[Any], workers: int) -> np.ndarray:
    """兼容旧名：等同 ``encode_states_thread_pool``。"""
    return encode_states_thread_pool(states, workers)


def shutdown_encode_pool() -> None:
    global _proc_pool, _proc_pool_workers, _thread_pool, _thread_pool_workers
    if _proc_pool is not None:
        _proc_pool.shutdown(wait=True, cancel_futures=False)
        _proc_pool = None
        _proc_pool_workers = 0
    if _thread_pool is not None:
        _thread_pool.shutdown(wait=True, cancel_futures=False)
        _thread_pool = None
        _thread_pool_workers = 0


atexit.register(shutdown_encode_pool)


def resolve_encode_workers(requested: int, n_states: int) -> int:
    """``requested<=1`` 或样本过少时返回 1；否则取 ``min(requested, n, cpu)``。"""
    if requested <= 1 or n_states <= 1:
        return 1
    return max(1, min(int(requested), n_states, os.cpu_count() or 4))
