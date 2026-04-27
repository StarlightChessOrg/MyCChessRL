"""XML ``.cbf`` 监督学习：联合合法槽 softmax + 价值 MSE。

每 epoch：**整轮训练集**（打乱棋谱文件顺序、每文件一轮）→ **整轮验证集** → 若验证 loss 更优则更新 ``best.pt``，并**总是**写入 ``last.pt``（对齐 YOLO 习惯）。进度条用 ``tqdm``。``--checkpoint`` 可选；SL 存盘可续优化器与 ``epoch``/``global_step``。
"""
from __future__ import annotations

import os

# 须在 import numpy/torch 之前：OpenBLAS/MKL 常在库初始化时读取线程环境变量
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path

from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

from mycchess_rl.encode_parallel import default_encode_workers
from mycchess_rl.model import JointPolicyValueNet, load_policy_value_for_play, torch_load_checkpoint
from mycchess_rl.sl_data import (
    count_joint_sl_samples_paths_parallel,
    discover_cbf_files,
    iter_collated_batches_from_finite_samples,
    iter_epoch_joint_samples_shuffled,
    split_paths_train_test,
    thread_prefetch_iterator,
)

_LOG = logging.getLogger("mycchess_rl.train_sl")


def _batch_tensors_to_device(
    x_np: np.ndarray,
    m_np: np.ndarray,
    yi_np: np.ndarray,
    vs_np: np.ndarray,
    hv_np: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """从 NumPy 拷入新张量，避免 ``from_numpy`` 与底层缓冲生命周期/CUDA 异步交互导致堆损坏。"""
    x = torch.tensor(np.ascontiguousarray(x_np), dtype=torch.float32, device=device)
    mask = torch.tensor(np.ascontiguousarray(m_np), dtype=torch.bool, device=device)
    tgt = torch.tensor(np.ascontiguousarray(yi_np.astype(np.int64, copy=False)), dtype=torch.long, device=device)
    v_sign = torch.tensor(np.ascontiguousarray(vs_np), dtype=torch.float32, device=device)
    has_v = torch.tensor(np.ascontiguousarray(hv_np), dtype=torch.bool, device=device)
    return x, mask, tgt, v_sign, has_v


def _setup_logging(log_file: Path | None) -> None:
    _LOG.setLevel(logging.INFO)
    _LOG.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)
    _LOG.addHandler(h)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        _LOG.addHandler(fh)


class _ExpVal:
    def __init__(self, exp_a: float = 0.97) -> None:
        self.val: float | None = None
        self.exp_a = exp_a

    def update(self, newval: float) -> None:
        if self.val is None:
            self.val = float(newval)
        else:
            self.val = self.exp_a * self.val + (1 - self.exp_a) * float(newval)

    def get(self) -> float | None:
        return None if self.val is None else round(self.val, 4)


def _save_ckpt(
    path: Path,
    model: JointPolicyValueNet,
    opt: torch.optim.Optimizer,
    *,
    epoch: int,
    global_step: int,
    best_val: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "in_channels": model.in_channels,
            "num_res_layers": model.num_res_layers,
            "filters": model.filters,
            "policy_max_legal": model.policy_max_legal,
            "value_scale": model.value_scale,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "best_val_loss": float(best_val),
            "kind": "sl",
        },
        path,
    )


def _raw_has_sl_epoch(raw: dict) -> bool:
    """含整数 ``epoch`` 时视为本脚本 SL 存盘（含 bootstrap 的 ``epoch=-1``），可续优化器与步数。"""
    if "epoch" not in raw or isinstance(raw["epoch"], bool):
        return False
    try:
        int(raw["epoch"])
    except (TypeError, ValueError):
        return False
    return True


_SL_COUNT_CACHE = "sl_sample_counts.json"


def _paths_fingerprint(paths: list[str]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.encode("utf-8", errors="surrogateescape"))
        h.update(b"\x00")
    return h.hexdigest()


def _resolve_sample_counts(
    save_dir: Path,
    train_files: list[str],
    val_files: list[str],
    policy_max_legal: int,
    *,
    force_recount: bool,
    count_workers: int,
) -> tuple[int, int, bool]:
    """返回 ``(n_train_samples, n_val_samples, from_cache)``。"""
    cache_path = save_dir / _SL_COUNT_CACHE
    t_fp = _paths_fingerprint(train_files)
    v_fp = _paths_fingerprint(val_files)
    if not force_recount and cache_path.is_file():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                raw.get("train_fingerprint") == t_fp
                and raw.get("val_fingerprint") == v_fp
                and int(raw.get("policy_max_legal", -1)) == int(policy_max_legal)
            ):
                return int(raw["train_samples"]), int(raw["val_samples"]), True
        except (OSError, TypeError, ValueError, KeyError):
            pass
    n_train = count_joint_sl_samples_paths_parallel(
        train_files,
        policy_max_legal=policy_max_legal,
        max_workers=count_workers,
        tqdm_desc="[计数·非训练] train 棋谱文件",
    )
    n_val = count_joint_sl_samples_paths_parallel(
        val_files,
        policy_max_legal=policy_max_legal,
        max_workers=count_workers,
        tqdm_desc="[计数·非训练] val 棋谱文件",
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "train_fingerprint": t_fp,
                "val_fingerprint": v_fp,
                "policy_max_legal": int(policy_max_legal),
                "train_samples": n_train,
                "val_samples": n_val,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return n_train, n_val, False


def main() -> None:
    p = argparse.ArgumentParser(description="MyCChessRL：cbf 监督学习（JointPolicyValueNet）")
    p.add_argument("--cbf-root", type=Path, default=None, help="递归搜集该目录下 .cbf（与 --cbf-manifest 二选一）")
    p.add_argument(
        "--cbf-manifest",
        type=Path,
        default=None,
        help="文本清单：每行一个 .cbf 绝对或相对路径；# 开头为注释",
    )
    p.add_argument("--cbf-shallow", action="store_true", help="仅 --cbf-root 一层目录，不递归")
    p.add_argument("--train-ratio", type=float, default=0.95, help="棋谱文件 train/val 划分比例")
    p.add_argument("--data-seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="本轮要跑的 epoch 数（续训时在已完成的 epoch 之后再跑这么多个）",
    )
    p.add_argument(
        "--recount-samples",
        action="store_true",
        help="忽略 save-dir 下样本数缓存，强制重新统计 train/val 条数",
    )
    p.add_argument(
        "--count-workers",
        type=int,
        default=0,
        help="统计样本条数时的进程数；0 表示自动 min(16, CPU)。1 表示单进程。使用 spawn 子进程，与主进程已加载 CUDA 兼容。",
    )
    p.add_argument(
        "--encode-workers",
        type=int,
        default=0,
        help="每个 batch 内根平面编码线程数；0 表示自动（约 min(8, CPU)）；1 强制单线程。样本在回放阶段仅打包，编码在 collate 并行。",
    )
    p.add_argument(
        "--prefetch-batches",
        type=int,
        default=4,
        help="后台线程预取已 collate 的 batch 数，与 GPU 前向重叠；0 关闭。",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--value-loss-weight", type=float, default=0.25, help="价值 MSE 相对策略 CE 权重")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="已有 .pt：加载权重；若为 SL 训练保存（含 epoch 字段）则同时恢复 AdamW 与进度。省略则在 --save-dir 下自动创建 bootstrap.pt 再开训",
    )
    p.add_argument("--save-dir", type=Path, default=Path("runs"))
    p.add_argument("--log-file", type=Path, default=None)
    args = p.parse_args()

    _setup_logging(args.log_file)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.cbf_root is not None:
        all_cbf = discover_cbf_files(args.cbf_root, recursive=not args.cbf_shallow)
        train_files, val_files = split_paths_train_test(all_cbf, float(args.train_ratio), seed=int(args.data_seed))
        _LOG.info(
            "数据: cbf-root=%s | 共 %d 局 -> train %d / val %d",
            args.cbf_root.resolve(),
            len(all_cbf),
            len(train_files),
            len(val_files),
        )
    elif args.cbf_manifest is not None:
        raw = [
            ln.strip()
            for ln in args.cbf_manifest.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        if not raw:
            _LOG.error("清单为空: %s", args.cbf_manifest)
            raise SystemExit(2)
        train_files, val_files = split_paths_train_test(raw, float(args.train_ratio), seed=int(args.data_seed))
        _LOG.info(
            "数据: manifest=%s | %d 条路径 -> train %d / val %d",
            args.cbf_manifest.resolve(),
            len(raw),
            len(train_files),
            len(val_files),
        )
    else:
        _LOG.error("请指定 --cbf-root 或 --cbf-manifest")
        raise SystemExit(2)

    _LOG.info(
        "监督学习（YOLO 式 epoch）：每轮 train 全量 → val 全量 → 更优则 best.pt，且每轮 last.pt；"
        "batch_size=%d；样本条数缓存在 %s（--recount-samples 强制重算）；"
        "首次统计可用多进程（--count-workers，默认自动）；"
        "加载侧 collate 多线程编码（--encode-workers）与 batch 预取（--prefetch-batches）。",
        int(args.batch_size),
        _SL_COUNT_CACHE,
    )

    device = torch.device("cpu")
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")

    epoch_begin = 0
    global_step = 0
    best_val = float("inf")

    ck = args.checkpoint
    if ck is None:
        model = JointPolicyValueNet().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        boot = save_dir / "bootstrap.pt"
        _save_ckpt(boot, model, opt, epoch=-1, global_step=0, best_val=float("inf"))
        _LOG.info("未指定 --checkpoint，已随机初始化并写入 -> %s", boot.resolve())
    else:
        cp = Path(ck)
        if not cp.is_file():
            _LOG.error("找不到 --checkpoint: %s", cp.resolve())
            raise SystemExit(2)
        model, _fl = load_policy_value_for_play(cp, device)
        opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        raw = torch_load_checkpoint(cp, device)
        if _raw_has_sl_epoch(raw):
            if "optimizer" in raw:
                try:
                    opt.load_state_dict(raw["optimizer"])
                    _LOG.info("已恢复 AdamW（SL 续训）")
                except Exception as e:
                    _LOG.warning("优化器状态不兼容，已跳过（新 AdamW）: %s", e)
            epoch_begin = int(raw.get("epoch", -1)) + 1
            global_step = int(raw.get("global_step", 0))
            bvl = raw.get("best_val_loss")
            if type(bvl) in (int, float):
                best_val = float(bvl)
            _LOG.info(
                "从 SL checkpoint 续训: 下一 epoch=%d | global_step=%d | best_val_loss=%.4f | %s",
                epoch_begin,
                global_step,
                best_val,
                cp.resolve(),
            )
        else:
            _LOG.info("从权重启动（新 AdamW，仅 model）: %s", cp.resolve())

    for g in opt.param_groups:
        g["lr"] = float(args.lr)
    pm = model.policy_max_legal
    bs = int(args.batch_size)
    cw = int(args.count_workers)
    if cw <= 0:
        cw = max(1, min(16, (os.cpu_count() or 4)))
    ew_arg = int(args.encode_workers)
    encode_workers = default_encode_workers() if ew_arg <= 0 else max(1, ew_arg)
    prefetch_batches = max(0, int(args.prefetch_batches))

    epoch = -1
    training_started = False
    t0 = time.perf_counter()
    try:
        n_train_samples, n_val_samples, counts_cached = _resolve_sample_counts(
            save_dir,
            train_files,
            val_files,
            pm,
            force_recount=bool(args.recount_samples),
            count_workers=cw,
        )
        if counts_cached:
            _LOG.info("样本条数已从缓存读取（未启动多进程扫描）")
        else:
            _LOG.info("样本条数统计完成（多进程 spawn，count_workers=%d）", cw)
        if n_train_samples <= 0:
            raise RuntimeError("train 集样本数为 0：请检查 .cbf 与标准开局 FEN")
        n_train_batches = (n_train_samples + bs - 1) // bs
        n_val_batches = (n_val_samples + bs - 1) // bs if n_val_samples > 0 else 0
        _LOG.info(
            "样本计数: train=%d 条 → %d 批 | val=%d 条 → %d 批",
            n_train_samples,
            n_train_batches,
            n_val_samples,
            n_val_batches,
        )
        _LOG.info(
            "数据吞吐: encode_workers=%d | prefetch_batches=%d",
            encode_workers,
            prefetch_batches,
        )
        vl_w = float(args.value_loss_weight)
        _LOG.info(
            "损失 total = pol_CE + (%.4g × val_MSE)；value_scale=%.4g。"
            "训练初期价值头未拟合时 val 项常远大于 pol（tot 可到十几～二十以上），"
            "随 val_MSE 下降后 total 会主要由 pol≈log(合法着法数) 主导（常见约 3～6）。",
            vl_w,
            float(model.value_scale),
        )

        for k in range(int(args.epochs)):
            epoch = epoch_begin + k
            exp_loss = _ExpVal()
            exp_acc = _ExpVal()
            model.train()

            ep_tr_rng = random.Random(int(args.data_seed) + epoch * 1_000_003 + 7)
            train_base = iter_collated_batches_from_finite_samples(
                iter_epoch_joint_samples_shuffled(train_files, policy_max_legal=pm, rng=ep_tr_rng),
                bs,
                drop_last=False,
                encode_workers=encode_workers,
            )
            train_iter = (
                thread_prefetch_iterator(train_base, prefetch_batches)
                if prefetch_batches > 0
                else train_base
            )
            pbar_tr = tqdm(
                train_iter,
                total=n_train_batches,
                desc=f"epoch {epoch} train",
                unit="batch",
                leave=True,
                mininterval=0.0,
                miniters=1,
                dynamic_ncols=True,
            )
            train_bi = 0
            training_started = True
            for x_np, m_np, yi_np, vs_np, hv_np in pbar_tr:
                train_bi += 1
                x, mask, tgt, v_sign, has_v = _batch_tensors_to_device(x_np, m_np, yi_np, vs_np, hv_np, device)
                opt.zero_grad(set_to_none=True)
                logits_m, v_pred = model(x)
                logits_masked = logits_m.masked_fill(~mask, -1e9)
                loss_p = F.cross_entropy(logits_masked, tgt)
                target_v = v_sign * float(model.value_scale)
                if has_v.any():
                    loss_v = F.mse_loss(v_pred[has_v], target_v[has_v])
                else:
                    loss_v = torch.zeros((), device=device)
                loss = loss_p + vl_w * loss_v
                loss_p_det = float(loss_p.detach().item())
                loss_v_w_det = vl_w * float(loss_v.detach().item())
                loss.backward()
                opt.step()
                global_step += 1
                with torch.no_grad():
                    pred = logits_masked.argmax(dim=-1)
                    acc = float((pred == tgt).float().mean().item())
                raw_l = float(loss.item())
                exp_loss.update(raw_l)
                exp_acc.update(acc * 100.0)
                el_b = exp_loss.get()
                ea_b = exp_acc.get()
                ema_l = float(el_b) if el_b is not None else raw_l
                ema_a = float(ea_b) if ea_b is not None else float(acc * 100.0)
                pbar_tr.set_postfix_str(
                    f"tot={raw_l:.4f}(pol={loss_p_det:.4f}+vw={loss_v_w_det:.4f}) | "
                    f"EMA={ema_l:.4f} acc={ema_a:.2f}% | step={global_step}",
                    refresh=True,
                )

            el = exp_loss.get()
            ea = exp_acc.get()
            if el is not None and ea is not None:
                _LOG.info("epoch %d train 结束 | EMA loss=%s acc%%=%s | step=%d", epoch, el, ea, global_step)

            model.eval()
            v_losses: list[float] = []
            v_accs: list[float] = []
            val_ema_loss = _ExpVal()
            val_ema_acc = _ExpVal()
            with torch.no_grad():
                if n_val_batches <= 0:
                    val_iter = iter(())
                    val_total = 0
                else:
                    ep_va_rng = random.Random(int(args.data_seed) + epoch * 1_000_003 + 900_017)
                    val_base = iter_collated_batches_from_finite_samples(
                        iter_epoch_joint_samples_shuffled(
                            val_files, policy_max_legal=pm, rng=ep_va_rng
                        ),
                        bs,
                        drop_last=False,
                        encode_workers=encode_workers,
                    )
                    val_iter = (
                        thread_prefetch_iterator(val_base, prefetch_batches)
                        if prefetch_batches > 0
                        else val_base
                    )
                    val_total = n_val_batches
                pbar_va = tqdm(
                    val_iter,
                    total=val_total,
                    desc=f"epoch {epoch} val",
                    unit="batch",
                    leave=True,
                    mininterval=0.0,
                    miniters=1,
                    dynamic_ncols=True,
                )
                val_bi = 0
                for x_np, m_np, yi_np, vs_np, hv_np in pbar_va:
                    val_bi += 1
                    x, mask, tgt, v_sign, has_v = _batch_tensors_to_device(
                        x_np, m_np, yi_np, vs_np, hv_np, device
                    )
                    logits_m, v_pred = model(x)
                    logits_masked = logits_m.masked_fill(~mask, -1e9)
                    loss_p = F.cross_entropy(logits_masked, tgt)
                    if has_v.any():
                        target_v = v_sign * float(model.value_scale)
                        loss_v = F.mse_loss(v_pred[has_v], target_v[has_v])
                    else:
                        loss_v = torch.zeros((), device=device)
                    loss = loss_p + vl_w * loss_v
                    pred = logits_masked.argmax(dim=-1)
                    v_b = float(loss.item())
                    pol_b = float(loss_p.detach().item())
                    vw_b = vl_w * float(loss_v.detach().item())
                    a_b = float((pred == tgt).float().mean().item() * 100.0)
                    v_losses.append(v_b)
                    v_accs.append(a_b)
                    val_ema_loss.update(v_b)
                    val_ema_acc.update(a_b)
                    vl_e = val_ema_loss.get()
                    va_e = val_ema_acc.get()
                    ema_vl = float(vl_e) if vl_e is not None else v_b
                    ema_va = float(va_e) if va_e is not None else a_b
                    pbar_va.set_postfix_str(
                        f"tot={v_b:.4f}(pol={pol_b:.4f}+vw={vw_b:.4f}) | "
                        f"EMA={ema_vl:.4f} acc={ema_va:.2f}%",
                        refresh=True,
                    )

            if v_losses:
                val_m = float(np.mean(v_losses))
                val_a = float(np.mean(v_accs))
                _LOG.info("epoch %d val | loss_mean=%.4f acc_mean=%.2f%%", epoch, val_m, val_a)
                if val_m < best_val:
                    best_val = val_m
                    out = save_dir / "best.pt"
                    _save_ckpt(out, model, opt, epoch=epoch, global_step=global_step, best_val=best_val)
                    _LOG.info("已写入 best -> %s", out.resolve())
            else:
                _LOG.info("epoch %d val | 无验证 batch，跳过 best.pt 更新", epoch)

            last = save_dir / "last.pt"
            _save_ckpt(last, model, opt, epoch=epoch, global_step=global_step, best_val=best_val)

        dt = time.perf_counter() - t0
        _LOG.info("监督训练结束 wall=%.1fs | best_val_loss=%.4f | last=%s", dt, best_val, (save_dir / "last.pt").resolve())

    except KeyboardInterrupt:
        _LOG.warning(
            "收到 KeyboardInterrupt（Ctrl+C）。指标请看 tqdm；本轮可能未完成 train/val。"
        )
        if training_started and global_step > 0:
            try:
                last = save_dir / "last.pt"
                _save_ckpt(
                    last,
                    model,
                    opt,
                    epoch=int(epoch),
                    global_step=int(global_step),
                    best_val=float(best_val),
                )
                _LOG.warning("已尽力写入 last.pt（epoch=%d step=%d），续训请自行确认是否重复本 epoch。", epoch, global_step)
            except Exception as e:
                _LOG.error("中断时保存 last.pt 失败: %s", e)
        sys.exit(130)


if __name__ == "__main__":
    main()
