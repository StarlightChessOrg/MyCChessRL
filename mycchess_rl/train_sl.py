"""XML ``.cbf`` 监督学习：联合合法槽 softmax + 价值 MSE。

指定棋谱目录后训练；``--checkpoint`` 可选：省略则在 ``--save-dir`` 下自动生成 ``bootstrap.pt`` 并开始训；若指向已有 ``.pt`` 则加载权重，若文件为本脚本保存的 SL 档（含 ``epoch``）则同时续优化器与进度。训练过程在 ``save_dir`` 下写 ``best.pt`` / ``last.pt``（YOLO 习惯）。
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mycchess_rl.model import JointPolicyValueNet, load_policy_value_for_play, torch_load_checkpoint
from mycchess_rl.sl_data import (
    build_joint_sl_train_val_loaders,
    discover_cbf_files,
    split_paths_train_test,
)

_LOG = logging.getLogger("mycchess_rl.train_sl")


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
    p.add_argument("--n-batch-train", type=int, default=200, help="每 epoch 训练批次数（Iterable 无限）")
    p.add_argument("--n-batch-val", type=int, default=30, help="每 epoch 验证批次数")
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
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader worker；Windows 或 xqwl 多进程建议 0")
    p.add_argument("--prefetch-factor", type=int, default=2)
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
    pin_mem = device.type == "cuda"
    pin_dev = str(device) if pin_mem else None
    train_loader, val_loader = build_joint_sl_train_val_loaders(
        train_files,
        val_files,
        int(args.batch_size),
        policy_max_legal=pm,
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        pin_memory=pin_mem,
        pin_memory_device=pin_dev,
    )

    train_it = iter(train_loader)
    val_it = iter(val_loader)
    t0 = time.perf_counter()

    for k in range(int(args.epochs)):
        epoch = epoch_begin + k
        exp_loss = _ExpVal()
        exp_acc = _ExpVal()
        model.train()
        for bi in range(int(args.n_batch_train)):
            try:
                x_np, m_np, yi_np, vs_np, hv_np = next(train_it)
            except StopIteration:
                train_it = iter(train_loader)
                x_np, m_np, yi_np, vs_np, hv_np = next(train_it)

            x = torch.from_numpy(np.ascontiguousarray(x_np)).to(device, non_blocking=pin_mem)
            mask = torch.from_numpy(m_np).to(device, non_blocking=pin_mem)
            tgt = torch.from_numpy(yi_np.astype(np.int64)).to(device, non_blocking=pin_mem)
            v_sign = torch.from_numpy(vs_np).to(device, non_blocking=pin_mem)
            has_v = torch.from_numpy(hv_np).to(device, non_blocking=pin_mem)

            opt.zero_grad(set_to_none=True)
            logits_m, v_pred = model(x)
            logits_masked = logits_m.masked_fill(~mask, -1e9)
            loss_p = F.cross_entropy(logits_masked, tgt)
            target_v = v_sign * float(model.value_scale)
            if has_v.any():
                loss_v = F.mse_loss(v_pred[has_v], target_v[has_v])
            else:
                loss_v = torch.zeros((), device=device)
            loss = loss_p + float(args.value_loss_weight) * loss_v
            loss.backward()
            opt.step()
            global_step += 1

            with torch.no_grad():
                pred = logits_masked.argmax(dim=-1)
                acc = float((pred == tgt).float().mean().item())
            exp_loss.update(float(loss.item()))
            exp_acc.update(acc * 100.0)

            if (bi + 1) % 50 == 0 or bi == 0:
                el = exp_loss.get()
                ea = exp_acc.get()
                _LOG.info(
                    "epoch %d train batch %d/%d | loss=%.4f (p=%.4f v=%.4f) | acc~%.2f%% | step=%d",
                    epoch,
                    bi + 1,
                    int(args.n_batch_train),
                    float(loss.item()),
                    float(loss_p.item()),
                    float(loss_v.item()) if has_v.any() else 0.0,
                    acc * 100.0,
                    global_step,
                )
                if el is not None and ea is not None:
                    _LOG.info("  EMA loss=%s acc%%=%s", el, ea)

        model.eval()
        v_losses: list[float] = []
        v_accs: list[float] = []
        with torch.no_grad():
            for _ in range(int(args.n_batch_val)):
                try:
                    x_np, m_np, yi_np, vs_np, hv_np = next(val_it)
                except StopIteration:
                    val_it = iter(val_loader)
                    x_np, m_np, yi_np, vs_np, hv_np = next(val_it)
                x = torch.from_numpy(np.ascontiguousarray(x_np)).to(device, non_blocking=pin_mem)
                mask = torch.from_numpy(m_np).to(device, non_blocking=pin_mem)
                tgt = torch.from_numpy(yi_np.astype(np.int64)).to(device, non_blocking=pin_mem)
                v_sign = torch.from_numpy(vs_np).to(device, non_blocking=pin_mem)
                has_v = torch.from_numpy(hv_np).to(device, non_blocking=pin_mem)
                logits_m, v_pred = model(x)
                logits_masked = logits_m.masked_fill(~mask, -1e9)
                loss_p = F.cross_entropy(logits_masked, tgt)
                if has_v.any():
                    target_v = v_sign * float(model.value_scale)
                    loss_v = F.mse_loss(v_pred[has_v], target_v[has_v])
                else:
                    loss_v = torch.zeros((), device=device)
                loss = loss_p + float(args.value_loss_weight) * loss_v
                pred = logits_masked.argmax(dim=-1)
                v_losses.append(float(loss.item()))
                v_accs.append(float((pred == tgt).float().mean().item() * 100.0))

        val_m = float(np.mean(v_losses)) if v_losses else 0.0
        val_a = float(np.mean(v_accs)) if v_accs else 0.0
        _LOG.info("epoch %d val | loss_mean=%.4f acc_mean=%.2f%%", epoch, val_m, val_a)
        if val_m < best_val:
            best_val = val_m
            out = save_dir / "best.pt"
            _save_ckpt(out, model, opt, epoch=epoch, global_step=global_step, best_val=best_val)
            _LOG.info("已写入 best -> %s", out.resolve())

        last = save_dir / "last.pt"
        _save_ckpt(last, model, opt, epoch=epoch, global_step=global_step, best_val=best_val)

    dt = time.perf_counter() - t0
    _LOG.info("监督训练结束 wall=%.1fs | best_val_loss=%.4f | last=%s", dt, best_val, (save_dir / "last.pt").resolve())


if __name__ == "__main__":
    main()
