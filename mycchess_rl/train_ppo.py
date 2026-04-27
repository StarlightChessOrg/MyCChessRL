"""并行环境 PPO 自对弈训练入口（示意实现）。"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mycchess_rl.encode_parallel import default_encode_workers
from mycchess_rl.model import JointPolicyValueNet, load_policy_value_for_play, torch_load_checkpoint
from mycchess_rl.policy_inference import (
    batched_encode_roots,
    batched_joint_logprob_on_moves,
    batched_value_expectation,
)
from mycchess_rl.ppo import PPOConfig, compute_gae, policy_value_loss_step
from mycchess_rl.vec_env import ParallelXiangqiVecEnv, collect_rollout_step, reset_finished

_LOG = logging.getLogger("mycchess_rl.train_ppo")


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


def _cuda_mem_mb(device: torch.device) -> str:
    """``memory_allocated``≈仍被张量占用的块；``memory_reserved``含 PyTorch 缓存池（峰值后常远高于 alloc，一般不是泄漏）。"""
    if device.type != "cuda" or not torch.cuda.is_available():
        return "n/a"
    try:
        idx = device.index if device.index is not None else 0
        alloc = torch.cuda.memory_allocated(idx) / (1024**2)
        reserv = torch.cuda.memory_reserved(idx) / (1024**2)
        peak = torch.cuda.max_memory_allocated(idx) / (1024**2)
        return f"torch_alloc={alloc:.0f}MB torch_reserved={reserv:.0f}MB peak_alloc={peak:.0f}MB"
    except Exception:
        return "n/a"


def main() -> None:
    p = argparse.ArgumentParser(description="MyCChessRL 并行 PPO 自对弈（规则：xqwl_core）")
    p.add_argument(
        "--n-env",
        type=int,
        default=384,
        help="并行环境数（增大可抬高 GPU 占用；显存不够时再调小）",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=192,
        help="每次更新前每个环境收集的步数（与 --n-env 相乘为每轮样本量上界）",
    )
    p.add_argument(
        "--updates",
        type=int,
        default=100_000,
        help="本轮要跑的 PPO 更新次数（续训时在已有全局 update 之后再跑这么多次；默认很大；Ctrl+C 仍会写 last）",
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="权重路径：默认仅加载 model，训练从 update=0 起；与 --resume 联用则另恢复 Adam 与全局轮次",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="续训：与 --checkpoint 指向训练存盘（含 optimizer 为佳），或省略 checkpoint 时用 --save-dir 下 last.pt；"
        "全局轮次从文件中 update+1 继续；本轮仍执行 --updates 次",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="每多少轮 update 打一次详细日志（1=每轮）",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="额外写入该路径（UTF-8）；与控制台相同内容",
    )
    p.add_argument(
        "--rollout-log-every",
        type=int,
        default=32,
        help="rollout 内每隔多少 timestep 打一条进度（0=关闭）",
    )
    p.add_argument(
        "--encode-backend",
        type=str,
        choices=("inline", "thread", "process"),
        default="inline",
        help="根平面编码：inline=主进程批 numpy+单次 H2D（rollout 默认，避免进程 IPC 拖 GPU）；"
        "thread/process 时用 --encode-workers",
    )
    p.add_argument(
        "--encode-workers",
        type=int,
        default=None,
        help="仅 encode-backend 为 thread/process 时生效；并行 worker 数，省略则 min(8, CPU核数)；1=退化为单 worker",
    )
    p.add_argument(
        "--random-action-prob",
        type=float,
        default=0.1,
        help="rollout 每步每环境：以该概率用「均匀随机合法着」替代策略采样（0=关闭）；利于乱战与非常规面，略增梯度方差",
    )
    p.add_argument(
        "--ppo-mini-batch",
        type=int,
        default=4096,
        help="PPO 反向时 GPU mini-batch 大小；整轮样本可达数万，过小则慢、过大易 OOM（A100 40GB 建议 2048~8192）",
    )
    p.add_argument(
        "--rollout-pipeline-groups",
        type=int,
        default=2,
        help="rollout 采样/价值前向时 CPU 编码分组数；>=2 时两半局面并行编码（主线程+守护线程）再拼批一次 trunk，"
        "叠合准备空档；1=关闭（与旧行为一致）",
    )
    p.add_argument(
        "--cuda-empty-cache-each-update",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="每轮 PPO 优化结束后释放本轮 GPU 大张量并 torch.cuda.empty_cache()；"
        "nvidia-smi 里「常驻」多为 reserved 缓存，非泄漏。用 --no-cuda-empty-cache-each-update 可关闭（略省开销）",
    )
    p.add_argument(
        "--save-dir",
        type=Path,
        default=Path("runs"),
        help="checkpoint 与最终权重保存目录（会创建）",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="每隔多少轮额外写入 save-dir/weights/upd_*.pt（0=不保留按步快照；best.pt/last.pt 仍按 YOLO 习惯更新）",
    )
    p.add_argument(
        "--reward-shaping-king",
        type=float,
        default=0.015,
        help="非终局：鼓励落点靠近对方将/帅（prox∈[0,1]）；若对方被应将再加 (0.4+0.6·prox)×同一系数。0=关闭",
    )
    p.add_argument(
        "--reward-shaping-capture",
        type=float,
        default=0.025,
        help="吃子奖励基量（兵卒=1×），车马炮等按子种加权；0=关闭",
    )
    args = p.parse_args()

    _setup_logging(args.log_file)
    save_dir: Path = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    save_every = max(0, int(args.save_every))

    enc_w = default_encode_workers() if args.encode_workers is None else int(args.encode_workers)
    enc_be = str(args.encode_backend).strip().lower()
    rp_groups = max(1, int(args.rollout_pipeline_groups))

    device = torch.device("cpu")
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")

    t_train0 = time.perf_counter()

    ckpt_path_model: Path | None = None
    if args.checkpoint is not None:
        ckpt_path_model = Path(args.checkpoint)
    elif args.resume:
        ckpt_path_model = save_dir / "last.pt"

    if ckpt_path_model is not None:
        if not ckpt_path_model.is_file():
            _LOG.error("找不到 checkpoint: %s", ckpt_path_model.resolve())
            raise SystemExit(2)
        model, flist = load_policy_value_for_play(ckpt_path_model, device)
        model.eval()
        _LOG.info("从 checkpoint 加载 model: %s", ckpt_path_model.resolve())
    else:
        if args.resume:
            _LOG.error("--resume 需要 --checkpoint，或先有 %s", (save_dir / "last.pt").resolve())
            raise SystemExit(2)
        model = JointPolicyValueNet().to(device)
        from mycchess_rl.chess import FEATURE_LIST

        flist = {"red": list(FEATURE_LIST["red"]), "black": list(FEATURE_LIST["black"])}
        model.eval()
        _LOG.info("随机初始化策略网络")

    n_params = sum(p.numel() for p in model.parameters())
    _LOG.info(
        "设备=%s | n_env=%d steps=%d updates=%d lr=%g | ResNet in_ch=%d 参数量=%s",
        device,
        args.n_env,
        args.steps,
        args.updates,
        args.lr,
        model.in_channels,
        f"{n_params:,}",
    )
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        name = torch.cuda.get_device_name(idx)
        _LOG.info("CUDA 设备: [%d] %s", idx, name)
        torch.cuda.reset_peak_memory_stats(idx)
    _LOG.info(
        "特征编码 backend=%s encode_workers=%d（仅 thread/process）| rollout_pipeline_groups=%d | PPO mini-batch=%d",
        enc_be,
        enc_w,
        rp_groups,
        int(args.ppo_mini_batch),
    )
    _LOG.info(
        "checkpoint 目录=%s | save_every=%d（0=不写 weights/upd_*.pt；仍每轮更新 last.pt，更优时写 best.pt）",
        save_dir.resolve(),
        save_every,
    )
    rs_king = float(args.reward_shaping_king)
    rs_cap = float(args.reward_shaping_capture)
    _LOG.info("奖励塑形 king=%g capture=%g（全 0=仅终局将死/和棋）", rs_king, rs_cap)

    cfg = PPOConfig(lr=args.lr)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_loss_total = float("inf")
    start_global = 0
    if args.resume:
        assert ckpt_path_model is not None
        raw = torch_load_checkpoint(ckpt_path_model, device)
        if "optimizer" in raw:
            try:
                opt.load_state_dict(raw["optimizer"])
                _LOG.info("已恢复 Adam 状态（含一阶/二阶矩缓冲）")
            except Exception as e:
                _LOG.warning("优化器状态与当前模型或设备不兼容，已忽略（按新 Adam 累积）: %s", e)
        else:
            _LOG.warning("checkpoint 中无 optimizer 字段，续训仅继承权重，Adam 从新开始")
        last_u = int(raw.get("update", -1))
        start_global = last_u + 1
        if start_global < 0:
            start_global = 0
        _LOG.info("续训：下一档全局 update=%d（文件中最后一档已完成=%d）", start_global, last_u)
        rb = raw.get("best_loss_total")
        if type(rb) in (int, float):
            best_loss_total = float(rb)
            _LOG.info("已恢复 best_loss_total=%.5f（用于与 best.pt 比较）", best_loss_total)

    for g in opt.param_groups:
        g["lr"] = float(args.lr)

    vec = ParallelXiangqiVecEnv(args.n_env)
    vec.reset_all()

    T = args.steps
    N = args.n_env
    gen = torch.Generator(device=device)
    explore_rng = np.random.default_rng()
    r_ap = float(args.random_action_prob)
    r_ap = min(1.0, max(0.0, r_ap))
    _LOG.info("rollout 随机合法着手探索概率 random_action_prob=%.4f", r_ap)
    log_every = max(1, int(args.log_every))
    roll_log = int(args.rollout_log_every)

    last_finished_update: int | None = None
    interrupted = False

    def _training_checkpoint_dict(finished_upd: int) -> dict:
        return {
            "model": model.state_dict(),
            "in_channels": model.in_channels,
            "num_res_layers": model.num_res_layers,
            "filters": model.filters,
            "policy_max_legal": model.policy_max_legal,
            "value_scale": model.value_scale,
            "update": int(finished_upd),
            "optimizer": opt.state_dict(),
            "kind": "ppo",
            "best_loss_total": float(best_loss_total),
        }

    def _save_checkpoint_file(path: Path, finished_upd: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_training_checkpoint_dict(finished_upd), path)

    try:
        for k in range(args.updates):
            upd = start_global + k
            t_upd0 = time.perf_counter()
            _LOG.info(
                "[ppo] update %d 开始 | 本轮 %d/%d | rollout 共 %d 步 × %d 环境",
                upd,
                k + 1,
                args.updates,
                T,
                N,
            )
            obs_buf: list[list[object]] = [[None] * N for _ in range(T)]
            act_buf = np.empty((T, N), dtype=object)
            rew_buf = np.zeros((T, N), dtype=np.float32)
            done_buf = np.zeros((T, N), dtype=np.bool_)
            val_buf = np.zeros((T, N), dtype=np.float32)

            t_roll0 = time.perf_counter()
            with torch.no_grad():
                v_cur = batched_value_expectation(
                    [s.game for s in vec.slots],
                    model,
                    device,
                    flist,
                    encode_workers=enc_w,
                    encode_backend=enc_be,
                    rollout_pipeline_groups=rp_groups,
                )
                v_cur = v_cur.detach().float().cpu().numpy()

            n_done_rollout = 0
            mean_abs_rew = 0.0

            for t in range(T):
                gps_pre = [s.game for s in vec.slots]
                for i in range(N):
                    obs_buf[t][i] = gps_pre[i]
                val_buf[t] = v_cur

                rew, done, _, moves = collect_rollout_step(
                    vec,
                    model,
                    device,
                    flist,
                    policy_temperature=1.0,
                    generator=gen,
                    encode_workers=enc_w,
                    encode_backend=enc_be,
                    rollout_pipeline_groups=rp_groups,
                    reward_shaping_capture=rs_cap,
                    reward_shaping_king=rs_king,
                    random_action_prob=r_ap,
                    explore_rng=explore_rng,
                )
                act_buf[t, :] = moves
                rew_buf[t] = rew
                done_buf[t] = done
                n_done_rollout += int(done.sum())
                mean_abs_rew += float(np.abs(rew).sum())

                with torch.no_grad():
                    v_cur = batched_value_expectation(
                        [s.game for s in vec.slots],
                        model,
                        device,
                        flist,
                        encode_workers=enc_w,
                        encode_backend=enc_be,
                        rollout_pipeline_groups=rp_groups,
                    )
                    v_cur = v_cur.detach().float().cpu().numpy()
                reset_finished(vec, done)

                if roll_log > 0 and (t + 1) % roll_log == 0:
                    _LOG.info(
                        "[ppo] update %d rollout 进度 %d/%d (%.0f%%) elapsed=%.1fs",
                        upd,
                        t + 1,
                        T,
                        100.0 * (t + 1) / T,
                        time.perf_counter() - t_roll0,
                    )

            t_roll1 = time.perf_counter()
            rollout_s = t_roll1 - t_roll0

            with torch.no_grad():
                last_v = batched_value_expectation(
                    [s.game for s in vec.slots],
                    model,
                    device,
                    flist,
                    encode_workers=enc_w,
                    encode_backend=enc_be,
                    rollout_pipeline_groups=rp_groups,
                )
                last_v = last_v.detach().float().cpu().numpy()
            adv, ret = compute_gae(rew_buf, val_buf, done_buf, last_v, gamma=cfg.gamma, lam=cfg.gae_lambda)
            adv_mean_b, adv_std_b = float(adv.mean()), float(adv.std())
            ret_mean_b, ret_std_b = float(ret.mean()), float(ret.std())
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            obs_list: list = []
            mv_list: list[str] = []
            old_v_list: list[float] = []
            adv_list: list[float] = []
            ret_list: list[float] = []

            for t in range(T):
                for i in range(N):
                    g = obs_buf[t][i]
                    mv = act_buf[t, i]
                    if not isinstance(mv, str) or len(mv) < 5:
                        continue
                    obs_list.append(g)
                    mv_list.append(mv)
                    old_v_list.append(float(val_buf[t, i]))
                    adv_list.append(float(adv[t, i]))
                    ret_list.append(float(ret[t, i]))

            slots_total = T * N
            n_skipped = slots_total - len(obs_list)
            if not obs_list:
                _LOG.warning("update=%d 无有效样本（全部被跳过），跳过优化", upd)
                continue

            n_opt_samples = len(obs_list)

            t_opt0 = time.perf_counter()
            _LOG.info(
                "[ppo] update %d 优化阶段 | 有效样本=%d / slots=%d | 编码+batch trunk...",
                upd,
                len(obs_list),
                slots_total,
            )
            xb = batched_encode_roots(
                obs_list,
                flist,
                device,
                encode_workers=enc_w,
                encode_backend=enc_be,
            )
            with torch.no_grad():
                model.eval()
                feat_roll = model._trunk_flat(xb)
                _LOG.info(
                    "[ppo] update %d 计算 old_logp（整批 head，样本数=%d）...",
                    upd,
                    feat_roll.shape[0],
                )
                old_lp, legal_mask_b, action_idx_b = batched_joint_logprob_on_moves(
                    obs_list,
                    mv_list,
                    feat_roll,
                    model,
                    device,
                    policy_temperature=1.0,
                )
                del feat_roll
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            _LOG.info(
                "[ppo] update %d old_logp 完成 elapsed=%.2fs，开始反向更新",
                upd,
                time.perf_counter() - t_opt0,
            )

            adv_b = torch.clamp(torch.tensor(adv_list, device=device), -5.0, 5.0)
            ret_b = torch.clamp(torch.tensor(ret_list, device=device), -10.0, 10.0)
            old_v = torch.tensor(old_v_list, device=device)

            loss, m = policy_value_loss_step(
                model,
                opt,
                xb,
                legal_mask_b,
                action_idx_b,
                old_lp,
                adv_b,
                ret_b,
                old_v,
                cfg,
                mini_batch_size=int(args.ppo_mini_batch),
                policy_temperature=1.0,
            )
            del xb, old_lp, adv_b, ret_b, old_v, legal_mask_b, action_idx_b
            obs_list.clear()
            mv_list.clear()
            adv_list.clear()
            ret_list.clear()
            old_v_list.clear()
            if device.type == "cuda" and bool(getattr(args, "cuda_empty_cache_each_update", True)):
                torch.cuda.empty_cache()

            t_opt1 = time.perf_counter()
            optimize_s = t_opt1 - t_opt0
            upd_s = time.perf_counter() - t_upd0

            if upd % log_every == 0:
                mem = _cuda_mem_mb(device)
                _LOG.info(
                    "[ppo] update %d | 本轮 %d/%d | wall_rollout=%.2fs wall_opt=%.2fs wall_total=%.2fs | "
                    "slots=%d samples=%d skipped=%d done_flags=%d mean|rew|_per_slot=%.4f",
                    upd,
                    k + 1,
                    args.updates,
                    rollout_s,
                    optimize_s,
                    upd_s,
                    slots_total,
                    n_opt_samples,
                    n_skipped,
                    n_done_rollout,
                    mean_abs_rew / max(slots_total, 1),
                )
                _LOG.info(
                    "[ppo] gae(before_norm) mean_adv=%.4f std_adv=%.4f mean_ret=%.4f std_ret=%.4f | "
                    "clip=%.3f gamma=%.4f gae_lambda=%.4f vf_coef=%.3f ent_coef=%.4f",
                    adv_mean_b,
                    adv_std_b,
                    ret_mean_b,
                    ret_std_b,
                    cfg.clip_eps,
                    cfg.gamma,
                    cfg.gae_lambda,
                    cfg.vf_coef,
                    cfg.ent_coef,
                )
                _LOG.info(
                    "[ppo] loss total=%.5f policy=%.5f value=%.5f vf_w=%.5f | "
                    "entropy_joint=%.4f | ratio mean=%.4f std=%.4f clip_frac=%.4f approx_kl=%.5f",
                    m["loss_total"],
                    m["loss_policy"],
                    m["loss_value"],
                    m["loss_vf_weighted"],
                    m["entropy_sum"],
                    m["ratio_mean"],
                    m["ratio_std"],
                    m["clip_frac"],
                    m["approx_kl"],
                )
                _LOG.info(
                    "[ppo] value batch: v_pred_mean=%.4f old_v_mean=%.4f | grad_norm=%.4f | %s",
                    m["v_pred_mean"],
                    m["old_v_mean"],
                    m["grad_norm"],
                    mem,
                )

            last_finished_update = upd

            last_p = save_dir / "last.pt"
            _save_checkpoint_file(last_p, upd)
            lt = float(m["loss_total"])
            if lt < best_loss_total:
                best_loss_total = lt
                best_p = save_dir / "best.pt"
                _save_checkpoint_file(best_p, upd)
                _LOG.info("新最佳 loss_total=%.5f -> %s", lt, best_p.resolve())

            if save_every > 0 and (upd + 1) % save_every == 0:
                wdir = save_dir / "weights"
                wdir.mkdir(parents=True, exist_ok=True)
                ckpt = wdir / f"upd_{upd:06d}.pt"
                _save_checkpoint_file(ckpt, upd)
                _LOG.info("已保存权重快照 update=%d -> %s", upd, ckpt)

    except KeyboardInterrupt:
        interrupted = True
        _LOG.warning("收到 Ctrl+C，中止训练（保留上一轮已完成的权重）")
    finally:
        total_s = time.perf_counter() - t_train0
        if last_finished_update is not None:
            out = save_dir / "last.pt"
            _save_checkpoint_file(out, last_finished_update)
            if interrupted:
                _LOG.info(
                    "已保存 last.pt wall_total=%.1fs | %s (update=%d)",
                    total_s,
                    out.resolve(),
                    last_finished_update,
                )
            else:
                _LOG.info("训练结束 wall_total=%.1fs | 已保存 %s", total_s, out.resolve())
        else:
            _LOG.warning("尚未完成任一整轮 PPO 更新，未写入 last.pt | wall_total=%.1fs", total_s)


if __name__ == "__main__":
    main()
