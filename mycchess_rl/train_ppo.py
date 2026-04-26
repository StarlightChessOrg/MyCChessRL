"""并行环境 PPO 自对弈训练入口（示意实现）。"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mycchess_rl.model import SuccessorPolicy, load_successor_policy_for_play
from mycchess_rl.policy_inference import (
    batched_encode_roots,
    batched_two_stage_logprob_on_moves,
    batched_value_expectation,
)
from mycchess_rl.ppo import PPOConfig, compute_gae, iccs_to_src_dst_onehot, policy_value_loss_step
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
    if device.type != "cuda" or not torch.cuda.is_available():
        return "n/a"
    try:
        idx = device.index if device.index is not None else 0
        alloc = torch.cuda.memory_allocated(idx) / (1024**2)
        reserv = torch.cuda.memory_reserved(idx) / (1024**2)
        return f"alloc={alloc:.0f}MB reserved={reserv:.0f}MB"
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
        default=800,
        help="PPO 更新轮数（单机强 GPU 可拉长总训练）",
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--checkpoint", type=Path, default=None, help="从已有权重微调；省略则随机初始化")
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
    args = p.parse_args()

    _setup_logging(args.log_file)

    device = torch.device("cpu")
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")

    t_train0 = time.perf_counter()

    if args.checkpoint is not None:
        model, flist = load_successor_policy_for_play(args.checkpoint, device)
        model.train()
        _LOG.info("从 checkpoint 加载: %s", args.checkpoint)
    else:
        model = SuccessorPolicy().to(device)
        from mycchess_rl.chess import FEATURE_LIST

        flist = {"red": list(FEATURE_LIST["red"]), "black": list(FEATURE_LIST["black"])}
        model.train()
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

    cfg = PPOConfig(lr=args.lr)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    vec = ParallelXiangqiVecEnv(args.n_env)
    vec.reset_all()

    T = args.steps
    N = args.n_env
    gen = torch.Generator(device=device)
    log_every = max(1, int(args.log_every))
    roll_log = int(args.rollout_log_every)

    for upd in range(args.updates):
        t_upd0 = time.perf_counter()
        _LOG.info(
            "[ppo] update %d/%d 开始 | rollout 共 %d 步 × %d 环境",
            upd,
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
            v_cur = batched_value_expectation([s.game for s in vec.slots], model, device, flist)
            v_cur = v_cur.detach().float().cpu().numpy()

        n_done_rollout = 0
        mean_abs_rew = 0.0

        for t in range(T):
            gps_pre = [s.game for s in vec.slots]
            for i in range(N):
                obs_buf[t][i] = gps_pre[i]
            val_buf[t] = v_cur

            rew, done, _, moves = collect_rollout_step(
                vec, model, device, flist, policy_temperature=1.0, generator=gen
            )
            act_buf[t, :] = moves
            rew_buf[t] = rew
            done_buf[t] = done
            n_done_rollout += int(done.sum())
            mean_abs_rew += float(np.abs(rew).sum())

            with torch.no_grad():
                v_cur = batched_value_expectation([s.game for s in vec.slots], model, device, flist)
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
            last_v = batched_value_expectation([s.game for s in vec.slots], model, device, flist)
            last_v = last_v.detach().float().cpu().numpy()
        adv, ret = compute_gae(rew_buf, val_buf, done_buf, last_v, gamma=cfg.gamma, lam=cfg.gae_lambda)
        adv_mean_b, adv_std_b = float(adv.mean()), float(adv.std())
        ret_mean_b, ret_std_b = float(ret.mean()), float(ret.std())
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_list: list = []
        mv_list: list[str] = []
        src_list: list[torch.Tensor] = []
        dst_list: list[torch.Tensor] = []
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
                oh_s, oh_d = iccs_to_src_dst_onehot(mv, device, torch.float32)
                src_list.append(oh_s)
                dst_list.append(oh_d)
                old_v_list.append(float(val_buf[t, i]))
                adv_list.append(float(adv[t, i]))
                ret_list.append(float(ret[t, i]))

        slots_total = T * N
        n_skipped = slots_total - len(obs_list)
        if not obs_list:
            _LOG.warning("update=%d 无有效样本（全部被跳过），跳过优化", upd)
            continue

        t_opt0 = time.perf_counter()
        _LOG.info(
            "[ppo] update %d 优化阶段 | 有效样本=%d / slots=%d | 编码+batch trunk...",
            upd,
            len(obs_list),
            slots_total,
        )
        xb = batched_encode_roots(obs_list, flist, device)
        with torch.no_grad():
            model.eval()
            feat_roll = model._trunk_flat(xb)
            _LOG.info(
                "[ppo] update %d 计算 old_logp（整批 head，样本数=%d）...",
                upd,
                feat_roll.shape[0],
            )
            old_lp = batched_two_stage_logprob_on_moves(
                obs_list,
                mv_list,
                feat_roll,
                model,
                device,
                policy_temperature=1.0,
            )
        _LOG.info(
            "[ppo] update %d old_logp 完成 elapsed=%.2fs，开始反向更新",
            upd,
            time.perf_counter() - t_opt0,
        )

        src_b = torch.stack(src_list, dim=0)
        dst_b = torch.stack(dst_list, dim=0)
        adv_b = torch.clamp(torch.tensor(adv_list, device=device), -5.0, 5.0)
        ret_b = torch.clamp(torch.tensor(ret_list, device=device), -10.0, 10.0)
        old_v = torch.tensor(old_v_list, device=device)

        loss, m = policy_value_loss_step(model, opt, xb, src_b, dst_b, old_lp, adv_b, ret_b, old_v, cfg)
        t_opt1 = time.perf_counter()
        optimize_s = t_opt1 - t_opt0
        upd_s = time.perf_counter() - t_upd0

        if upd % log_every == 0:
            mem = _cuda_mem_mb(device)
            _LOG.info(
                "[ppo] update %d/%d | wall_rollout=%.2fs wall_opt=%.2fs wall_total=%.2fs | "
                "slots=%d samples=%d skipped=%d done_flags=%d mean|rew|_per_slot=%.4f",
                upd,
                args.updates,
                rollout_s,
                optimize_s,
                upd_s,
                slots_total,
                len(obs_list),
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
                "entropy sum=%.4f (src=%.4f dst=%.4f) | ratio mean=%.4f std=%.4f clip_frac=%.4f approx_kl=%.5f",
                m["loss_total"],
                m["loss_policy"],
                m["loss_value"],
                m["loss_vf_weighted"],
                m["entropy_sum"],
                m["entropy_src"],
                m["entropy_dst"],
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

    out = Path("mycchess_ppo_last.pt")
    torch.save({"model": model.state_dict(), "in_channels": model.in_channels}, out)
    total_s = time.perf_counter() - t_train0
    _LOG.info("训练结束 wall_total=%.1fs | 已保存 %s", total_s, out)


if __name__ == "__main__":
    main()
