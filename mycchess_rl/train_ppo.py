"""并行环境 PPO 自对弈训练入口（示意实现）。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from mycchess_rl.model import SuccessorPolicy, load_successor_policy_for_play
from mycchess_rl.policy_inference import batched_encode_roots, batched_value_expectation, two_stage_logprob_on_move
from mycchess_rl.ppo import PPOConfig, compute_gae, iccs_to_src_dst_onehot, policy_value_loss_step
from mycchess_rl.vec_env import ParallelXiangqiVecEnv, collect_rollout_step, reset_finished


def main() -> None:
    p = argparse.ArgumentParser(description="MyCChessRL 并行 PPO 自对弈（规则：xqwl_core）")
    p.add_argument(
        "--n-env",
        type=int,
        default=192,
        help="并行环境数（默认可喂满 A100 40GB 类 GPU 的批推理）",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=128,
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
    args = p.parse_args()

    device = torch.device("cpu")
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")

    if args.checkpoint is not None:
        model, flist = load_successor_policy_for_play(args.checkpoint, device)
        model.train()
    else:
        model = SuccessorPolicy().to(device)
        from mycchess_rl.chess import FEATURE_LIST

        flist = {"red": list(FEATURE_LIST["red"]), "black": list(FEATURE_LIST["black"])}
        model.train()

    cfg = PPOConfig(lr=args.lr)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    vec = ParallelXiangqiVecEnv(args.n_env)
    vec.reset_all()

    T = args.steps
    N = args.n_env
    gen = torch.Generator(device=device)

    for upd in range(args.updates):
        obs_buf: list[list[object]] = [[None] * N for _ in range(T)]
        act_buf = np.empty((T, N), dtype=object)
        rew_buf = np.zeros((T, N), dtype=np.float32)
        done_buf = np.zeros((T, N), dtype=np.bool_)
        val_buf = np.zeros((T, N), dtype=np.float32)

        with torch.no_grad():
            v_cur = batched_value_expectation([s.game for s in vec.slots], model, device, flist).cpu().numpy()

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

            with torch.no_grad():
                v_cur = batched_value_expectation([s.game for s in vec.slots], model, device, flist).cpu().numpy()
            reset_finished(vec, done)

        with torch.no_grad():
            last_v = batched_value_expectation([s.game for s in vec.slots], model, device, flist).cpu().numpy()
        adv, ret = compute_gae(rew_buf, val_buf, done_buf, last_v, gamma=cfg.gamma, lam=cfg.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_list: list = []
        src_list: list[torch.Tensor] = []
        dst_list: list[torch.Tensor] = []
        old_logp_list: list[float] = []
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
                oh_s, oh_d = iccs_to_src_dst_onehot(mv, device, torch.float32)
                src_list.append(oh_s)
                dst_list.append(oh_d)
                with torch.no_grad():
                    lp = float(
                        two_stage_logprob_on_move(g, model, device, flist, mv, policy_temperature=1.0).cpu()
                    )
                old_logp_list.append(lp)
                old_v_list.append(float(val_buf[t, i]))
                adv_list.append(float(adv[t, i]))
                ret_list.append(float(ret[t, i]))

        if not obs_list:
            continue

        xb = batched_encode_roots(obs_list, flist, device)
        src_b = torch.stack(src_list, dim=0)
        dst_b = torch.stack(dst_list, dim=0)
        old_lp = torch.tensor(old_logp_list, device=device)
        old_v = torch.tensor(old_v_list, device=device)
        adv_b = torch.clamp(torch.tensor(adv_list, device=device), -5.0, 5.0)
        ret_b = torch.clamp(torch.tensor(ret_list, device=device), -10.0, 10.0)

        loss = policy_value_loss_step(model, opt, xb, src_b, dst_b, old_lp, adv_b, ret_b, old_v, cfg)
        if upd % 10 == 0:
            print(f"update {upd} loss={loss:.4f} mean_ret={float(ret.mean()):.4f}", flush=True)

    out = Path("mycchess_ppo_last.pt")
    torch.save({"model": model.state_dict(), "in_channels": model.in_channels}, out)
    print(f"已保存 {out}", flush=True)


if __name__ == "__main__":
    main()
