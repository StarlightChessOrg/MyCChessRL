"""PPO 辅助：GAE 与联合合法着法分布 + 标量价值的更新。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from mycchess_rl.model import PolicyValueBackbone, policy_temperature_scalar


@dataclass
class PPOConfig:
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    lr: float = 3e-4
    max_grad_norm: float = 1.0


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    *,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    t_max, n = rewards.shape
    adv = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros(n, dtype=np.float32)
    next_v = last_values.copy()
    for t in reversed(range(t_max)):
        mask = 1.0 - dones[t].astype(np.float32)
        delta = rewards[t] + gamma * next_v * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        adv[t] = last_gae
        next_v = values[t]
    ret = adv + values
    return adv, ret


def _ppo_forward_loss_joint(
    model: PolicyValueBackbone,
    obs: torch.Tensor,
    legal_mask: torch.Tensor,
    action_idx: torch.Tensor,
    old_logp: torch.Tensor,
    adv: torch.Tensor,
    ret_value: torch.Tensor,
    old_v: torch.Tensor,
    cfg: PPOConfig,
    *,
    policy_temperature: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    T = policy_temperature_scalar(policy_temperature)
    logits_m, v_pred = model(obs)
    scaled = logits_m / T
    scaled = scaled.masked_fill(~legal_mask, -1e9)
    log_p_all = F.log_softmax(scaled, dim=1)
    logp = log_p_all.gather(1, action_idx.unsqueeze(1)).squeeze(1)
    p = torch.softmax(scaled, dim=1)
    ent_row = -(p * log_p_all).sum(dim=1)
    ent = ent_row.mean()

    logp = torch.clamp(logp, -80.0, 0.0)
    old_logp_c = torch.clamp(old_logp, -80.0, 0.0)
    ratio = torch.exp(torch.clamp(logp - old_logp_c, -5.0, 5.0))
    ratio = torch.clamp(ratio, 0.0, 32.0)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
    pol_loss = -torch.min(surr1, surr2).mean()
    v_loss = F.smooth_l1_loss(v_pred, ret_value, beta=0.5)
    loss = pol_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent

    clip_lo, clip_hi = 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps
    clip_frac = ((ratio < clip_lo) | (ratio > clip_hi)).float().mean()
    approx_kl = (old_logp_c - logp).mean()

    with torch.no_grad():
        dbg = {
            "pol": float(pol_loss.detach().cpu()),
            "v": float(v_loss.detach().cpu()),
            "ent": float(ent.detach().cpu()),
            "ratio_mean": float(ratio.mean().cpu()),
            "ratio_sq_mean": float((ratio * ratio).mean().cpu()),
            "clip_frac": float(clip_frac.detach().cpu()),
            "approx_kl": float(approx_kl.detach().cpu()),
            "v_pred_mean": float(v_pred.mean().cpu()),
            "old_v_mean": float(old_v.mean().cpu()),
        }
    return loss, dbg


def policy_value_loss_step(
    model: PolicyValueBackbone,
    opt: torch.optim.Optimizer,
    obs: torch.Tensor,
    legal_mask: torch.Tensor,
    action_idx: torch.Tensor,
    old_logp: torch.Tensor,
    adv: torch.Tensor,
    ret_value: torch.Tensor,
    old_v: torch.Tensor,
    cfg: PPOConfig,
    *,
    mini_batch_size: int | None = None,
    policy_temperature: float = 1.0,
) -> tuple[float, dict[str, float]]:
    """
    PPO 更新。``mini_batch_size`` 为 None 或 ≥ N 时整批一次前向；
    否则按小批梯度累积。

    使用 ``model.eval()``：ResNet 中含 BatchNorm 时，必须与 rollout / ``old_logp`` 的 eval 前向一致。
    """
    model.eval()
    n = int(obs.shape[0])
    if n == 0:
        return 0.0, {"batch": 0.0}

    mbs = n if mini_batch_size is None else max(1, int(mini_batch_size))
    if mbs >= n:
        loss, dbg = _ppo_forward_loss_joint(
            model,
            obs,
            legal_mask,
            action_idx,
            old_logp,
            adv,
            ret_value,
            old_v,
            cfg,
            policy_temperature=policy_temperature,
        )
        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        opt.step()
        rm, r2 = dbg["ratio_mean"], dbg["ratio_sq_mean"]
        rstd = (max(0.0, r2 - rm * rm)) ** 0.5
        metrics: dict[str, float] = {
            "loss_total": float(loss.detach().cpu()),
            "loss_policy": dbg["pol"],
            "loss_value": dbg["v"],
            "loss_vf_weighted": cfg.vf_coef * dbg["v"],
            "entropy_sum": dbg["ent"],
            "entropy_src": dbg["ent"],
            "entropy_dst": 0.0,
            "ratio_mean": rm,
            "ratio_std": rstd,
            "clip_frac": dbg["clip_frac"],
            "approx_kl": dbg["approx_kl"],
            "grad_norm": float(
                grad_norm.detach().cpu() if isinstance(grad_norm, torch.Tensor) else grad_norm
            ),
            "batch": float(n),
            "v_pred_mean": dbg["v_pred_mean"],
            "old_v_mean": dbg["old_v_mean"],
        }
        return metrics["loss_total"], metrics

    opt.zero_grad()
    acc_pol = acc_v = acc_ent = 0.0
    acc_rm = acc_r2 = 0.0
    acc_clip = acc_kl = 0.0
    acc_vpred = acc_oldv = 0.0
    acc_loss_log = 0.0

    for start in range(0, n, mbs):
        end = min(start + mbs, n)
        w = (end - start) / n
        loss_mb, dbg = _ppo_forward_loss_joint(
            model,
            obs[start:end],
            legal_mask[start:end],
            action_idx[start:end],
            old_logp[start:end],
            adv[start:end],
            ret_value[start:end],
            old_v[start:end],
            cfg,
            policy_temperature=policy_temperature,
        )
        (loss_mb * w).backward()
        acc_loss_log += float(loss_mb.detach().cpu()) * w
        acc_pol += dbg["pol"] * w
        acc_v += dbg["v"] * w
        acc_ent += dbg["ent"] * w
        acc_rm += dbg["ratio_mean"] * w
        acc_r2 += dbg["ratio_sq_mean"] * w
        acc_clip += dbg["clip_frac"] * w
        acc_kl += dbg["approx_kl"] * w
        acc_vpred += dbg["v_pred_mean"] * w
        acc_oldv += dbg["old_v_mean"] * w

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
    opt.step()
    rstd = (max(0.0, acc_r2 - acc_rm * acc_rm)) ** 0.5
    metrics = {
        "loss_total": acc_loss_log,
        "loss_policy": acc_pol,
        "loss_value": acc_v,
        "loss_vf_weighted": cfg.vf_coef * acc_v,
        "entropy_sum": acc_ent,
        "entropy_src": acc_ent,
        "entropy_dst": 0.0,
        "ratio_mean": acc_rm,
        "ratio_std": rstd,
        "clip_frac": acc_clip,
        "approx_kl": acc_kl,
        "grad_norm": float(
            grad_norm.detach().cpu() if isinstance(grad_norm, torch.Tensor) else grad_norm
        ),
        "batch": float(n),
        "v_pred_mean": acc_vpred,
        "old_v_mean": acc_oldv,
    }
    return acc_loss_log, metrics
