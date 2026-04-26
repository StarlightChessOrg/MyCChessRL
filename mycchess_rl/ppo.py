"""PPO 辅助：GAE 与联合策略-价值更新（两阶段离散动作）。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from mycchess_rl.model import SuccessorPolicy


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


def value_expectation_from_logits(logits_v: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits_v.float(), dim=-1)
    w = torch.tensor([3.0, 1.0, -3.0], device=logits_v.device, dtype=logits_v.dtype)
    return (p * w).sum(dim=-1)


def _ppo_forward_loss(
    model: SuccessorPolicy,
    obs: torch.Tensor,
    src_oh: torch.Tensor,
    dst_oh: torch.Tensor,
    old_logp: torch.Tensor,
    adv: torch.Tensor,
    ret_value: torch.Tensor,
    old_v: torch.Tensor,
    cfg: PPOConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """一次前向：返回可反传的 ``loss`` 及 detached 日志字典（对该 mini-batch 的 mean）。"""
    logits_s, logits_d, logits_v = model(obs, src_oh)
    logp_s = (F.log_softmax(logits_s, dim=1) * src_oh).sum(dim=1)
    logp_d = (F.log_softmax(logits_d, dim=1) * dst_oh).sum(dim=1)
    logp = torch.clamp(logp_s + logp_d, -80.0, 0.0)
    old_logp_c = torch.clamp(old_logp, -80.0, 0.0)
    ratio = torch.exp(torch.clamp(logp - old_logp_c, -5.0, 5.0))
    ratio = torch.clamp(ratio, 0.0, 32.0)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
    pol_loss = -torch.min(surr1, surr2).mean()
    v_pred = value_expectation_from_logits(logits_v)
    v_loss = F.smooth_l1_loss(v_pred, ret_value, beta=0.5)
    ent_s = (-(F.softmax(logits_s, 1) * F.log_softmax(logits_s, 1)).sum(1)).mean()
    ent_d = (-(F.softmax(logits_d, 1) * F.log_softmax(logits_d, 1)).sum(1)).mean()
    ent = ent_s + ent_d
    loss = pol_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent

    clip_lo, clip_hi = 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps
    clip_frac = ((ratio < clip_lo) | (ratio > clip_hi)).float().mean()
    approx_kl = (old_logp_c - logp).mean()

    with torch.no_grad():
        dbg = {
            "pol": float(pol_loss.detach().cpu()),
            "v": float(v_loss.detach().cpu()),
            "ent": float(ent.detach().cpu()),
            "ent_s": float(ent_s.detach().cpu()),
            "ent_d": float(ent_d.detach().cpu()),
            "ratio_mean": float(ratio.mean().cpu()),
            "ratio_sq_mean": float((ratio * ratio).mean().cpu()),
            "clip_frac": float(clip_frac.detach().cpu()),
            "approx_kl": float(approx_kl.detach().cpu()),
            "v_pred_mean": float(v_pred.mean().cpu()),
            "old_v_mean": float(old_v.mean().cpu()),
        }
    return loss, dbg


def policy_value_loss_step(
    model: SuccessorPolicy,
    opt: torch.optim.Optimizer,
    obs: torch.Tensor,
    src_oh: torch.Tensor,
    dst_oh: torch.Tensor,
    old_logp: torch.Tensor,
    adv: torch.Tensor,
    ret_value: torch.Tensor,
    old_v: torch.Tensor,
    cfg: PPOConfig,
    *,
    mini_batch_size: int | None = None,
) -> tuple[float, dict[str, float]]:
    """
    PPO 更新。``mini_batch_size`` 为 None 或 ≥ N 时整批一次前向；
    否则按小批 **梯度累积**：``backward(loss * k/N)``，等价于全样本平均梯度，峰值显存随小批大小变化。

    使用 ``model.eval()``：ResNet 中含 BatchNorm 时，必须与 rollout / ``old_logp`` 的 eval 前向一致，
    否则 train 下 BN 用 batch 统计量会导致 ``logp`` 与 ``old_logp`` 不可比，ratio/approx_kl 失真。
    """
    model.eval()
    n = int(obs.shape[0])
    if n == 0:
        return 0.0, {"batch": 0.0}

    mbs = n if mini_batch_size is None else max(1, int(mini_batch_size))
    if mbs >= n:
        loss, dbg = _ppo_forward_loss(model, obs, src_oh, dst_oh, old_logp, adv, ret_value, old_v, cfg)
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
            "entropy_src": dbg["ent_s"],
            "entropy_dst": dbg["ent_d"],
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
    acc_pol = acc_v = acc_ent = acc_ents = acc_entd = 0.0
    acc_rm = acc_r2 = 0.0
    acc_clip = acc_kl = 0.0
    acc_vpred = acc_oldv = 0.0
    acc_loss_log = 0.0

    for start in range(0, n, mbs):
        end = min(start + mbs, n)
        w = (end - start) / n
        loss_mb, dbg = _ppo_forward_loss(
            model,
            obs[start:end],
            src_oh[start:end],
            dst_oh[start:end],
            old_logp[start:end],
            adv[start:end],
            ret_value[start:end],
            old_v[start:end],
            cfg,
        )
        (loss_mb * w).backward()
        acc_loss_log += float(loss_mb.detach().cpu()) * w
        acc_pol += dbg["pol"] * w
        acc_v += dbg["v"] * w
        acc_ent += dbg["ent"] * w
        acc_ents += dbg["ent_s"] * w
        acc_entd += dbg["ent_d"] * w
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
        "entropy_src": acc_ents,
        "entropy_dst": acc_entd,
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


def iccs_to_src_dst_onehot(iccs: str, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    x1, y1, x2, y2 = int(iccs[0]), int(iccs[1]), int(iccs[3]), int(iccs[4])
    s = y1 * 9 + x1
    d = y2 * 9 + x2
    oh_s = torch.zeros(90, device=device, dtype=dtype)
    oh_d = torch.zeros(90, device=device, dtype=dtype)
    oh_s[s] = 1.0
    oh_d[d] = 1.0
    return oh_s, oh_d
