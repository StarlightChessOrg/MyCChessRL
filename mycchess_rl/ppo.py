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
) -> float:
    model.train()
    logits_s, logits_d, logits_v = model(obs, src_oh)
    logp_s = (F.log_softmax(logits_s, dim=1) * src_oh).sum(dim=1)
    logp_d = (F.log_softmax(logits_d, dim=1) * dst_oh).sum(dim=1)
    logp = torch.clamp(logp_s + logp_d, -80.0, 0.0)
    old_logp = torch.clamp(old_logp, -80.0, 0.0)
    ratio = torch.exp(torch.clamp(logp - old_logp, -5.0, 5.0))
    ratio = torch.clamp(ratio, 0.0, 32.0)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
    pol_loss = -torch.min(surr1, surr2).mean()
    v_pred = value_expectation_from_logits(logits_v)
    v_loss = F.smooth_l1_loss(v_pred, ret_value, beta=0.5)
    ent = (-(F.softmax(logits_s, 1) * F.log_softmax(logits_s, 1)).sum(1)).mean() + (
        -(F.softmax(logits_d, 1) * F.log_softmax(logits_d, 1)).sum(1)
    ).mean()
    loss = pol_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
    opt.step()
    return float(loss.detach().cpu())


def iccs_to_src_dst_onehot(iccs: str, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    x1, y1, x2, y2 = int(iccs[0]), int(iccs[1]), int(iccs[3]), int(iccs[4])
    s = y1 * 9 + x1
    d = y2 * 9 + x2
    oh_s = torch.zeros(90, device=device, dtype=dtype)
    oh_d = torch.zeros(90, device=device, dtype=dtype)
    oh_s[s] = 1.0
    oh_d[d] = 1.0
    return oh_s, oh_d
