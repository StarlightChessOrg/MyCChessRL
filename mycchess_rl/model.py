"""合法着法联合策略 + 标量价值（与旧两阶段 head 不兼容，需重新训练）。"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from mycchess_rl.chess.rationale import POLICY_MAX_LEGAL_MOVES, POLICY_SELECT_IN_CHANNELS


def count_resnet_blocks_in_state(sd: dict, prefix: str = "blocks.") -> int:
    mx = -1
    for k in sd:
        if not k.startswith(prefix):
            continue
        rest = k[len(prefix) :]
        lead = rest.split(".", 1)[0]
        if lead.isdigit():
            mx = max(mx, int(lead))
    return mx + 1 if mx >= 0 else 0


class ResBlock(nn.Module):
    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.elu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        return F.elu(out)


class JointPolicyValueNet(nn.Module):
    """(B,C,10,9) → 对至多 ``policy_max_legal`` 个**有序合法着法槽位**的 logits；价值为 ``tanh·value_scale`` 标量。"""

    def __init__(
        self,
        num_res_layers: int = 10,
        in_channels: int | None = None,
        filters: int = 256,
        *,
        policy_max_legal: int | None = None,
        value_scale: float = 10.0,
    ) -> None:
        super().__init__()
        c = in_channels if in_channels is not None else POLICY_SELECT_IN_CHANNELS
        self.in_channels = int(c)
        self.filters = int(filters)
        self.num_res_layers = int(num_res_layers)
        self.policy_max_legal = int(policy_max_legal or POLICY_MAX_LEGAL_MOVES)
        self.value_scale = float(value_scale)

        self.stem_conv = nn.Conv2d(self.in_channels, self.filters, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(self.filters)
        self.blocks = nn.Sequential(*[ResBlock(self.filters) for _ in range(self.num_res_layers)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.policy_head = nn.Linear(self.filters, self.policy_max_legal)
        self.value_fc = nn.Linear(self.filters, 1)

    def _trunk_flat(self, x_nchw: torch.Tensor) -> torch.Tensor:
        t = F.elu(self.stem_bn(self.stem_conv(x_nchw)))
        t = self.blocks(t)
        return self.pool(t).flatten(1)

    def forward_heads_from_feat(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits_moves = self.policy_head(feat)
        v = torch.tanh(self.value_fc(feat).squeeze(-1)) * self.value_scale
        return logits_moves, v

    def forward(self, x_nchw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_heads_from_feat(self._trunk_flat(x_nchw))


def torch_load_checkpoint(path: str | Path, map_location: torch.device | str) -> dict:
    p = Path(p)
    try:
        return torch.load(p, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(p, map_location=map_location)


def _infer_filters_from_state(sd: dict) -> int:
    w = sd.get("stem_conv.weight")
    if w is not None:
        return int(w.shape[0])
    return 256


def load_policy_value_for_play(
    checkpoint: Path,
    device: torch.device,
    *,
    in_channels: int | None = None,
) -> tuple[JointPolicyValueNet, dict[str, list[str]]]:
    from mycchess_rl.chess import FEATURE_LIST

    ckpt = torch_load_checkpoint(checkpoint, device)
    sd = ckpt["model"]
    filters = int(ckpt.get("filters", _infer_filters_from_state(sd)))
    num_res = int(ckpt.get("num_res_layers", 0))
    if num_res <= 0:
        num_res = count_resnet_blocks_in_state(sd) or 10
    in_ch = int(
        in_channels
        if in_channels is not None
        else ckpt.get("in_channels", ckpt.get("select_in_channels", POLICY_SELECT_IN_CHANNELS))
    )
    pm = int(ckpt.get("policy_max_legal", POLICY_MAX_LEGAL_MOVES))
    vs = float(ckpt.get("value_scale", 10.0))
    model = JointPolicyValueNet(
        num_res_layers=num_res,
        in_channels=in_ch,
        filters=filters,
        policy_max_legal=pm,
        value_scale=vs,
    ).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    flist: dict[str, list[str]] = {
        "red": list(FEATURE_LIST["red"]),
        "black": list(FEATURE_LIST["black"]),
    }
    return model, flist


def policy_temperature_scalar(policy_temperature: float) -> float:
    t = float(policy_temperature)
    if not (t > 0.0) or math.isnan(t) or math.isinf(t):
        return 1.0
    return t
