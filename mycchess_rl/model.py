"""两阶段策略塔（起点 + 落点 + 行棋方三分类价值）；根卷积输入通道与 icyElephant 14 路棋子平面一致。"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from mycchess_rl.chess.rationale import POLICY_GRID_NUMEL, POLICY_SELECT_IN_CHANNELS


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


class SuccessorPolicy(nn.Module):
    """(B,C,10,9) → logits_src (B,90), logits_dst (B,90|feat+src_oh), logits_val (B,3)。"""

    def __init__(
        self,
        num_res_layers: int = 10,
        in_channels: int | None = None,
        filters: int = 256,
        grid: int = POLICY_GRID_NUMEL,
    ) -> None:
        super().__init__()
        c = in_channels if in_channels is not None else POLICY_SELECT_IN_CHANNELS
        self.in_channels = c
        self.grid = grid
        self.filters = filters
        self.stem_conv = nn.Conv2d(c, filters, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(filters)
        self.blocks = nn.Sequential(*[ResBlock(filters) for _ in range(num_res_layers)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_src = nn.Linear(filters, grid)
        self.head_dst = nn.Linear(filters + grid, grid)
        self.value_head = nn.Linear(filters, 3)

    def _trunk_flat(self, x_nchw: torch.Tensor) -> torch.Tensor:
        t = F.elu(self.stem_bn(self.stem_conv(x_nchw)))
        t = self.blocks(t)
        return self.pool(t).flatten(1)

    def forward(
        self, x_cur: torch.Tensor, src_one_hot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self._trunk_flat(x_cur)
        logits_src = self.head_src(feat)
        logits_dst = self.head_dst(torch.cat([feat, src_one_hot], dim=1))
        logits_val = self.value_head(feat)
        return logits_src, logits_dst, logits_val


def torch_load_checkpoint(path: str | Path, map_location: torch.device | str) -> dict:
    p = Path(path)
    try:
        return torch.load(p, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(p, map_location=map_location)


def _infer_filters_from_state(sd: dict) -> int:
    w = sd.get("stem_conv.weight")
    if w is not None:
        return int(w.shape[0])
    return 256


def load_successor_policy_for_play(
    checkpoint: Path,
    device: torch.device,
    *,
    in_channels: int | None = None,
) -> tuple[SuccessorPolicy, dict[str, list[str]]]:
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
    model = SuccessorPolicy(
        num_res_layers=num_res,
        in_channels=in_ch,
        filters=filters,
    ).to(device)
    model.load_state_dict(sd, strict=False)
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
