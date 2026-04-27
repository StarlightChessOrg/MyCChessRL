"""Inception-v3 风格浅宽共享主干 + 策略/价值头（残差堆叠与双路 Conv+Trm 已移除，旧权重不兼容）。"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from mycchess_rl.chess.rationale import POLICY_MAX_LEGAL_MOVES, POLICY_SELECT_IN_CHANNELS
from mycchess_rl.hierarchical_sl import HIER_PIECE_HEAD_DIM, HIER_SQUARE_HEAD_DIM

# 每元组：(1×1, 1×1→3×3 降维, 3×3, 1×1→3×3→3×3 两段降维/中间/输出, 池化后 1×1)
# 两支 3×3 串联为 v3 「最新常用」因子分解形式；浅塔仅 2 个 Inception 模块 + 宽 stem。
DEFAULT_INCEPTION_SPECS: tuple[tuple[int, int, int, int, int, int, int], ...] = (
    (96, 64, 128, 64, 96, 128, 64),
    (128, 96, 192, 96, 128, 192, 96),
)


def _conv_bn(in_ch: int, out_ch: int, k: int, *, padding: int = 0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
    )


class InceptionModule(nn.Module):
    """四支并行：1×1；1×1→3×3；1×1→3×3→3×3；AvgPool3×3→1×1（空间尺寸不变）。"""

    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3reduce: int,
        ch3x3: int,
        ch3x3dbl_reduce: int,
        ch3x3dbl_1: int,
        ch3x3dbl_2: int,
        pool_planes: int,
    ) -> None:
        super().__init__()
        self.b1 = nn.Sequential(_conv_bn(in_channels, ch1x1, 1), nn.ELU(inplace=True))
        self.b2 = nn.Sequential(
            _conv_bn(in_channels, ch3x3reduce, 1),
            nn.ELU(inplace=True),
            _conv_bn(ch3x3reduce, ch3x3, 3, padding=1),
            nn.ELU(inplace=True),
        )
        self.b3 = nn.Sequential(
            _conv_bn(in_channels, ch3x3dbl_reduce, 1),
            nn.ELU(inplace=True),
            _conv_bn(ch3x3dbl_reduce, ch3x3dbl_1, 3, padding=1),
            nn.ELU(inplace=True),
            _conv_bn(ch3x3dbl_1, ch3x3dbl_2, 3, padding=1),
            nn.ELU(inplace=True),
        )
        self.b4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            _conv_bn(in_channels, pool_planes, 1),
            nn.ELU(inplace=True),
        )
        self.out_channels = ch1x1 + ch3x3 + ch3x3dbl_2 + pool_planes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class InceptionPolicyBackbone(nn.Module):
    """浅而宽：stem + 若干 InceptionModule，全局均值池化得到共享特征向量。"""

    def __init__(
        self,
        in_channels: int,
        stem_channels: int,
        specs: tuple[tuple[int, int, int, int, int, int, int], ...],
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.stem_channels = int(stem_channels)
        self.stem = nn.Sequential(
            _conv_bn(self.in_channels, self.stem_channels, 3, padding=1),
            nn.ELU(inplace=True),
        )
        layers: list[InceptionModule] = []
        c = self.stem_channels
        for t in specs:
            m = InceptionModule(c, *t)
            layers.append(m)
            c = m.out_channels
        self.tower = nn.ModuleList(layers)
        self.out_dim = c
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x_nchw: torch.Tensor) -> torch.Tensor:
        h = self.stem(x_nchw)
        for blk in self.tower:
            h = blk(h)
        return self.pool(h).flatten(1)


class InceptionJointPolicyValueNet(nn.Module):
    """联合合法槽策略 + 价值；策略与价值共用 ``InceptionPolicyBackbone`` 输出。"""

    def __init__(
        self,
        in_channels: int | None = None,
        *,
        stem_channels: int = 128,
        inception_specs: tuple[tuple[int, int, int, int, int, int, int], ...] | None = None,
        policy_max_legal: int | None = None,
        value_scale: float = 10.0,
    ) -> None:
        super().__init__()
        c = int(in_channels if in_channels is not None else POLICY_SELECT_IN_CHANNELS)
        sp = inception_specs if inception_specs is not None else DEFAULT_INCEPTION_SPECS
        self.arch = "inception_joint"
        self.policy_kind = "joint"
        self.in_channels = c
        self.stem_channels = int(stem_channels)
        self.inception_specs: tuple[tuple[int, int, int, int, int, int, int], ...] = tuple(
            tuple(int(x) for x in row) for row in sp
        )
        self.policy_max_legal = int(policy_max_legal or POLICY_MAX_LEGAL_MOVES)
        self.value_scale = float(value_scale)
        self.backbone = InceptionPolicyBackbone(c, self.stem_channels, self.inception_specs)
        d = self.backbone.out_dim
        self.policy_head = nn.Linear(d, self.policy_max_legal)
        self.value_fc = nn.Linear(d, 1)
        self.filters = self.stem_channels
        self.num_res_layers = len(self.inception_specs)

    def _trunk_flat(self, x_nchw: torch.Tensor) -> torch.Tensor:
        return self.backbone(x_nchw)

    def forward_heads_from_feat(
        self, feat: torch.Tensor, *, value_feat: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vf = feat if value_feat is None else value_feat
        logits_moves = self.policy_head(feat)
        v = torch.tanh(self.value_fc(vf).squeeze(-1)) * self.value_scale
        return logits_moves, v

    def forward(self, x_nchw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = self._trunk_flat(x_nchw)
        return self.forward_heads_from_feat(t)


class InceptionHierarchicalPolicyValueNet(nn.Module):
    """子种→源→目 三头 + 价值；与联合槽位不兼容。"""

    def __init__(
        self,
        in_channels: int | None = None,
        *,
        stem_channels: int = 128,
        inception_specs: tuple[tuple[int, int, int, int, int, int, int], ...] | None = None,
        value_scale: float = 10.0,
    ) -> None:
        super().__init__()
        c = int(in_channels if in_channels is not None else POLICY_SELECT_IN_CHANNELS)
        sp = inception_specs if inception_specs is not None else DEFAULT_INCEPTION_SPECS
        self.arch = "inception_hierarchical"
        self.policy_kind = "hierarchical"
        self.in_channels = c
        self.stem_channels = int(stem_channels)
        self.inception_specs = tuple(tuple(int(x) for x in row) for row in sp)
        self.policy_max_legal = 0
        self.value_scale = float(value_scale)
        self.backbone = InceptionPolicyBackbone(c, self.stem_channels, self.inception_specs)
        d = self.backbone.out_dim
        self.policy_head_type = nn.Linear(d, HIER_PIECE_HEAD_DIM)
        self.policy_head_from = nn.Linear(d, HIER_SQUARE_HEAD_DIM)
        self.policy_head_to = nn.Linear(d, HIER_SQUARE_HEAD_DIM)
        self.value_fc = nn.Linear(d, 1)
        self.filters = self.stem_channels
        self.num_res_layers = len(self.inception_specs)

    def _trunk_flat(self, x_nchw: torch.Tensor) -> torch.Tensor:
        return self.backbone(x_nchw)

    def forward(
        self, x_nchw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self._trunk_flat(x_nchw)
        lt = self.policy_head_type(feat)
        lf = self.policy_head_from(feat)
        lto = self.policy_head_to(feat)
        v = torch.tanh(self.value_fc(feat).squeeze(-1)) * self.value_scale
        return lt, lf, lto, v


def torch_load_checkpoint(path: str | Path, map_location: torch.device | str) -> dict:
    p = Path(path)
    try:
        return torch.load(p, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(p, map_location=map_location)


def _checkpoint_is_legacy_resnet_or_convtrm(sd: dict) -> bool:
    if "backbone.stem.0.weight" in sd:
        return False
    if any(k.startswith("blocks.") for k in sd):
        return True
    if "policy_proj.weight" in sd or "value_encoder.layers" in sd:
        return True
    if "stem_conv.weight" in sd and "backbone" not in "".join(sd.keys()):
        return True
    return False


def _parse_inception_specs_ckpt(ckpt: dict) -> tuple[tuple[int, int, int, int, int, int, int], ...]:
    raw = ckpt.get("inception_specs")
    if isinstance(raw, list) and raw:
        try:
            rows: list[tuple[int, int, int, int, int, int, int]] = []
            for row in raw:
                if not isinstance(row, (list, tuple)) or len(row) != 7:
                    return DEFAULT_INCEPTION_SPECS
                rows.append(tuple(int(x) for x in row))
            return tuple(rows)
        except (TypeError, ValueError):
            return DEFAULT_INCEPTION_SPECS
    return DEFAULT_INCEPTION_SPECS


def policy_value_checkpoint_meta(model: nn.Module) -> dict:
    arch = str(getattr(model, "arch", "inception_joint")).lower()
    out: dict = {"arch": arch, "policy_kind": str(getattr(model, "policy_kind", "joint"))}
    out["stem_channels"] = int(getattr(model, "stem_channels"))
    specs = getattr(model, "inception_specs", DEFAULT_INCEPTION_SPECS)
    out["inception_specs"] = [list(map(int, row)) for row in specs]
    out["num_res_layers"] = len(specs)
    out["filters"] = int(getattr(model, "stem_channels"))
    out["trunk_dim"] = int(model.backbone.out_dim)  # type: ignore[attr-defined]
    return out


PolicyValueBackbone = InceptionJointPolicyValueNet | InceptionHierarchicalPolicyValueNet


def trunk_policy_value_feats(model: nn.Module, x_nchw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """共享主干：策略与价值使用同一池化特征向量。"""
    t = model._trunk_flat(x_nchw)
    return t, t


def load_policy_value_for_play(
    checkpoint: Path,
    device: torch.device,
    *,
    in_channels: int | None = None,
) -> tuple[PolicyValueBackbone, dict[str, list[str]]]:
    from mycchess_rl.chess import FEATURE_LIST

    ckpt = torch_load_checkpoint(checkpoint, device)
    sd = ckpt["model"]
    if _checkpoint_is_legacy_resnet_or_convtrm(sd):
        raise RuntimeError(
            "该 checkpoint 为已移除的 ResNet / Conv+Transformer 结构，"
            "请改用 Inception 共享主干重新训练（当前仅支持 arch=inception_joint / inception_hierarchical）。"
        )
    in_ch = int(
        in_channels
        if in_channels is not None
        else ckpt.get("in_channels", ckpt.get("select_in_channels", POLICY_SELECT_IN_CHANNELS))
    )
    pm = int(ckpt.get("policy_max_legal", POLICY_MAX_LEGAL_MOVES))
    vs = float(ckpt.get("value_scale", 10.0))
    stem = int(ckpt.get("stem_channels", 128))
    specs = _parse_inception_specs_ckpt(ckpt)

    if "policy_head_type.weight" in sd:
        model: nn.Module = InceptionHierarchicalPolicyValueNet(
            in_channels=in_ch,
            stem_channels=stem,
            inception_specs=specs,
            value_scale=vs,
        ).to(device)
    else:
        model = InceptionJointPolicyValueNet(
            in_channels=in_ch,
            stem_channels=stem,
            inception_specs=specs,
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
