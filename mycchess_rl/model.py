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
        self.arch = "resnet"
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


class JointPolicyValueConvTrm(nn.Module):
    """卷积残差茎压稀疏根平面 → 空间 token → **浅而宽** TransformerEncoder → 策略/价值。

    棋盘固定 ``10×9``，展平为 ``90`` 个 token；默认 ``d_model=384``、``2`` 层 encoder，利于短训内拟合价值。
    """

    _BOARD_TOKENS = 90  # 10 * 9

    def __init__(
        self,
        in_channels: int | None = None,
        *,
        stem_channels: int = 96,
        stem_num_res: int = 2,
        d_model: int = 384,
        nhead: int = 8,
        trm_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.08,
        policy_max_legal: int | None = None,
        value_scale: float = 10.0,
    ) -> None:
        super().__init__()
        c = int(in_channels if in_channels is not None else POLICY_SELECT_IN_CHANNELS)
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} 须能被 nhead={nhead} 整除")
        df = int(dim_feedforward) if dim_feedforward is not None else int(d_model) * 4
        self.arch = "conv_transformer"
        self.in_channels = c
        self.stem_channels = int(stem_channels)
        self.stem_num_res = int(stem_num_res)
        self.d_model = int(d_model)
        self.trm_layers = int(trm_layers)
        self.trm_nhead = int(nhead)
        self.dim_feedforward = df
        self.policy_max_legal = int(policy_max_legal or POLICY_MAX_LEGAL_MOVES)
        self.value_scale = float(value_scale)
        # 与 ResNet 版 checkpoint 字段对齐，便于日志/旧脚本读取
        self.filters = self.d_model
        self.num_res_layers = int(trm_layers)

        self.stem_conv = nn.Conv2d(c, self.stem_channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(self.stem_channels)
        self.stem_blocks = nn.Sequential(
            *[ResBlock(self.stem_channels) for _ in range(max(0, self.stem_num_res))]
        )
        self.proj = nn.Conv2d(self.stem_channels, self.d_model, 1, bias=False)
        self.proj_bn = nn.BatchNorm2d(self.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self._BOARD_TOKENS, self.d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=df,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        try:
            self.encoder = nn.TransformerEncoder(
                enc, num_layers=self.trm_layers, enable_nested_tensor=False
            )
        except TypeError:
            self.encoder = nn.TransformerEncoder(enc, num_layers=self.trm_layers)
        self.out_ln = nn.LayerNorm(self.d_model)
        self.policy_head = nn.Linear(self.d_model, self.policy_max_legal)
        self.value_fc = nn.Linear(self.d_model, 1)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _trunk_flat(self, x_nchw: torch.Tensor) -> torch.Tensor:
        t = F.elu(self.stem_bn(self.stem_conv(x_nchw)))
        t = self.stem_blocks(t)
        t = F.elu(self.proj_bn(self.proj(t)))
        b, _, h, w = t.shape
        if h * w != self._BOARD_TOKENS:
            raise RuntimeError(f"期望 H*W={self._BOARD_TOKENS}，得到 {h}×{w}")
        tok = t.flatten(2).transpose(1, 2)
        tok = tok + self.pos_embed
        tok = self.encoder(tok)
        tok = self.out_ln(tok)
        return tok.mean(dim=1)

    def forward_heads_from_feat(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits_moves = self.policy_head(feat)
        v = torch.tanh(self.value_fc(feat).squeeze(-1)) * self.value_scale
        return logits_moves, v

    def forward(self, x_nchw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_heads_from_feat(self._trunk_flat(x_nchw))


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


def count_transformer_encoder_layers_in_state(sd: dict, prefix: str = "encoder.layers.") -> int:
    mx = -1
    for k in sd:
        if not k.startswith(prefix):
            continue
        rest = k[len(prefix) :]
        lead = rest.split(".", 1)[0]
        if lead.isdigit():
            mx = max(mx, int(lead))
    return mx + 1 if mx >= 0 else 0


def _infer_trm_d_model_from_state(sd: dict) -> int | None:
    pe = sd.get("pos_embed")
    if pe is not None and getattr(pe, "ndim", 0) >= 2:
        return int(pe.shape[-1])
    w = sd.get("encoder.layers.0.self_attn.in_proj_weight")
    if w is not None and getattr(w, "ndim", 0) == 2:
        return int(w.shape[1])
    return None


def _infer_dim_feedforward_from_state(sd: dict, d_model: int) -> int:
    w = sd.get("encoder.layers.0.linear1.weight")
    if w is not None and getattr(w, "ndim", 0) == 2:
        return int(w.shape[0])
    return int(d_model) * 4


def _infer_stem_channels_from_state(sd: dict) -> int | None:
    w = sd.get("stem_conv.weight")
    if w is not None and getattr(w, "ndim", 0) == 4:
        return int(w.shape[0])
    return None


def _default_nhead_for_d_model(d_model: int) -> int:
    for h in (8, 6, 4, 2, 1):
        if d_model % h == 0:
            return h
    return 1


def policy_value_checkpoint_meta(model: nn.Module) -> dict:
    """训练脚本存盘用：写入 ``arch`` 及与 ``load_policy_value_for_play`` 对称的字段。"""
    arch = str(getattr(model, "arch", "resnet")).lower()
    out: dict = {"arch": arch}
    if arch == "conv_transformer":
        out["stem_channels"] = int(getattr(model, "stem_channels"))
        out["stem_num_res"] = int(getattr(model, "stem_num_res"))
        out["trm_d_model"] = int(getattr(model, "d_model"))
        out["trm_layers"] = int(getattr(model, "trm_layers"))
        out["trm_nhead"] = int(getattr(model, "trm_nhead"))
        out["dim_feedforward"] = int(getattr(model, "dim_feedforward"))
        out["num_res_layers"] = int(getattr(model, "num_res_layers"))
        out["filters"] = int(getattr(model, "filters"))
    else:
        out["num_res_layers"] = int(model.num_res_layers)
        out["filters"] = int(model.filters)
    return out


PolicyValueBackbone = JointPolicyValueNet | JointPolicyValueConvTrm


def load_policy_value_for_play(
    checkpoint: Path,
    device: torch.device,
    *,
    in_channels: int | None = None,
) -> tuple[PolicyValueBackbone, dict[str, list[str]]]:
    from mycchess_rl.chess import FEATURE_LIST

    ckpt = torch_load_checkpoint(checkpoint, device)
    sd = ckpt["model"]
    in_ch = int(
        in_channels
        if in_channels is not None
        else ckpt.get("in_channels", ckpt.get("select_in_channels", POLICY_SELECT_IN_CHANNELS))
    )
    pm = int(ckpt.get("policy_max_legal", POLICY_MAX_LEGAL_MOVES))
    vs = float(ckpt.get("value_scale", 10.0))

    arch = str(ckpt.get("arch", "resnet")).lower()
    if arch == "resnet" and (
        "pos_embed" in sd or any(str(k).startswith("encoder.layers.") for k in sd)
    ):
        arch = "conv_transformer"

    if arch == "conv_transformer":
        dm = int(ckpt.get("trm_d_model", _infer_trm_d_model_from_state(sd) or ckpt.get("filters", 384)))
        nl = int(ckpt.get("trm_layers", 0))
        if nl <= 0:
            nl = count_transformer_encoder_layers_in_state(sd) or int(ckpt.get("num_res_layers", 2))
        nh = int(ckpt.get("trm_nhead", 8))
        if dm % nh != 0:
            nh = _default_nhead_for_d_model(dm)
        df = int(ckpt.get("dim_feedforward", _infer_dim_feedforward_from_state(sd, dm)))
        sc = int(ckpt.get("stem_channels", _infer_stem_channels_from_state(sd) or 96))
        snr = int(ckpt.get("stem_num_res", 0))
        if snr <= 0:
            snr = count_resnet_blocks_in_state(sd, prefix="stem_blocks.") or 2
        model: nn.Module = JointPolicyValueConvTrm(
            in_channels=in_ch,
            stem_channels=sc,
            stem_num_res=snr,
            d_model=dm,
            nhead=nh,
            trm_layers=nl,
            dim_feedforward=df,
            policy_max_legal=pm,
            value_scale=vs,
        ).to(device)
    else:
        filters = int(ckpt.get("filters", _infer_filters_from_state(sd)))
        num_res = int(ckpt.get("num_res_layers", 0))
        if num_res <= 0:
            num_res = count_resnet_blocks_in_state(sd) or 10
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
