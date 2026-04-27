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


class JointPolicyValueConvTrm(nn.Module):
    """共享卷积茎 → **双主干**：宽 ``ResBlock×1`` 出策略；宽 **单层 Transformer** 出价值。

    茎与原先一致（窄通道 + ``stem_num_res`` 个 ResBlock）；策略与价值从茎特征分流，互不共享 trunk 表示。
    价值 Transformer 默认 **``d_model=128``** 即可；容量主要给 **策略 Res 主干**（``policy_trunk_channels`` 默认较宽）。
    """

    _BOARD_TOKENS = 90  # 10 * 9

    def __init__(
        self,
        in_channels: int | None = None,
        *,
        stem_channels: int = 160,
        stem_num_res: int = 1,
        policy_trunk_channels: int = 608,
        d_model: int = 128,
        nhead: int = 8,
        trm_layers: int = 1,
        dim_feedforward: int | None = None,
        dropout: float = 0.08,
        policy_max_legal: int | None = None,
        value_scale: float = 10.0,
    ) -> None:
        super().__init__()
        c = int(in_channels if in_channels is not None else POLICY_SELECT_IN_CHANNELS)
        if int(d_model) % int(nhead) != 0:
            raise ValueError(f"价值 d_model={d_model} 须能被 nhead={nhead} 整除")
        df = int(dim_feedforward) if dim_feedforward is not None else int(d_model) * 4
        self.arch = "conv_transformer"
        self.in_channels = c
        self.stem_channels = int(stem_channels)
        self.stem_num_res = int(stem_num_res)
        self.policy_trunk_channels = int(policy_trunk_channels)
        self.d_model = int(d_model)
        self.trm_layers = max(1, int(trm_layers))
        self.trm_nhead = int(nhead)
        self.dim_feedforward = df
        self.policy_max_legal = int(policy_max_legal or POLICY_MAX_LEGAL_MOVES)
        self.value_scale = float(value_scale)
        self.filters = max(self.policy_trunk_channels, self.d_model)
        self.num_res_layers = int(self.trm_layers)

        self.stem_conv = nn.Conv2d(c, self.stem_channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(self.stem_channels)
        self.stem_blocks = nn.Sequential(
            *[ResBlock(self.stem_channels) for _ in range(max(0, self.stem_num_res))]
        )

        self.policy_proj = nn.Conv2d(self.stem_channels, self.policy_trunk_channels, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(self.policy_trunk_channels)
        self.policy_block = ResBlock(self.policy_trunk_channels)
        self.policy_pool = nn.AdaptiveAvgPool2d(1)
        self.policy_head = nn.Linear(self.policy_trunk_channels, self.policy_max_legal)

        self.value_proj = nn.Conv2d(self.stem_channels, self.d_model, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(self.d_model)
        self.value_pos_embed = nn.Parameter(torch.zeros(1, self._BOARD_TOKENS, self.d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=df,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        try:
            self.value_encoder = nn.TransformerEncoder(
                enc_layer, num_layers=self.trm_layers, enable_nested_tensor=False
            )
        except TypeError:
            self.value_encoder = nn.TransformerEncoder(enc_layer, num_layers=self.trm_layers)
        self.value_out_ln = nn.LayerNorm(self.d_model)
        self.value_fc = nn.Linear(self.d_model, 1)
        nn.init.trunc_normal_(self.value_pos_embed, std=0.02)

    def _stem_spatial(self, x_nchw: torch.Tensor) -> torch.Tensor:
        t = F.elu(self.stem_bn(self.stem_conv(x_nchw)))
        return self.stem_blocks(t)

    def _policy_trunk_flat(self, x_nchw: torch.Tensor) -> torch.Tensor:
        s = self._stem_spatial(x_nchw)
        p = F.elu(self.policy_bn(self.policy_proj(s)))
        p = self.policy_block(p)
        return self.policy_pool(p).flatten(1)

    def _value_trunk_flat(self, x_nchw: torch.Tensor) -> torch.Tensor:
        s = self._stem_spatial(x_nchw)
        v = F.elu(self.value_bn(self.value_proj(s)))
        b, _, h, w = v.shape
        if h * w != self._BOARD_TOKENS:
            raise RuntimeError(f"期望 H*W={self._BOARD_TOKENS}，得到 {h}×{w}")
        tok = v.flatten(2).transpose(1, 2) + self.value_pos_embed
        tok = self.value_encoder(tok)
        tok = self.value_out_ln(tok)
        return tok.mean(dim=1)

    def forward_trunk_split(self, x_nchw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """单次茎前向，再分流；供推理/PPO 复用特征。"""
        s = self._stem_spatial(x_nchw)
        p = F.elu(self.policy_bn(self.policy_proj(s)))
        p = self.policy_block(p)
        pol = self.policy_pool(p).flatten(1)
        v = F.elu(self.value_bn(self.value_proj(s)))
        b, _, h, w = v.shape
        if h * w != self._BOARD_TOKENS:
            raise RuntimeError(f"期望 H*W={self._BOARD_TOKENS}，得到 {h}×{w}")
        tok = v.flatten(2).transpose(1, 2) + self.value_pos_embed
        tok = self.value_encoder(tok)
        tok = self.value_out_ln(tok)
        val = tok.mean(dim=1)
        return pol, val

    def _trunk_flat(self, x_nchw: torch.Tensor) -> torch.Tensor:
        """仅策略向量（与旧 ``_trunk_flat``→policy 用法兼容；价值请用 ``_value_trunk_flat`` 或 ``forward_trunk_split``）。"""
        return self._policy_trunk_flat(x_nchw)

    def forward_heads_from_feat(
        self, policy_feat: torch.Tensor, *, value_feat: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if value_feat is None:
            raise TypeError(
                "JointPolicyValueConvTrm 需要显式 value_feat=…（请用 forward_trunk_split 或分别调用两路 trunk）"
            )
        logits_moves = self.policy_head(policy_feat)
        v = torch.tanh(self.value_fc(value_feat).squeeze(-1)) * self.value_scale
        return logits_moves, v

    def forward(self, x_nchw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pol, val = self.forward_trunk_split(x_nchw)
        return self.forward_heads_from_feat(pol, value_feat=val)


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


def count_transformer_encoder_layers_in_state(sd: dict, prefix: str = "value_encoder.layers.") -> int:
    mx = -1
    for k in sd:
        if not k.startswith(prefix):
            continue
        rest = k[len(prefix) :]
        lead = rest.split(".", 1)[0]
        if lead.isdigit():
            mx = max(mx, int(lead))
    return mx + 1 if mx >= 0 else 0


def _infer_value_d_model_from_state(sd: dict) -> int | None:
    pe = sd.get("value_pos_embed")
    if pe is not None and getattr(pe, "ndim", 0) >= 2:
        return int(pe.shape[-1])
    w = sd.get("value_encoder.layers.0.self_attn.in_proj_weight")
    if w is not None and getattr(w, "ndim", 0) == 2:
        return int(w.shape[1])
    return None


def _infer_policy_trunk_channels_from_state(sd: dict) -> int | None:
    w = sd.get("policy_block.conv1.weight")
    if w is not None and getattr(w, "ndim", 0) == 4:
        return int(w.shape[0])
    return None


def _infer_dim_feedforward_from_state(sd: dict, d_model: int) -> int:
    w = sd.get("value_encoder.layers.0.linear1.weight")
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
        out["policy_trunk_channels"] = int(getattr(model, "policy_trunk_channels"))
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


def trunk_policy_value_feats(model: nn.Module, x_nchw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """双主干 ConvTrm 返回 ``(policy_feat, value_feat)``；ResNet 两路相同。"""
    fn = getattr(model, "forward_trunk_split", None)
    if callable(fn):
        return fn(x_nchw)
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
    in_ch = int(
        in_channels
        if in_channels is not None
        else ckpt.get("in_channels", ckpt.get("select_in_channels", POLICY_SELECT_IN_CHANNELS))
    )
    pm = int(ckpt.get("policy_max_legal", POLICY_MAX_LEGAL_MOVES))
    vs = float(ckpt.get("value_scale", 10.0))

    arch = str(ckpt.get("arch", "resnet")).lower()
    if arch == "resnet" and (
        "policy_proj.weight" in sd
        or "value_pos_embed" in sd
        or (
            "pos_embed" in sd
            and "proj.weight" in sd
            and "policy_proj.weight" not in sd
        )
    ):
        arch = "conv_transformer"

    if arch == "conv_transformer":
        if "policy_proj.weight" not in sd and "proj.weight" in sd:
            raise RuntimeError(
                "该 checkpoint 为旧版「单路 Conv+Transformer」（共享 trunk），"
                "与当前「共享茎 + 策略宽 ResBlock / 价值宽 Transformer」双主干不兼容，请用新结构重训。"
            )
        _ptc = ckpt.get("policy_trunk_channels")
        ptc = int(_ptc) if _ptc is not None else int(_infer_policy_trunk_channels_from_state(sd) or 608)
        _dm = ckpt.get("trm_d_model")
        dm = (
            int(_dm)
            if _dm is not None
            else int(_infer_value_d_model_from_state(sd) or ckpt.get("filters", 128))
        )
        nl = int(ckpt.get("trm_layers", 0))
        if nl <= 0:
            nl = count_transformer_encoder_layers_in_state(sd, prefix="value_encoder.layers.") or 1
        nl = max(1, nl)
        nh = int(ckpt.get("trm_nhead", 8))
        if dm % nh != 0:
            nh = _default_nhead_for_d_model(dm)
        df = int(ckpt.get("dim_feedforward", _infer_dim_feedforward_from_state(sd, dm)))
        sc = int(ckpt.get("stem_channels", _infer_stem_channels_from_state(sd) or 160))
        snr = int(ckpt.get("stem_num_res", 0))
        if snr <= 0:
            snr = count_resnet_blocks_in_state(sd, prefix="stem_blocks.") or 1
        model: nn.Module = JointPolicyValueConvTrm(
            in_channels=in_ch,
            stem_channels=sc,
            stem_num_res=snr,
            policy_trunk_channels=ptc,
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
