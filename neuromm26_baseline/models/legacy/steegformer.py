"""ST-EEGFormer adapter for NeuroMM EEG tensors.

Adapted from https://github.com/LiuyinYang1101/STEEGFormer.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block

from neuromm26_baseline.models.rhop_pooling import RiemannianHighOrderPooling

LOGGER = logging.getLogger(__name__)

VARIANT_CONFIGS: dict[str, dict[str, int]] = {
    "small": {"embed_dim": 512, "depth": 8, "num_heads": 8},
    "base": {"embed_dim": 768, "depth": 12, "num_heads": 12},
    "large": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
    "largev2": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
}


CHANNEL_PRESETS: dict[str, tuple[int, ...]] = {
    "seed23": (
        6,
        112,
        60,
        86,
        13,
        133,
        72,
        71,
        42,
        25,
        54,
        11,
        3,
        118,
        122,
        49,
        70,
        18,
        14,
        34,
        75,
        19,
        22,
    ),
    "seed26": (
        6,
        112,
        60,
        86,
        13,
        133,
        72,
        71,
        42,
        25,
        54,
        11,
        3,
        118,
        122,
        49,
        70,
        18,
        14,
        34,
        75,
        19,
        22,
        65,
        20,
        130,
    ),
    "neuromm21_strict": (
        6,    # Fp1
        60,   # Fp2
        71,   # F3
        11,   # F4
        130,  # C3
        2,    # C4
        74,   # P3
        116,  # P4
        95,   # O1
        103,  # O2
        133,  # F7
        118,  # F8
        61,   # T3
        93,   # T4
        50,   # T5
        81,   # T6
        25,   # Fz
        110,  # Cz
        1,    # Pz
        73,   # A1
        66,   # A2
    ),
}

STANDARD_1020_COORDS: dict[str, tuple[float, float, float]] = {
    # Approximate normalized 10-20 coordinates in NeuroMM channel order.
    "Fp1": (-0.35, 0.95, 0.15),
    "Fp2": (0.35, 0.95, 0.15),
    "F3": (-0.45, 0.50, 0.45),
    "F4": (0.45, 0.50, 0.45),
    "C3": (-0.55, 0.00, 0.60),
    "C4": (0.55, 0.00, 0.60),
    "P3": (-0.45, -0.50, 0.45),
    "P4": (0.45, -0.50, 0.45),
    "O1": (-0.30, -0.95, 0.15),
    "O2": (0.30, -0.95, 0.15),
    "F7": (-0.90, 0.45, 0.20),
    "F8": (0.90, 0.45, 0.20),
    "T3": (-1.00, 0.00, 0.10),
    "T4": (1.00, 0.00, 0.10),
    "T5": (-0.90, -0.45, 0.10),
    "T6": (0.90, -0.45, 0.10),
    "Fz": (0.00, 0.55, 0.65),
    "Cz": (0.00, 0.00, 1.00),
    "Pz": (0.00, -0.55, 0.65),
    "PG1": (-0.55, -0.75, 0.20),
    "PG2": (0.55, -0.75, 0.20),
    "A1": (-1.05, -0.10, -0.10),
    "A2": (1.05, -0.10, -0.10),
    "ECG": (0.00, 0.00, -1.00),
    "EMG": (0.00, -0.20, -0.90),
    "EXG": (0.00, 0.20, -0.90),
}


MONTAGE_NAMES_BY_PRESET: dict[str, tuple[str, ...]] = {
    "seed23": (
        "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8",
        "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz", "PG1", "PG2", "A1", "A2",
    ),
    "seed26": (
        "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8",
        "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz", "PG1", "PG2", "A1", "A2",
        "ECG", "EMG", "EXG",
    ),
    "neuromm21_strict": (
        "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8",
        "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz", "A1", "A2",
    ),
}


def _montage_coordinate_tensor(channel_preset: str, use_channels: int) -> torch.Tensor:
    names = MONTAGE_NAMES_BY_PRESET.get(channel_preset, MONTAGE_NAMES_BY_PRESET["seed23"])
    coords: list[tuple[float, float, float]] = []
    for name in names[: int(use_channels)]:
        coords.append(STANDARD_1020_COORDS.get(name, (0.0, 0.0, 0.0)))
    while len(coords) < int(use_channels):
        coords.append((0.0, 0.0, 0.0))
    return torch.tensor(coords, dtype=torch.float32)


def _montage_sinusoidal_feature_tensor(coordinates: torch.Tensor, embed_dim: int) -> torch.Tensor:
    embed_dim = int(embed_dim)
    if embed_dim <= 0:
        return coordinates.new_zeros((coordinates.size(0), 0))
    coords = coordinates.float()
    radius = coords.norm(dim=-1, keepdim=True)
    base = torch.cat([coords, radius], dim=-1)
    bands = max(1, math.ceil(embed_dim / (2 * base.size(1))))
    frequencies = torch.exp(torch.linspace(0.0, math.log(16.0), steps=bands, dtype=base.dtype))
    phase = base.unsqueeze(-1) * frequencies.view(1, 1, -1)
    features = torch.cat([torch.sin(phase), torch.cos(phase)], dim=1).flatten(start_dim=1)
    if features.size(1) < embed_dim:
        pad = features.new_zeros((features.size(0), embed_dim - features.size(1)))
        features = torch.cat([features, pad], dim=1)
    return features[:, :embed_dim].contiguous()



class PatchEmbedEEG(nn.Module):
    def __init__(self, patch_size: int = 16, embed_dim: int = 512) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Linear(self.patch_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = x.shape
        usable = (length // self.patch_size) * self.patch_size
        if usable != length:
            x = x[..., :usable]
        seq = usable // self.patch_size
        x = x.view(batch_size, channels, seq, self.patch_size)
        x = x.permute(0, 2, 1, 3).contiguous()
        return self.proj(x)


class ChannelPositionalEmbed(nn.Module):
    def __init__(self, embedding_dim: int, num_embeddings: int = 145) -> None:
        super().__init__()
        self.channel_transformation = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.zeros_(self.channel_transformation.weight)

    def forward(self, channel_indices: torch.Tensor) -> torch.Tensor:
        return self.channel_transformation(channel_indices)


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).float())
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position.float() * div_term)
        pe[0, :, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer("pe", pe)

    def get_cls_token(self) -> torch.Tensor:
        return self.pe[0, 0, :]

    def forward(self, seq_indices: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = seq_indices.shape
        return self.pe[0, seq_indices.reshape(-1)].view(batch_size, seq_len, -1)


class STEEGFormerEncoder(nn.Module):
    def __init__(
        self,
        *,
        patch_size: int = 16,
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        channel_embedding_size: int = 145,
        temporal_max_len: int = 512,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_embed = PatchEmbedEEG(patch_size=patch_size, embed_dim=embed_dim)
        self.enc_channel_emd = ChannelPositionalEmbed(embed_dim, channel_embedding_size)
        self.enc_temporal_emd = TemporalPositionalEncoding(embed_dim, temporal_max_len)
        self.pos_drop = nn.Dropout(p=drop_rate)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dpr = torch.linspace(0, drop_path_rate, steps=depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[index],
                    norm_layer=norm_layer,
                )
                for index in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward_tokens(
        self,
        eeg: torch.Tensor,
        chan_idx: torch.Tensor,
        channel_extra_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = eeg.shape[0]
        x = self.patch_embed(eeg)
        batch_size, seq, channels, dim = x.shape
        token_count = seq * channels
        x = x.view(batch_size, token_count, dim)

        chan_idx = chan_idx.to(eeg.device)
        eeg_chan_indices = chan_idx.unsqueeze(0).unsqueeze(1).repeat(batch_size, seq, 1).view(batch_size, token_count)
        seq_tensor = torch.arange(1, seq + 1, device=eeg.device)
        eeg_seq_indices = seq_tensor.unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, channels).view(batch_size, token_count)
        channel_embedding = self.enc_channel_emd(eeg_chan_indices)
        if channel_extra_embedding is not None:
            extra = channel_extra_embedding.to(device=eeg.device, dtype=x.dtype)
            if extra.shape != (channels, dim):
                raise ValueError(
                    f"channel_extra_embedding shape {tuple(extra.shape)} does not match {(channels, dim)}"
                )
            extra = extra.unsqueeze(0).unsqueeze(0).expand(batch_size, seq, channels, dim)
            channel_embedding = channel_embedding + extra.reshape(batch_size, token_count, dim)
        x = x + self.enc_temporal_emd(eeg_seq_indices) + channel_embedding

        cls_token = self.cls_token + self.enc_temporal_emd.get_cls_token()
        cls_tokens = cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

    def forward_features(
        self,
        eeg: torch.Tensor,
        chan_idx: torch.Tensor,
        channel_extra_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.forward_tokens(eeg, chan_idx, channel_extra_embedding=channel_extra_embedding)
        return x[:, 0]

    def forward(
        self,
        eeg: torch.Tensor,
        chan_idx: torch.Tensor,
        channel_extra_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_features(eeg, chan_idx, channel_extra_embedding=channel_extra_embedding)


class LoRALinear(nn.Module):
    """Low-rank additive adapter for an existing linear projection."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(float(dropout))
        self.scaling = float(alpha) / float(rank)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        weight_key = prefix + "weight"
        bias_key = prefix + "bias"
        base_weight_key = prefix + "base.weight"
        base_bias_key = prefix + "base.bias"
        if weight_key in state_dict and base_weight_key not in state_dict:
            state_dict[base_weight_key] = state_dict.pop(weight_key)
        if bias_key in state_dict and base_bias_key not in state_dict:
            state_dict[base_bias_key] = state_dict.pop(bias_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EXGMoCELiteExpert(nn.Module):
    """Small EXG expert with bounded residual gating.

    This is a competition-safe version of the EEG-MoCE idea: EXG gets a
    separate physiological encoder and a reliability scalar, but can only add a
    capped residual to the main EEG logit.
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        token_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.08,
        gate_max: float = 0.12,
        gate_init: float = 0.25,
        delta_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.gate_max = float(gate_max)
        self.delta_scale = float(delta_scale)
        token_dim = int(token_dim)
        hidden_dim = int(hidden_dim)
        dropout = float(dropout)
        if self.in_channels <= 0:
            raise ValueError("EXGMoCELiteExpert requires in_channels > 0")
        if token_dim <= 0 or hidden_dim <= 0:
            raise ValueError("EXGMoCELiteExpert token_dim/hidden_dim must be positive")

        self.encoder = nn.Sequential(
            nn.Conv1d(self.in_channels, 32, kernel_size=33, stride=4, padding=16, bias=False),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=17, stride=2, padding=8, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, token_dim, kernel_size=9, stride=2, padding=4, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.embedding = nn.Sequential(
            nn.LayerNorm(token_dim * 2),
            nn.Linear(token_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        curvature_hidden = max(32, hidden_dim // 4)
        residual_hidden = max(64, hidden_dim // 2)
        self.curvature_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, curvature_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(curvature_hidden, 1),
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, residual_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(residual_hidden, 1),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(hidden_dim + 2),
            nn.Linear(hidden_dim + 2, residual_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(residual_hidden, 1),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        gate_init = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, math.log(gate_init / (1.0 - gate_init)))

    def forward(self, exg: torch.Tensor, main_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.encoder(exg.float())
        pooled = torch.cat([tokens.mean(dim=-1), tokens.amax(dim=-1)], dim=-1)
        embedding = self.embedding(pooled)
        curvature_logit = self.curvature_head(embedding)
        reliability = torch.sigmoid(curvature_logit)

        main_prob = torch.sigmoid(main_logits.detach().float())
        uncertainty = 1.0 - (main_prob - 0.5).abs() * 2.0
        gate_input = torch.cat([embedding, reliability, uncertainty], dim=-1)
        gate = torch.sigmoid(self.gate_head(gate_input)) * self.gate_max
        delta = torch.tanh(self.delta_head(embedding)) * self.delta_scale
        curvature = F.softplus(curvature_logit) + 1e-4
        return delta, gate, curvature


def _parse_lora_targets(targets: str | tuple[str, ...] | list[str]) -> set[str]:
    if isinstance(targets, str):
        raw = targets.replace(";", ",").replace("+", ",").split(",")
    else:
        raw = list(targets)
    values = {str(item).strip().lower() for item in raw if str(item).strip()}
    if "attn" in values:
        values.update({"qkv", "proj"})
    if "mlp" in values:
        values.update({"fc1", "fc2"})
    if "all" in values:
        values.update({"qkv", "proj", "fc1", "fc2"})
    return values


def _inject_lora_adapters(
    encoder: STEEGFormerEncoder,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    targets,
    last_n: int = 0,
) -> int:
    target_set = _parse_lora_targets(targets)
    count = 0
    blocks = list(encoder.blocks)
    last_n = int(last_n or 0)
    if last_n > 0:
        blocks = blocks[-min(last_n, len(blocks)) :]
    for block in blocks:
        replacements: list[tuple[nn.Module, str]] = []
        if "qkv" in target_set:
            replacements.append((block.attn, "qkv"))
        if "proj" in target_set:
            replacements.append((block.attn, "proj"))
        if "fc1" in target_set:
            replacements.append((block.mlp, "fc1"))
        if "fc2" in target_set:
            replacements.append((block.mlp, "fc2"))
        for parent, name in replacements:
            layer = getattr(parent, name)
            if not isinstance(layer, nn.Linear):
                raise TypeError(f"Cannot inject LoRA into {parent.__class__.__name__}.{name}: {type(layer)!r}")
            setattr(parent, name, LoRALinear(layer, rank=rank, alpha=alpha, dropout=dropout))
            count += 1
    return count


def _filter_compatible_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    current = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    dropped = []
    for key, value in state_dict.items():
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            filtered[key] = value
        else:
            dropped.append(key)
    if dropped:
        LOGGER.info("Dropped %d incompatible STEEGFormer checkpoint keys", len(dropped))
    return filtered


class Net(nn.Module):
    """NeuroMM wrapper for ST-EEGFormer.

    NeuroMM loader returns 23 EEG channels plus 3 derived ECG/EMG channels.
    The `seed23` preset uses only the first 23 channels by default.
    """

    supports_feature_output = True

    def __init__(
        self,
        num_classes: int = 1,
        input_channels: int = 26,
        use_channels: int = 23,
        channel_preset: str = "seed23",
        pretrained_path: str = "models/STEEGFormer/checkpoint-300-small.pth",
        variant: str = "small",
        resample_timesteps: int = 512,
        dropout: float = 0.3,
        drop_path_rate: float = 0.1,
        normalize: bool = True,
        input_channel_indices: tuple[int, ...] | list[int] | None = None,
        montage_coord_embedding_enabled: bool = False,
        montage_coord_hidden_dim: int = 64,
        montage_coord_scale: float = 0.10,
        channel_dropout_prob: float = 0.0,
        channel_dropout_max_fraction: float = 0.0,
        channel_dropout_min_keep: int = 1,
        feature_adapter_enabled: bool = False,
        feature_adapter_hidden_dim: int = 256,
        feature_adapter_scale: float = 1.0,
        feature_adapter_dropout: float = 0.1,
        token_mil_enabled: bool = False,
        token_mil_hidden_dim: int = 64,
        token_mil_dropout: float = 0.1,
        token_mil_topk: int = 16,
        token_mil_scale: float = 0.25,
        learnable_car_enabled: bool = False,
        learnable_car_channels: int = 23,
        learnable_car_init: float = 1.0,
        learnable_car_max: float = 1.5,
        learnable_car_mode: str = "channel",
        lora_enabled: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_targets: str | tuple[str, ...] | list[str] = "qkv,proj",
        lora_last_n: int = 0,
        exg_moce_enabled: bool = False,
        exg_moce_start_channel: int = 23,
        exg_moce_channels: int = 3,
        exg_moce_token_dim: int = 128,
        exg_moce_hidden_dim: int = 256,
        exg_moce_dropout: float = 0.08,
        exg_moce_gate_max: float = 0.12,
        exg_moce_gate_init: float = 0.25,
        exg_moce_delta_scale: float = 1.0,
        exg_moce_normalize: bool = True,
        rhop_enabled: bool = False,
        rhop_token_dim: int = 32,
        rhop_dropout: float = 0.10,
        rhop_scale: float = 0.50,
        rhop_eps: float = 1e-4,
    ) -> None:
        super().__init__()
        variant_key = variant.lower()
        if variant_key not in VARIANT_CONFIGS:
            raise ValueError(f"Unsupported STEEGFormer variant: {variant!r}")
        if channel_preset not in CHANNEL_PRESETS:
            raise ValueError(f"Unsupported STEEGFormer channel_preset: {channel_preset}")

        self.input_channels = int(input_channels)
        self.use_channels = int(use_channels)
        self.resample_timesteps = int(resample_timesteps)
        self.normalize = bool(normalize)
        self.variant = variant_key
        self.feature_adapter_enabled = bool(feature_adapter_enabled)
        self.feature_adapter_scale = float(feature_adapter_scale)
        self.token_mil_enabled = bool(token_mil_enabled)
        self.token_mil_topk = int(token_mil_topk)
        self.token_mil_scale = float(token_mil_scale)
        self.learnable_car_enabled = bool(learnable_car_enabled)
        self.learnable_car_channels = int(learnable_car_channels)
        self.learnable_car_max = float(learnable_car_max)
        self.learnable_car_mode = str(learnable_car_mode).lower()
        self.lora_enabled = bool(lora_enabled)
        self.exg_moce_enabled = bool(exg_moce_enabled)
        self.exg_moce_start_channel = int(exg_moce_start_channel)
        self.exg_moce_channels = int(exg_moce_channels)
        self.exg_moce_normalize = bool(exg_moce_normalize)
        self.rhop_enabled = bool(rhop_enabled)
        self.rhop_scale = float(rhop_scale)
        self.montage_coord_embedding_enabled = bool(montage_coord_embedding_enabled)
        self.montage_coord_scale = float(montage_coord_scale)
        self.channel_dropout_prob = float(channel_dropout_prob)
        self.channel_dropout_max_fraction = float(channel_dropout_max_fraction)
        self.channel_dropout_min_keep = int(channel_dropout_min_keep)
        variant_config = VARIANT_CONFIGS[variant_key]
        self.feature_dim = int(variant_config["embed_dim"])

        channel_indices = torch.tensor(CHANNEL_PRESETS[channel_preset][: self.use_channels], dtype=torch.long)
        if channel_indices.numel() != self.use_channels:
            raise ValueError(f"channel_preset={channel_preset} has {channel_indices.numel()} indices, need {self.use_channels}")
        self.register_buffer("channel_indices", channel_indices, persistent=False)
        montage_coordinates = _montage_coordinate_tensor(channel_preset, self.use_channels)
        self.register_buffer("montage_coordinates", montage_coordinates, persistent=False)
        self.register_buffer(
            "montage_coord_features",
            _montage_sinusoidal_feature_tensor(montage_coordinates, self.feature_dim),
            persistent=False,
        )
        if input_channel_indices is None:
            selected_input_indices = torch.empty(0, dtype=torch.long)
        else:
            selected_input_indices = torch.tensor(tuple(int(index) for index in input_channel_indices), dtype=torch.long)
            if selected_input_indices.numel() != self.use_channels:
                raise ValueError(
                    f"input_channel_indices has {selected_input_indices.numel()} indices, need {self.use_channels}"
                )
        self.register_buffer("input_channel_indices", selected_input_indices, persistent=False)
        if self.learnable_car_enabled:
            car_channels = max(1, min(self.learnable_car_channels, self.use_channels))
            self.learnable_car_channels = car_channels
            if self.learnable_car_max <= 0:
                raise ValueError("learnable_car_max must be positive")
            init_alpha = max(1e-4, min(float(learnable_car_init), self.learnable_car_max - 1e-4))
            init_prob = init_alpha / self.learnable_car_max
            init_logit = math.log(init_prob / (1.0 - init_prob))
            if self.learnable_car_mode == "scalar":
                gate_shape = (1, 1, 1)
            elif self.learnable_car_mode == "channel":
                gate_shape = (1, car_channels, 1)
            else:
                raise ValueError(f"Unsupported learnable_car_mode={learnable_car_mode!r}")
            self.learnable_car_logit = nn.Parameter(torch.full(gate_shape, float(init_logit)))
        else:
            self.register_parameter("learnable_car_logit", None)

        self.encoder = STEEGFormerEncoder(
            embed_dim=self.feature_dim,
            depth=int(variant_config["depth"]),
            num_heads=int(variant_config["num_heads"]),
            drop_path_rate=drop_path_rate,
        )
        self.montage_coord_norm = nn.Identity()
        self.montage_coord_adapter = None
        if self.feature_adapter_enabled:
            hidden_dim = max(1, int(feature_adapter_hidden_dim))
            self.feature_adapter_norm = nn.LayerNorm(self.feature_dim)
            self.feature_adapter = nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(float(feature_adapter_dropout)),
                nn.Linear(hidden_dim, self.feature_dim),
            )
            nn.init.zeros_(self.feature_adapter[-1].weight)
            nn.init.zeros_(self.feature_adapter[-1].bias)
        else:
            self.feature_adapter_norm = nn.Identity()
            self.feature_adapter = nn.Identity()
        if self.token_mil_enabled:
            hidden_dim = max(1, int(token_mil_hidden_dim))
            self.token_mil_norm = nn.LayerNorm(self.feature_dim)
            self.token_mil_head = nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(float(token_mil_dropout)),
                nn.Linear(hidden_dim, num_classes),
            )
            nn.init.zeros_(self.token_mil_head[-1].weight)
            nn.init.zeros_(self.token_mil_head[-1].bias)
        else:
            self.token_mil_norm = nn.Identity()
            self.token_mil_head = nn.Identity()
        if self.rhop_enabled:
            self.rhop_pool = RiemannianHighOrderPooling(
                self.feature_dim,
                token_dim=int(rhop_token_dim),
                dropout=float(rhop_dropout),
                eps=float(rhop_eps),
            )
            rhop_fusion_dim = self.feature_dim + int(self.rhop_pool.output_dim)
            self.rhop_fusion_norm = nn.LayerNorm(rhop_fusion_dim)
            self.rhop_fusion = nn.Sequential(
                nn.Linear(rhop_fusion_dim, self.feature_dim),
                nn.GELU(),
                nn.Dropout(float(rhop_dropout)),
                nn.Linear(self.feature_dim, self.feature_dim),
            )
            nn.init.zeros_(self.rhop_fusion[-1].weight)
            nn.init.zeros_(self.rhop_fusion[-1].bias)
        else:
            self.rhop_pool = None
            self.rhop_fusion_norm = nn.Identity()
            self.rhop_fusion = nn.Identity()
        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.feature_dim, num_classes)
        if self.exg_moce_enabled:
            self.exg_moce = EXGMoCELiteExpert(
                in_channels=max(1, self.exg_moce_channels),
                token_dim=int(exg_moce_token_dim),
                hidden_dim=int(exg_moce_hidden_dim),
                dropout=float(exg_moce_dropout),
                gate_max=float(exg_moce_gate_max),
                gate_init=float(exg_moce_gate_init),
                delta_scale=float(exg_moce_delta_scale),
            )
        else:
            self.exg_moce = None
        self._load_pretrained(pretrained_path)
        self.lora_adapter_count = 0
        if self.lora_enabled:
            self.lora_adapter_count = _inject_lora_adapters(
                self.encoder,
                rank=int(lora_rank),
                alpha=float(lora_alpha),
                dropout=float(lora_dropout),
                targets=lora_targets,
                last_n=int(lora_last_n),
            )
            LOGGER.info(
                "Injected STEEGFormer LoRA adapters count=%d rank=%s alpha=%s targets=%s last_n=%s",
                self.lora_adapter_count,
                lora_rank,
                lora_alpha,
                lora_targets,
                lora_last_n,
            )

    def _load_pretrained(self, pretrained_path: str) -> None:
        if not pretrained_path:
            LOGGER.info("STEEGFormer pretrained_path is empty; skipping preload")
            return
        path = Path(pretrained_path)
        if not path.exists() or path.is_dir():
            LOGGER.warning("STEEGFormer checkpoint not found: %s", path)
            return
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        filtered = _filter_compatible_state_dict(self.encoder, state_dict)
        missing, unexpected = self.encoder.load_state_dict(filtered, strict=False)
        LOGGER.info(
            "Loaded STEEGFormer checkpoint %s compatible_keys=%d missing=%d unexpected=%d",
            path,
            len(filtered),
            len(missing),
            len(unexpected),
        )

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)
        if x.dim() != 3:
            raise ValueError(f"Expected STEEGFormer input [B,C,T] or [B,1,C,T], got {tuple(x.shape)}")
        if self.input_channel_indices.numel() > 0:
            x = x.index_select(1, self.input_channel_indices.to(x.device))
        else:
            x = x[:, : self.use_channels, :]
        x = self._apply_learnable_car(x.float())
        if self.resample_timesteps > 0 and x.size(-1) != self.resample_timesteps:
            x = F.interpolate(x, size=self.resample_timesteps, mode="linear", align_corners=False)
        else:
            x = x
        if self.normalize:
            center = x.mean(dim=-1, keepdim=True)
            scale = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
            x = (x - center) / scale
        return self._apply_channel_dropout(x)

    def _prepare_exg_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)
        if x.dim() != 3:
            raise ValueError(f"Expected EXG source input [B,C,T] or [B,1,C,T], got {tuple(x.shape)}")
        batch_size, channels, timesteps = x.shape
        start = max(0, min(self.exg_moce_start_channel, channels))
        end = max(start, min(start + self.exg_moce_channels, channels))
        if end > start:
            exg = x[:, start:end, :].float()
        else:
            exg = x.new_zeros((batch_size, 0, timesteps), dtype=torch.float32)
        if exg.size(1) < self.exg_moce_channels:
            pad = x.new_zeros((batch_size, self.exg_moce_channels - exg.size(1), timesteps), dtype=torch.float32)
            exg = torch.cat([exg, pad], dim=1)
        elif exg.size(1) > self.exg_moce_channels:
            exg = exg[:, : self.exg_moce_channels, :]
        if self.resample_timesteps > 0 and exg.size(-1) != self.resample_timesteps:
            exg = F.interpolate(exg, size=self.resample_timesteps, mode="linear", align_corners=False)
        if self.exg_moce_normalize:
            center = exg.mean(dim=-1, keepdim=True)
            scale = exg.std(dim=-1, keepdim=True).clamp_min(1e-6)
            exg = (exg - center) / scale
        return exg

    def _apply_learnable_car(self, x: torch.Tensor) -> torch.Tensor:
        if not self.learnable_car_enabled or self.learnable_car_logit is None:
            return x
        car_channels = max(1, min(self.learnable_car_channels, x.size(1)))
        reference = x[:, :car_channels, :].mean(dim=1, keepdim=True)
        alpha = torch.sigmoid(self.learnable_car_logit.to(dtype=x.dtype, device=x.device)) * self.learnable_car_max
        out = x.clone()
        out[:, :car_channels, :] = out[:, :car_channels, :] - alpha * reference
        return out

    def _apply_channel_dropout(self, x: torch.Tensor) -> torch.Tensor:
        if (
            not self.training
            or self.channel_dropout_prob <= 0.0
            or self.channel_dropout_max_fraction <= 0.0
            or x.size(1) <= self.channel_dropout_min_keep
        ):
            return x
        max_drop = int(round(float(x.size(1)) * self.channel_dropout_max_fraction))
        max_drop = max(1, min(max_drop, x.size(1) - max(1, self.channel_dropout_min_keep)))
        if max_drop <= 0:
            return x
        out = x.clone()
        active = torch.rand(out.size(0), device=out.device) < self.channel_dropout_prob
        for batch_index in active.nonzero(as_tuple=False).flatten().tolist():
            drop_count = int(torch.randint(1, max_drop + 1, (1,), device=out.device).item())
            drop_idx = torch.randperm(out.size(1), device=out.device)[:drop_count]
            out[batch_index, drop_idx, :] = 0.0
        return out

    def _channel_extra_embedding(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if not self.montage_coord_embedding_enabled:
            return None
        if self.montage_coord_adapter is not None:
            coords = self.montage_coordinates.to(device=device, dtype=torch.float32)
            delta = self.montage_coord_adapter(self.montage_coord_norm(coords))
            return (self.montage_coord_scale * delta).to(dtype=dtype)
        return (self.montage_coord_scale * self.montage_coord_features.to(device=device, dtype=dtype)).contiguous()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        channel_extra = self._channel_extra_embedding(x.device, x.dtype)
        if self.rhop_enabled:
            tokens = self.encoder.forward_tokens(x, self.channel_indices, channel_extra_embedding=channel_extra)
            features = self._adapt_features(tokens[:, 0])
            return self._apply_rhop(features, tokens)
        features = self.encoder(x, self.channel_indices, channel_extra_embedding=channel_extra)
        return self._adapt_features(features)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        channel_extra = self._channel_extra_embedding(x.device, x.dtype)
        return self.encoder.forward_tokens(x, self.channel_indices, channel_extra_embedding=channel_extra)

    def _adapt_features(self, features: torch.Tensor) -> torch.Tensor:
        if not self.feature_adapter_enabled:
            return features
        delta = self.feature_adapter(self.feature_adapter_norm(features))
        return features + self.feature_adapter_scale * delta

    def _apply_rhop(self, features: torch.Tensor, tokens: torch.Tensor | None) -> torch.Tensor:
        if not self.rhop_enabled or self.rhop_pool is None or tokens is None:
            return features
        patch_tokens = tokens[:, 1:]
        if int(patch_tokens.shape[1]) <= 0:
            return features
        descriptor = self.rhop_pool(patch_tokens)
        fused = torch.cat([features, descriptor.to(dtype=features.dtype)], dim=-1)
        delta = self.rhop_fusion(self.rhop_fusion_norm(fused))
        return features + self.rhop_scale * delta

    def _token_mil_logits(self, tokens: torch.Tensor) -> torch.Tensor | None:
        if not self.token_mil_enabled:
            return None
        patch_tokens = tokens[:, 1:]
        if patch_tokens.numel() == 0:
            return None
        token_logits = self.token_mil_head(self.token_mil_norm(patch_tokens)).squeeze(-1)
        token_count = token_logits.size(1)
        topk = self.token_mil_topk
        if topk <= 0 or topk >= token_count:
            pooled = token_logits.mean(dim=1, keepdim=True)
        else:
            pooled = token_logits.topk(topk, dim=1).values.mean(dim=1, keepdim=True)
        return pooled

    def forward(self, x: torch.Tensor, return_features: bool = False, return_tokens: bool = False):
        need_tokens = return_tokens or self.token_mil_enabled or self.rhop_enabled
        prepared = self._prepare_input(x)
        channel_extra = self._channel_extra_embedding(prepared.device, prepared.dtype)
        tokens = (
            self.encoder.forward_tokens(prepared, self.channel_indices, channel_extra_embedding=channel_extra)
            if need_tokens
            else None
        )
        features = (
            self._adapt_features(tokens[:, 0])
            if tokens is not None
            else self._adapt_features(self.encoder(prepared, self.channel_indices, channel_extra_embedding=channel_extra))
        )
        if self.rhop_enabled:
            features = self._apply_rhop(features, tokens)
        logits = self.head(self.head_drop(features))
        if self.token_mil_enabled and tokens is not None:
            token_logits = self._token_mil_logits(tokens)
            if token_logits is not None:
                logits = logits + self.token_mil_scale * token_logits

        exg_info = None
        if self.exg_moce_enabled and self.exg_moce is not None:
            exg = self._prepare_exg_input(x)
            exg_delta, exg_gate, exg_curvature = self.exg_moce(exg, logits)
            logits = logits + exg_gate.to(dtype=logits.dtype) * exg_delta.to(dtype=logits.dtype)
            exg_info = {
                "exg_delta": exg_delta,
                "exg_gate": exg_gate,
                "exg_curvature": exg_curvature,
            }

        if return_tokens:
            output = {"logits": logits, "features": features, "tokens": tokens}
            if exg_info is not None:
                output.update(exg_info)
            if not return_features:
                output.pop("features")
            return output
        if return_features:
            output = {"logits": logits, "features": features}
            if exg_info is not None:
                output.update(exg_info)
            return output
        return logits

    def finetune_parameter_groups(self, *, base_lr: float, weight_decay: float, config: Mapping[str, object]):
        def _scale(name: str, default: float) -> float:
            value = config.get(name, default)
            return default if value is None else float(value)

        head_lr = float(base_lr) * _scale("head_lr_scale", 1.0)
        adapter_lr_scale = _scale("adapter_lr_scale", 0.3)
        feature_adapter_lr = float(base_lr) * _scale("feature_adapter_lr_scale", adapter_lr_scale)
        token_mil_lr = float(base_lr) * _scale("token_mil_lr_scale", 0.5)
        exg_moce_lr = float(base_lr) * _scale("exg_moce_lr_scale", 1.0)
        rhop_lr = float(base_lr) * _scale("rhop_lr_scale", 1.0)
        car_lr = float(base_lr) * _scale("learnable_car_lr_scale", adapter_lr_scale)
        stem_adapter_lr = float(base_lr) * _scale("stem_adapter_lr_scale", adapter_lr_scale)
        lora_lr = float(base_lr) * _scale("lora_lr_scale", adapter_lr_scale)
        backbone_lr = float(base_lr) * _scale("backbone_lr_scale", 0.05)
        no_decay_norm_bias = bool(config.get("no_decay_norm_bias", True))

        buckets: dict[tuple[str, float, float], list[nn.Parameter]] = {}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("head."):
                group_name = "head"
                lr = head_lr
            elif ".lora_" in name or name.startswith("lora_"):
                group_name = "lora"
                lr = lora_lr
            elif name.startswith("token_mil"):
                group_name = "token_mil"
                lr = token_mil_lr
            elif name.startswith("learnable_car"):
                group_name = "learnable_car"
                lr = car_lr
            elif name.startswith("exg_moce"):
                group_name = "exg_moce"
                lr = exg_moce_lr
            elif name.startswith("rhop_"):
                group_name = "rhop"
                lr = rhop_lr
            elif name.startswith("feature_adapter"):
                group_name = "feature_adapter"
                lr = feature_adapter_lr
            elif name.startswith("montage_coord_"):
                group_name = "stem_adapter"
                lr = stem_adapter_lr
            elif (
                name == "encoder.cls_token"
                or name.startswith("encoder.patch_embed.")
                or name.startswith("encoder.enc_channel_emd.")
                or name.startswith("encoder.enc_temporal_emd.")
            ):
                group_name = "stem_adapter"
                lr = stem_adapter_lr
            else:
                group_name = "backbone"
                lr = backbone_lr

            if lr <= 0.0:
                parameter.requires_grad_(False)
                continue
            group_decay = 0.0
            if group_name != "lora" and not (
                no_decay_norm_bias
                and (parameter.ndim <= 1 or name.endswith(".bias") or ".norm" in name or "norm." in name)
            ):
                group_decay = float(weight_decay)
            buckets.setdefault((group_name, lr, group_decay), []).append(parameter)

        return [
            {"params": params, "lr": lr, "weight_decay": group_decay, "name": group_name}
            for (group_name, lr, group_decay), params in buckets.items()
            if params
        ]
