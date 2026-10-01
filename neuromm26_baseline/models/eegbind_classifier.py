"""EEG-anchored video classifier used by the released five-fold checkpoints."""
from __future__ import annotations
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F

class EEGBindClassifier(nn.Module):
    def __init__(self, *, video_feature_dims: dict[str, int], eeg_feature_dim: int,
                 num_classes: int = 5, video_embed_dim: int = 512,
                 classifier_hidden_dim: int = 768, dropout: float = 0.25,
                 temperature_init: float = 0.07, attentive_layers: int = 2,
                 attentive_heads: int = 8, attentive_ff_mult: float = 4.0,
                 attentive_max_segments: int = 16) -> None:
        super().__init__()
        if not video_feature_dims:
            raise ValueError("video_feature_dims must be non-empty")
        self.video_feature_names = list(video_feature_dims)
        self.video_feature_dims = {str(k): int(v) for k, v in video_feature_dims.items()}
        self.eeg_feature_dim = int(eeg_feature_dim)
        self.video_embed_dim = int(video_embed_dim)
        self.attentive_max_segments = int(attentive_max_segments)
        self.video_token_adapters = nn.ModuleDict({name: nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, video_embed_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(video_embed_dim, video_embed_dim), nn.GELU(), nn.LayerNorm(video_embed_dim),
        ) for name, dim in self.video_feature_dims.items()})
        self.video_stream_embedding = nn.Embedding(len(self.video_feature_names), video_embed_dim)
        self.video_segment_embedding = nn.Embedding(max(attentive_max_segments, 1), video_embed_dim)
        self.stream_type_embedding = nn.Embedding(2, video_embed_dim)
        if attentive_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=video_embed_dim, nhead=attentive_heads,
                dim_feedforward=max(int(round(video_embed_dim * attentive_ff_mult)), video_embed_dim),
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            self.video_token_encoder = nn.TransformerEncoder(layer, num_layers=attentive_layers)
        else:
            self.video_token_encoder = nn.Identity()
        self.video_query_attention = nn.MultiheadAttention(video_embed_dim, attentive_heads, dropout=dropout, batch_first=True)
        self.study_query_attention = nn.MultiheadAttention(video_embed_dim, attentive_heads, dropout=dropout, batch_first=True)
        self.eeg_token_projection = nn.Sequential(
            nn.LayerNorm(eeg_feature_dim), nn.Linear(eeg_feature_dim, video_embed_dim),
            nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(video_embed_dim),
        )
        self.study_to_eeg = nn.Sequential(
            nn.LayerNorm(video_embed_dim), nn.Linear(video_embed_dim, video_embed_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(video_embed_dim, eeg_feature_dim),
        )
        self.video_query = nn.Parameter(torch.empty(1, 1, video_embed_dim))
        self.study_query = nn.Parameter(torch.empty(1, 1, video_embed_dim))
        nn.init.trunc_normal_(self.video_query, std=0.02)
        nn.init.trunc_normal_(self.study_query, std=0.02)
        self.video_to_eeg = nn.Sequential(
            nn.LayerNorm(video_embed_dim), nn.Linear(video_embed_dim, video_embed_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(video_embed_dim, eeg_feature_dim),
        )
        hidden = max(classifier_hidden_dim // 2, 128)
        self.classifier = nn.Sequential(
            nn.LayerNorm(4 * eeg_feature_dim + 1),
            nn.Linear(4 * eeg_feature_dim + 1, classifier_hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / max(float(temperature_init), 1e-4)).log())

    def _video_feature_tokens(self, name: str, feature: torch.Tensor) -> torch.Tensor:
        feature = feature.float()
        dim = self.video_feature_dims[name]
        if feature.ndim == 2:
            tokens = feature.unsqueeze(1)
        elif feature.ndim == 3:
            tokens = feature
        elif int(feature.shape[-1]) == dim:
            tokens = feature.reshape(int(feature.shape[0]), -1, dim)
        else:
            tokens = feature.reshape(int(feature.shape[0]), 1, -1)
        if int(tokens.shape[-1]) != dim:
            raise ValueError(f"Video feature {name!r} expected dim {dim}, got {int(tokens.shape[-1])}")
        return self.video_token_adapters[name](tokens)

    def forward_with_aux(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        eeg_feature = batch["eeg_feature"].float()
        eeg_z = F.normalize(F.layer_norm(eeg_feature, (self.eeg_feature_dim,)), dim=1)
        batch_size = int(eeg_z.shape[0])
        token_parts, stream_ids = [], []
        for stream_index, name in enumerate(self.video_feature_names):
            tokens = self._video_feature_tokens(name, batch["video_features"][name])
            count = int(tokens.shape[1])
            segment_ids = torch.arange(count, device=tokens.device).clamp_max(self.attentive_max_segments - 1)
            stream_id = torch.full((count,), stream_index, device=tokens.device, dtype=torch.long)
            tokens = (tokens
                + self.video_stream_embedding(stream_id).view(1, count, self.video_embed_dim)
                + self.video_segment_embedding(segment_ids).view(1, count, self.video_embed_dim)
                + self.stream_type_embedding.weight[0].view(1, 1, self.video_embed_dim))
            token_parts.append(tokens)
            stream_ids.append(stream_id)
        video_tokens = torch.cat(token_parts, dim=1)
        stream_id_tensor = torch.cat(stream_ids, dim=0)
        padding_mask = torch.zeros(batch_size, int(video_tokens.shape[1]), dtype=torch.bool, device=video_tokens.device)
        if isinstance(self.video_token_encoder, nn.Identity):
            encoded = video_tokens
        else:
            encoded = self.video_token_encoder(video_tokens, src_key_padding_mask=padding_mask)
        eeg_token = self.eeg_token_projection(eeg_z).unsqueeze(1)
        eeg_token = eeg_token + self.stream_type_embedding.weight[1].view(1, 1, self.video_embed_dim)
        video_embed, attention = self.video_query_attention(
            self.video_query.expand(batch_size, -1, -1), encoded, encoded,
            key_padding_mask=padding_mask, need_weights=True, average_attn_weights=True,
        )
        video_embed = video_embed.squeeze(1)
        route_weights = attention.mean(dim=1).new_zeros(batch_size, len(self.video_feature_names))
        route_weights.scatter_add_(1, stream_id_tensor.view(1, -1).expand(batch_size, -1), attention.mean(dim=1))
        route_weights = route_weights / route_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        study_tokens = torch.cat([eeg_token, encoded], dim=1)
        study_padding = torch.cat([torch.zeros(batch_size, 1, dtype=torch.bool, device=video_tokens.device), padding_mask], dim=1)
        study_embed, _ = self.study_query_attention(
            self.study_query.expand(batch_size, -1, -1), study_tokens, study_tokens,
            key_padding_mask=study_padding, need_weights=False,
        )
        study_z = F.normalize(self.study_to_eeg(study_embed.squeeze(1)), dim=1)
        video_z = F.normalize(self.video_to_eeg(video_embed), dim=1)
        fused = torch.cat([eeg_z, study_z, eeg_z * study_z, torch.abs(eeg_z - study_z),
                           torch.sum(eeg_z * study_z, dim=1).view(-1, 1)], dim=1)
        return {"logits": self.classifier(fused),
                "pair_similarity": torch.sum(eeg_z * video_z, dim=1),
                "moe_weights": route_weights}

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.forward_with_aux(batch)["logits"]
