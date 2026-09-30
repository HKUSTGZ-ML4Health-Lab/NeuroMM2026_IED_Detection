"""Registry for migrated EEG-only baseline variants stored in the legacy subpackage."""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping

import torch.nn as nn


class LegacyEEGModelAdapter(nn.Module):
    """Adapts EEG-only models that consume tensors into the current batch-based API."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.feature_dim = getattr(model, "feature_dim", getattr(model, "feat_dim", None))
        self.supports_feature_output = bool(getattr(model, "supports_feature_output", False))

    def forward(self, batch, return_features: bool = False, return_tokens: bool = False):
        if return_tokens:
            if not hasattr(self.model, "forward_tokens"):
                raise RuntimeError(f"{self.model.__class__.__name__} does not expose token output")
            return self.model(batch["eeg"], return_features=return_features, return_tokens=True)
        if return_features and self.supports_feature_output:
            return self.model(batch["eeg"], return_features=True)
        return self.model(batch["eeg"])

    def finetune_parameter_groups(self, *, base_lr: float, weight_decay: float, config: Mapping[str, object]):
        if hasattr(self.model, "finetune_parameter_groups"):
            return self.model.finetune_parameter_groups(
                base_lr=base_lr,
                weight_decay=weight_decay,
                config=config,
            )
        return [{"params": list(self.parameters()), "lr": float(base_lr), "weight_decay": float(weight_decay), "name": "all"}]


MODEL_SPECS: dict[str, tuple[str, Mapping[str, object]]] = {
    "steegformer_large_seed23": ("steegformer", {"num_classes": 1, "input_channels": 26, "use_channels": 23, "channel_preset": "seed23", "pretrained_path": "", "variant": "large", "resample_timesteps": 512, "dropout": 0.3, "drop_path_rate": 0.1, "normalize": True}),
}

LEGACY_EEG_MODEL_NAMES = tuple(sorted(MODEL_SPECS))


def _build_local_model(
    module_name: str,
    kwargs: Mapping[str, object],
    overrides: Mapping[str, object] | None = None,
) -> nn.Module:
    module = importlib.import_module(f"{__package__}.{module_name}")
    model_cls = getattr(module, "Net")
    resolved_kwargs = dict(kwargs)
    if overrides:
        resolved_kwargs.update(dict(overrides))
    if os.environ.get("NEUROMM_DISABLE_LEGACY_PRETRAINED") == "1":
        for key in tuple(resolved_kwargs):
            if key == "pretrained" or key.startswith("pretrained_"):
                resolved_kwargs[key] = False
    return LegacyEEGModelAdapter(model_cls(**resolved_kwargs))


def is_legacy_eeg_model_name(model_name: str | None) -> bool:
    return bool(model_name) and model_name in MODEL_SPECS


def build_legacy_eeg_model(model_name: str, overrides: Mapping[str, object] | None = None) -> nn.Module:
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unsupported legacy EEG model: {model_name}")
    module_name, kwargs = MODEL_SPECS[model_name]
    return _build_local_model(module_name, kwargs, overrides=overrides)
