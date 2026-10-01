"""Task3 helpers: build legacy EEG models with a custom num_classes (e.g. 5).

This file does not modify the existing registry — it only re-exports the
existing MODEL_SPECS and provides one extra builder that overrides the
`num_classes` kwarg.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch.nn as nn

from .registry import MODEL_SPECS, _build_local_model, is_legacy_eeg_model_name


def build_legacy_eeg_model_with_num_classes(
    model_name: str,
    num_classes: int = 5,
    overrides: Mapping[str, object] | None = None,
) -> nn.Module:
    """Build a legacy EEG model that outputs (B, num_classes) logits."""
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unsupported legacy EEG model: {model_name}")
    module_name, kwargs = MODEL_SPECS[model_name]
    new_kwargs: dict = {**dict(kwargs), "num_classes": int(num_classes)}
    if overrides:
        new_kwargs.update(dict(overrides))
    return _build_local_model(module_name, new_kwargs)
