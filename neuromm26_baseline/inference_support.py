"""Inference checkpoint and EEG anchor helpers."""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from neuromm26_baseline.models.legacy.registry_task3 import build_legacy_eeg_model_with_num_classes

def _safe_name(value: str) -> str:
    return value.replace("/", "-").replace(" ", "_")


def _checkpoint_hash(path: str | Path) -> str:
    return hashlib.sha1(str(Path(path)).encode("utf-8")).hexdigest()[:10]


def _load_torch_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict at {path}, got {type(checkpoint)!r}")
    return checkpoint


def _extract_model_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state = checkpoint["model"]
    else:
        state = checkpoint
    return dict(state)


def _load_eeg_anchor(model: nn.Module, checkpoint_path: str | Path, logger) -> None:
    checkpoint = _load_torch_checkpoint(checkpoint_path)
    state = _extract_model_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"EEG anchor checkpoint mismatch: missing={missing} unexpected={unexpected}"
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    logger.info("loaded_frozen_eeg_anchor=%s keys=%d", checkpoint_path, len(state))


def detect_video_feature_dims(video_feature_root: str, video_feature_names: list[str]) -> dict[str, int]:
    root = Path(video_feature_root)
    dims: dict[str, int] = {}
    for name in video_feature_names:
        feature_dir = root / name
        samples = sorted(feature_dir.glob("*.npy"))
        if not samples:
            raise FileNotFoundError(f"No .npy files under {feature_dir}")
        arr = np.load(samples[0], mmap_mode="r")
        dims[name] = int(arr.shape[0]) if arr.ndim == 1 else int(arr.shape[-1])
    return dims


def task3_moe_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated: dict[str, Any] = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if key in {"video_features", "real_video_features"}:
            names = list(values[0].keys())
            video_batch: dict[str, torch.Tensor] = {}
            for name in names:
                tensors = [item[name] for item in values]
                if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
                    raise RuntimeError(f"Missing non-tensor video feature for {name}")
                video_batch[name] = torch.stack(tensors)
            collated[key] = video_batch
        elif all(isinstance(value, torch.Tensor) for value in values):
            collated[key] = torch.stack(values)
        elif all(value is None for value in values):
            collated[key] = None
        else:
            collated[key] = values
    return collated


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: _move_to_device(value, device) for key, value in batch.items()}


def _build_eeg_anchor(model_name: str, num_classes: int) -> nn.Module:
    if model_name != "steegformer_large_seed23" or num_classes != 5:
        raise ValueError("Unsupported EEG anchor configuration")
    return build_legacy_eeg_model_with_num_classes(model_name, num_classes)
