"""Read the official unlabeled candidate features and write submission files."""
from __future__ import annotations
import csv
import tempfile
import zipfile
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from neuromm26_baseline.datasets.collate_fn import neuromm_collate
from neuromm26_baseline.datasets.task3_datasets import Task3EEGFeatureDataset
from neuromm26_baseline.inference_support import _load_torch_checkpoint

class _PrintLogger:
    def info(self, message: str, *args: Any) -> None:
        print(message % args if args else message, flush=True)

    def warning(self, message: str, *args: Any) -> None:
        print(message % args if args else message, flush=True)


class CandidateFeatureDataset(Dataset):
    def __init__(
        self,
        *,
        sample_ids: list[str],
        eeg_cache: dict[str, Any],
        video_feature_root: Path,
        video_feature_names: list[str],
        video_cache: dict[str, torch.Tensor] | None = None,
    ) -> None:
        if list(eeg_cache["sample_ids"]) != sample_ids:
            raise ValueError("Cached EEG sample order does not match candidate ids")
        self.sample_ids = sample_ids
        self.eeg_features = eeg_cache["eeg_features"].float()
        self.video_feature_root = video_feature_root
        self.video_feature_names = video_feature_names
        self.video_cache = video_cache
        if self.video_cache is not None:
            missing = [name for name in self.video_feature_names if name not in self.video_cache]
            if missing:
                raise ValueError(f"Video cache is missing routes: {missing}")
            for name in self.video_feature_names:
                cache = self.video_cache[name]
                if int(cache.shape[0]) != len(self.sample_ids):
                    raise ValueError(
                        f"Video cache route {name} has {int(cache.shape[0])} rows, expected {len(self.sample_ids)}"
                    )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _load_video_tensor(self, name: str, sample_id: str) -> torch.Tensor:
        path = self.video_feature_root / name / f"{sample_id}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing video feature: {path}")
        return torch.from_numpy(np.load(path).copy()).float()

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_id = self.sample_ids[index]
        item = {
            "sample_id": sample_id,
            "eeg_feature": self.eeg_features[index],
            "video_features": {
                name: self.video_cache[name][index] if self.video_cache is not None else self._load_video_tensor(name, sample_id)
                for name in self.video_feature_names
            },
        }
        return item

    def to_device(self, device: torch.device) -> None:
        self.eeg_features = self.eeg_features.to(device, non_blocking=True)
        if self.video_cache is not None:
            self.video_cache = {
                name: cache.to(device, non_blocking=True)
                for name, cache in self.video_cache.items()
            }


def _read_candidate_ids(candidate_dir: Path) -> list[str]:
    path = candidate_dir / "candidate_ids.txt"
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"{path} is empty")
    return ids


def _build_candidate_manifest(sample_ids: list[str]) -> Path:
    handle = tempfile.NamedTemporaryFile(
        "w",
        suffix="_neuromm26_task3_clip_candidate_manifest.csv",
        delete=False,
        newline="",
    )
    with handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "split", "label", "label_type", "raw_video_relpath", "subject_id", "eeg_source_relpath"])
        for sample_id in sample_ids:
            writer.writerow([sample_id, "candidate", "", "", "", "", ""])
    return Path(handle.name)


def _move_eeg_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    moved["eeg"] = batch["eeg"].to(device, non_blocking=True)
    if isinstance(batch.get("label"), torch.Tensor):
        moved["label"] = batch["label"].to(device, non_blocking=True)
    return moved


def _extract_or_load_candidate_eeg_cache(
    *,
    eeg_model: torch.nn.Module,
    manifest_path: Path,
    candidate_dir: Path,
    sample_ids: list[str],
    cache_path: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    force: bool,
    logger: _PrintLogger,
    eeg_preprocess: str = "baseline26",
    eeg_normalization: str = "none",
) -> dict[str, Any]:
    if cache_path.exists() and not force:
        cache = _load_torch_checkpoint(cache_path)
        if list(cache.get("sample_ids", [])) == sample_ids:
            logger.info("loaded_eeg_cache=%s samples=%d", cache_path, len(sample_ids))
            return cache
        logger.warning("Ignoring stale EEG cache with mismatched sample order: %s", cache_path)

    dataset = Task3EEGFeatureDataset(
        manifest_csv=str(manifest_path),
        split="candidate",
        eeg_feature_root=str(candidate_dir / "eeg"),
        preload_in_memory=False,
        target_shape=(26, 2000),
        eeg_preprocess=eeg_preprocess,
        eeg_normalization=eeg_normalization,
    )
    if list(dataset.sample_ids) != sample_ids:
        raise ValueError("candidate EEG dataset sample order mismatch")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=neuromm_collate,
    )
    eeg_model.eval()
    all_features: list[torch.Tensor] = []
    all_ids: list[str] = []
    logger.info("extracting_eeg_cache=%s samples=%d", cache_path, len(dataset))
    for batch in loader:
        batch = _move_eeg_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            output = eeg_model(batch, return_features=True)
        if not isinstance(output, dict) or "features" not in output:
            raise RuntimeError("EEG anchor does not return features")
        all_features.append(output["features"].detach().float().cpu())
        all_ids.extend(batch["sample_id"])
    if all_ids != sample_ids:
        raise RuntimeError("EEG cache extraction changed sample order")
    cache = {
        "sample_ids": all_ids,
        "eeg_features": torch.cat(all_features),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    logger.info("wrote_eeg_cache=%s feature_shape=%s", cache_path, tuple(cache["eeg_features"].shape))
    return cache


def _write_submission(csv_path: Path, sample_ids: list[str], predictions: np.ndarray) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "prediction"])
        for sample_id, prediction in zip(sample_ids, predictions.tolist()):
            writer.writerow([sample_id, int(prediction)])


def _zip_submission(csv_path: Path, zip_path: Path, arcname: str) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname=arcname)


def _validate_submission(csv_path: Path, zip_path: Path, sample_ids: list[str], num_classes: int) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
    if names != ["submission.csv"]:
        raise ValueError(f"ZIP must contain exactly submission.csv, got {names}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["sample_id", "prediction"]:
            raise ValueError(f"Unexpected submission header: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != len(sample_ids):
        raise ValueError(f"Expected {len(sample_ids)} rows, got {len(rows)}")
    ids = [row["sample_id"] for row in rows]
    if ids != sample_ids:
        raise ValueError("Submission sample order does not match candidate_ids.txt")
    if len(set(ids)) != len(ids):
        raise ValueError("Submission contains duplicated sample_id")
    for row in rows:
        try:
            pred = int(row["prediction"])
        except ValueError as exc:
            raise ValueError(f"Non-numeric prediction for {row['sample_id']}: {row['prediction']}") from exc
        if pred < 1 or pred > num_classes:
            raise ValueError(f"Prediction outside 1..{num_classes}: {row['sample_id']}={pred}")
