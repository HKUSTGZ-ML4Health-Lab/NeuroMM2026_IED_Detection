#!/usr/bin/env python3
"""Predict all official unlabeled candidates with a released EEG-video fold."""
from __future__ import annotations
import argparse
import csv
import io
import json
import sys
import tarfile
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from neuromm26_baseline.models.eegbind_classifier import EEGBindClassifier
from neuromm26_baseline.inference_support import (
    _build_eeg_anchor, _checkpoint_hash, _load_eeg_anchor, _load_torch_checkpoint,
    _move_batch, _safe_name, detect_video_feature_dims, task3_moe_collate,
)
from scripts.candidate_data import (
    CandidateFeatureDataset, _PrintLogger, _build_candidate_manifest,
    _extract_or_load_candidate_eeg_cache, _read_candidate_ids,
    _validate_submission, _write_submission, _zip_submission,
)

def _make_model(checkpoint: dict[str, Any]) -> EEGBindClassifier:
    if checkpoint.get("architecture") != "task3_eegbind_direct_classifier_v1":
        raise ValueError("Unsupported checkpoint architecture")
    config = dict(checkpoint.get("model_config", {}))
    if config.get("fusion_arch") != "attentive_probe":
        raise ValueError("This release supports the attentive-probe model only")
    unsupported = ("region_similarity_features", "task1_scalar_dim", "video_ib_dim", "route_uncertainty",
                   "clara_adapter_rank", "aux_class_route_queries", "attentive_temporal_crop_count")
    if any(config.get(key) for key in unsupported):
        raise ValueError("Checkpoint enables an unsupported auxiliary branch")
    return EEGBindClassifier(
        video_feature_dims={str(k): int(v) for k, v in checkpoint["video_feature_dims"].items()},
        eeg_feature_dim=int(checkpoint["eeg_feature_dim"]),
        num_classes=int(checkpoint.get("num_classes", 5)),
        video_embed_dim=int(config["video_embed_dim"]),
        classifier_hidden_dim=int(config["classifier_hidden_dim"]),
        dropout=float(config["dropout"]),
        temperature_init=float(config["temperature_init"]),
        attentive_layers=int(config["attentive_layers"]),
        attentive_heads=int(config["attentive_heads"]),
        attentive_ff_mult=float(config["attentive_ff_mult"]),
        attentive_max_segments=int(config["attentive_max_segments"]),
    )

def _output_paths(args: argparse.Namespace, checkpoint_path: Path) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    if args.output_prefix:
        prefix = args.output_prefix
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"task3submission_eegbind_{checkpoint_path.parent.name}_{timestamp}"
    return {
        "csv": output_dir / f"{prefix}.csv",
        "zip": output_dir / f"{prefix}.zip",
        "meta": output_dir / f"{prefix}.meta.json",
        "diagnostics": output_dir / f"{prefix}.diagnostics.csv",
    }


def _refuse_overwrite(paths: dict[str, Path], *, write_diagnostics: bool, overwrite: bool) -> None:
    checked = [paths["csv"], paths["zip"], paths["meta"]]
    if write_diagnostics:
        checked.append(paths["diagnostics"])
    for path in checked:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {path}. Use --overwrite or choose another prefix.")


def _validate_video_cache(
    cache: dict[str, Any],
    *,
    sample_ids: list[str],
    video_feature_names: list[str],
    cache_path: Path,
) -> dict[str, torch.Tensor]:
    if list(cache.get("sample_ids", [])) != sample_ids:
        raise ValueError(f"Video cache sample order mismatch: {cache_path}")
    cached_names = [str(name) for name in cache.get("video_feature_names", [])]
    if cached_names and cached_names != video_feature_names:
        raise ValueError(f"Video cache route order mismatch: {cache_path} has {cached_names}, expected {video_feature_names}")
    raw_features = cache.get("video_features")
    if not isinstance(raw_features, dict):
        raise TypeError(f"Video cache missing video_features dict: {cache_path}")
    video_cache: dict[str, torch.Tensor] = {}
    for name in video_feature_names:
        if name not in raw_features:
            raise ValueError(f"Video cache {cache_path} is missing route {name}")
        tensor = raw_features[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Video cache route {name} is not a tensor: {type(tensor)!r}")
        if int(tensor.shape[0]) != len(sample_ids):
            raise ValueError(f"Video cache route {name} rows={int(tensor.shape[0])}, expected {len(sample_ids)}")
        video_cache[name] = tensor.float().contiguous()
    return video_cache


def _load_video_route_from_tar(
    *,
    tar_path: Path,
    sample_ids: list[str],
    logger: _PrintLogger,
) -> torch.Tensor:
    id_to_pos = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    route_tensors: list[torch.Tensor | None] = [None] * len(sample_ids)
    matched = 0
    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".npy"):
                continue
            sample_id = Path(member.name).stem
            pos = id_to_pos.get(sample_id)
            if pos is None:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                raise FileNotFoundError(f"Could not read {member.name} from {tar_path}")
            array = np.load(io.BytesIO(handle.read()))
            route_tensors[pos] = torch.from_numpy(np.asarray(array, dtype=np.float32).copy())
            matched += 1
    missing = [sample_id for sample_id, tensor in zip(sample_ids, route_tensors) if tensor is None]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"{tar_path} missing {len(missing)} candidate video features; first: {preview}")
    stacked = torch.stack([tensor for tensor in route_tensors if tensor is not None]).contiguous()
    logger.info("cached_video_route_from_tar=%s shape=%s matched=%d", tar_path.name, tuple(stacked.shape), matched)
    return stacked


def _load_video_route_from_files(
    *,
    video_feature_root: Path,
    name: str,
    sample_ids: list[str],
) -> torch.Tensor:
    route_tensors: list[torch.Tensor] = []
    for sample_id in sample_ids:
        path = video_feature_root / name / f"{sample_id}.npy"
        try:
            array = np.load(path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Missing video feature: {path}") from exc
        route_tensors.append(torch.from_numpy(np.asarray(array, dtype=np.float32).copy()))
    return torch.stack(route_tensors).contiguous()


def _load_or_build_candidate_video_cache(
    *,
    sample_ids: list[str],
    video_feature_root: Path,
    video_feature_names: list[str],
    cache_path: Path | None,
    force: bool,
    logger: _PrintLogger,
) -> dict[str, torch.Tensor] | None:
    if cache_path is None:
        return None
    if cache_path.exists() and not force:
        cache = _load_torch_checkpoint(cache_path)
        video_cache = _validate_video_cache(
            cache,
            sample_ids=sample_ids,
            video_feature_names=video_feature_names,
            cache_path=cache_path,
        )
        logger.info("loaded_video_cache=%s routes=%d samples=%d", cache_path, len(video_feature_names), len(sample_ids))
        return video_cache

    video_cache: dict[str, torch.Tensor] = {}
    archive_root = video_feature_root.parent / "archives"
    logger.info("building_video_cache=%s routes=%d samples=%d", cache_path, len(video_feature_names), len(sample_ids))
    for name in video_feature_names:
        tar_path = archive_root / f"video_{name}.tar"
        if tar_path.exists():
            video_cache[name] = _load_video_route_from_tar(tar_path=tar_path, sample_ids=sample_ids, logger=logger)
        else:
            logger.info("loading_video_route_from_files=%s", name)
            video_cache[name] = _load_video_route_from_files(
                video_feature_root=video_feature_root,
                name=name,
                sample_ids=sample_ids,
            )
            logger.info("cached_video_route=%s shape=%s", name, tuple(video_cache[name].shape))

    payload = {
        "sample_ids": list(sample_ids),
        "video_feature_names": list(video_feature_names),
        "video_features": video_cache,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    logger.info("wrote_video_cache=%s", cache_path)
    return video_cache


def _write_diagnostics(path: Path, result: dict[str, Any], video_names: list[str], num_classes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["sample_id", "prediction", "max_probability", "alignment_similarity"]
        header.extend(f"moe_{name}" for name in video_names)
        header.extend(f"prob_{idx}" for idx in range(1, num_classes + 1))
        writer.writerow(header)
        for row_idx, sample_id in enumerate(result["sample_ids"]):
            row = [
                sample_id,
                int(result["predictions_one_based"][row_idx]),
                float(result["max_probability"][row_idx]),
                float(result["alignment_similarity"][row_idx]),
            ]
            row.extend(float(v) for v in result["moe_weights"][row_idx].tolist())
            row.extend(float(v) for v in result["probabilities"][row_idx].tolist())
            writer.writerow(row)


@torch.no_grad()
def _predict(model: EEGBindClassifier, loader: DataLoader, device: torch.device, *, num_classes: int) -> dict[str, Any]:
    model.eval()
    sample_ids, logits_list, sim_list, route_list = [], [], [], []
    for batch in loader:
        batch = _move_batch(batch, device)
        out = model.forward_with_aux(batch)
        sample_ids.extend(batch["sample_id"])
        logits_list.append(out["logits"].float().view(-1, num_classes).cpu())
        sim_list.append(out["pair_similarity"].float().cpu())
        route_list.append(out["moe_weights"].float().cpu())
    logits = torch.cat(logits_list)
    probs = torch.softmax(logits, dim=1)
    return {
        "sample_ids": sample_ids, "logits": logits.numpy(), "probabilities": probs.numpy(),
        "predictions_one_based": probs.argmax(dim=1).numpy() + 1,
        "max_probability": probs.max(dim=1).values.numpy(),
        "alignment_similarity": torch.cat(sim_list).numpy(),
        "moe_weights": torch.cat(route_list).numpy(),
    }

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eeg-anchor-checkpoint", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-enabled", action="store_true")
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--cache-on-device", action="store_true")
    parser.add_argument("--force-eeg-cache", action="store_true")
    parser.add_argument("--eeg-cache-root", default=None)
    parser.add_argument("--video-cache-path", default=None)
    parser.add_argument("--force-video-cache", action="store_true")
    parser.add_argument("--write-diagnostics", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    paths = _output_paths(args, checkpoint_path)
    _refuse_overwrite(paths, write_diagnostics=args.write_diagnostics, overwrite=args.overwrite)
    candidate_dir = Path(args.candidate_dir)
    sample_ids = _read_candidate_ids(candidate_dir)
    if len(sample_ids) != 20000:
        raise ValueError(f"expected 20000 candidate ids, got {len(sample_ids)}")
    manifest_path = _build_candidate_manifest(sample_ids)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    amp_enabled = args.amp_enabled and device.type == "cuda"
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    checkpoint = _load_torch_checkpoint(checkpoint_path)
    config = dict(checkpoint.get("model_config", {}))
    video_names = [str(name) for name in checkpoint.get("video_feature_names", [])]
    if not video_names:
        raise ValueError("Checkpoint lacks video feature names")
    if not checkpoint.get("video_feature_dims"):
        checkpoint["video_feature_dims"] = detect_video_feature_dims(str(candidate_dir / "video"), video_names)
    preprocess = str(checkpoint.get("task3_eeg_preprocess") or config.get("task3_eeg_preprocess", "baseline26"))
    normalization = str(checkpoint.get("task3_eeg_normalization") or config.get("task3_eeg_normalization", "none"))
    if preprocess != "baseline26" or normalization not in {"none", ""}:
        raise ValueError("Unsupported EEG preprocessing configuration")
    if checkpoint.get("eeg_backbone_montage_config") or config.get("eeg_backbone_montage_config"):
        raise ValueError("Montage adaptation is not part of this release")
    eeg_model_name = str(checkpoint.get("eeg_model", "steegformer_large_seed23"))
    num_classes = int(checkpoint.get("num_classes", 5))
    logger = _PrintLogger()
    eeg_model = _build_eeg_anchor(eeg_model_name, num_classes).to(device)
    _load_eeg_anchor(eeg_model, args.eeg_anchor_checkpoint, logger=logger)
    cache_dir = Path(args.eeg_cache_root) if args.eeg_cache_root else Path(args.output_dir).parent / "eeg_cache"
    cache_path = cache_dir / f"{_safe_name(eeg_model_name)}__candidate__ckpt{_checkpoint_hash(args.eeg_anchor_checkpoint)}.pt"
    eeg_cache = _extract_or_load_candidate_eeg_cache(
        eeg_model=eeg_model, manifest_path=manifest_path, candidate_dir=candidate_dir,
        sample_ids=sample_ids, cache_path=cache_path, device=device, batch_size=args.batch_size,
        num_workers=args.num_workers, amp_enabled=amp_enabled, amp_dtype=amp_dtype,
        force=args.force_eeg_cache, logger=logger, eeg_preprocess=preprocess, eeg_normalization=normalization,
    )
    del eeg_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    video_cache = _load_or_build_candidate_video_cache(
        sample_ids=sample_ids, video_feature_root=candidate_dir / "video", video_feature_names=video_names,
        cache_path=Path(args.video_cache_path) if args.video_cache_path else None,
        force=args.force_video_cache, logger=logger,
    )
    dataset = CandidateFeatureDataset(
        sample_ids=sample_ids, eeg_cache=eeg_cache, video_feature_root=candidate_dir / "video",
        video_feature_names=video_names, video_cache=video_cache,
    )
    num_workers = args.num_workers
    if args.cache_on_device and device.type == "cuda":
        dataset.to_device(device)
        num_workers = 0
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda" and not args.cache_on_device,
        persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=task3_moe_collate,
    )
    model = _make_model(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    result = _predict(model, loader, device, num_classes=num_classes)
    if result["sample_ids"] != sample_ids:
        raise ValueError("Candidate sample order mismatch")
    predictions = result["predictions_one_based"].astype(np.int64)
    _write_submission(paths["csv"], sample_ids, predictions)
    _zip_submission(paths["csv"], paths["zip"], "submission.csv")
    _validate_submission(paths["csv"], paths["zip"], sample_ids, num_classes)
    if args.write_diagnostics:
        _write_diagnostics(paths["diagnostics"], result, video_names, num_classes)
    paths["meta"].write_text(json.dumps({"checkpoint": str(checkpoint_path), "num_samples": len(sample_ids)}, indent=2) + "\n")
    print(f"[ok] wrote {paths['zip']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
