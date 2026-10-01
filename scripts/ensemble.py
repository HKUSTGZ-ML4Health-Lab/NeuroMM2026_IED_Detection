#!/usr/bin/env python3
"""Reference-free Task3 log-probability ensemble for official reproduction."""

from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np


def read_probs(path: Path, *, num_classes: int) -> tuple[list[str], np.ndarray]:
    sample_ids: list[str] = []
    probs: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", *(f"prob_{idx}" for idx in range(1, num_classes + 1))}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{path} must contain {sorted(required)}")
        for row in reader:
            sample_ids.append(str(row["sample_id"]))
            probs.append([float(row[f"prob_{idx}"]) for idx in range(1, num_classes + 1)])
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != num_classes or not np.all(np.isfinite(arr)):
        raise ValueError(f"bad probability matrix from {path}: {arr.shape}")
    return sample_ids, arr


def write_submission(path: Path, sample_ids: list[str], preds_one: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "prediction"])
        for sample_id, pred in zip(sample_ids, preds_one.tolist()):
            writer.writerow([sample_id, int(pred)])


def write_zip(csv_path: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="submission.csv")


def write_diagnostics(path: Path, sample_ids: list[str], preds_one: np.ndarray, probs: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    num_classes = int(probs.shape[1])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "prediction", "max_probability", *(f"prob_{idx}" for idx in range(1, num_classes + 1))])
        for idx, sample_id in enumerate(sample_ids):
            writer.writerow([sample_id, int(preds_one[idx]), float(probs[idx].max()), *[float(v) for v in probs[idx].tolist()]])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", nargs="+", required=True)
    parser.add_argument("--weights", nargs="*", type=float, default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-prefix", default="submission")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    diag_paths = [Path(p) for p in args.diagnostics]
    if args.weights:
        if len(args.weights) != len(diag_paths):
            raise ValueError("--weights length must match --diagnostics")
        weights = np.asarray(args.weights, dtype=np.float64)
    else:
        weights = np.ones(len(diag_paths), dtype=np.float64)
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("weights must be nonnegative and sum to > 0")
    weights = weights / weights.sum()

    sample_ids: list[str] | None = None
    logp_acc: np.ndarray | None = None
    eps = 1.0e-12
    for weight, path in zip(weights.tolist(), diag_paths):
        ids, probs = read_probs(path, num_classes=int(args.num_classes))
        if sample_ids is None:
            sample_ids = ids
            logp_acc = np.zeros_like(probs, dtype=np.float64)
        elif ids != sample_ids:
            raise ValueError(f"sample order mismatch in {path}")
        assert logp_acc is not None
        logp_acc += float(weight) * np.log(np.clip(probs, eps, 1.0))
    assert sample_ids is not None and logp_acc is not None

    logits = logp_acc - logp_acc.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), eps, math.inf)
    preds_one = probs.argmax(axis=1).astype(np.int64) + 1

    out_root = Path(args.output_root)
    paths = {
        "csv": out_root / "submissions" / f"{args.output_prefix}.csv",
        "zip": out_root / "submissions" / f"{args.output_prefix}.zip",
        "diagnostics": out_root / "submissions" / f"{args.output_prefix}.diagnostics.csv",
        "meta": out_root / "submissions" / f"{args.output_prefix}.meta.json",
    }
    if not args.overwrite:
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite: {existing[:3]}")

    write_submission(paths["csv"], sample_ids, preds_one)
    write_zip(paths["csv"], paths["zip"])
    write_diagnostics(paths["diagnostics"], sample_ids, preds_one, probs)

    counts = Counter(int(v) for v in preds_one.tolist())
    meta = {
        "diagnostics": [str(path) for path in diag_paths],
        "weights": [float(v) for v in weights.tolist()],
        "aggregation": "log_probability_average",
        "num_samples": len(sample_ids),
        "prediction_counts": {str(k): int(counts.get(k, 0)) for k in range(1, int(args.num_classes) + 1)},
        **{key: str(path) for key, path in paths.items()},
    }
    paths["meta"].write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote {paths['zip']}")
    print(json.dumps(meta["prediction_counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
