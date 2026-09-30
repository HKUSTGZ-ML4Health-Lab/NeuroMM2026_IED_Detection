#!/usr/bin/env python3
"""Create subject-level 5-fold Task3 manifests.

Each output manifest keeps the original rows but rewrites ``split`` to
``train`` for subjects assigned outside the held-out fold and ``val`` for the
held-out subjects. The final five-fold models train with ``--no-val``
and ``TRAIN_SPLIT=train`` in the original training workspace. These manifests
provide the subject-disjoint training subsets.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        required = {"sample_id", "split", "label_type", "subject_id"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        return [dict(row) for row in reader]


def _subject_counts(rows: list[dict[str, str]]) -> dict[str, Counter[int]]:
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    for row in rows:
        counts[str(row["subject_id"])][int(float(row["label_type"]))] += 1
    return counts


def _assign_subject_folds(
    subject_counts: dict[str, Counter[int]],
    *,
    num_folds: int,
    seed: int,
) -> dict[str, int]:
    rng = random.Random(int(seed))
    subjects = list(subject_counts)
    rng.shuffle(subjects)
    # Large/rare-positive subjects first makes the greedy balance less brittle.
    subjects.sort(
        key=lambda sid: (
            -sum(count for label, count in subject_counts[sid].items() if label != 0),
            -sum(subject_counts[sid].values()),
            sid,
        )
    )
    fold_counts = [Counter() for _ in range(num_folds)]
    fold_subjects = [0 for _ in range(num_folds)]
    total = Counter()
    for counts in subject_counts.values():
        total.update(counts)
    target = {label: total[label] / float(num_folds) for label in range(6)}
    target_rows = sum(total.values()) / float(num_folds)
    target_subjects = len(subjects) / float(num_folds)

    def global_score() -> float:
        score = 0.0
        for fold in range(num_folds):
            rows = sum(fold_counts[fold].values())
            score += 8.0 * ((rows - target_rows) / max(target_rows, 1.0)) ** 2
            score += 2.0 * ((fold_subjects[fold] - target_subjects) / max(target_subjects, 1.0)) ** 2
            for label in range(6):
                weight = 3.0 if label != 0 else 0.25
                score += weight * ((fold_counts[fold][label] - target[label]) / max(target[label], 1.0)) ** 2
        return score

    assignment: dict[str, int] = {}
    # Seed one subject per fold to avoid the empty-fold failure mode when rare
    # subtypes are highly concentrated by subject.
    for fold, sid in enumerate(subjects[:num_folds]):
        assignment[sid] = fold
        fold_counts[fold].update(subject_counts[sid])
        fold_subjects[fold] += 1

    for sid in subjects[num_folds:]:
        counts = subject_counts[sid]
        best_fold = 0
        best_score: tuple[float, int, int] | None = None
        for fold in range(num_folds):
            fold_counts[fold].update(counts)
            fold_subjects[fold] += 1
            score = global_score()
            fold_counts[fold].subtract(counts)
            fold_subjects[fold] -= 1
            candidate = (score, fold_subjects[fold], fold)
            if best_score is None or candidate < best_score:
                best_score = candidate
                best_fold = fold
        assignment[sid] = best_fold
        fold_counts[best_fold].update(counts)
        fold_subjects[best_fold] += 1
    return assignment


def _write_manifest(path: Path, rows: list[dict[str, str]], subject_to_fold: dict[str, int], *, heldout_fold: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["split"] = "val" if subject_to_fold[str(row["subject_id"])] == heldout_fold else "train"
            writer.writerow(out)


def _summarize(rows: list[dict[str, str]], subject_to_fold: dict[str, int], *, num_folds: int) -> dict[str, object]:
    fold_rows = [[] for _ in range(num_folds)]
    for row in rows:
        fold_rows[subject_to_fold[str(row["subject_id"])]].append(row)
    summary: dict[str, object] = {"num_folds": num_folds, "folds": []}
    for fold, heldout in enumerate(fold_rows):
        train = [row for idx, part in enumerate(fold_rows) if idx != fold for row in part]
        summary["folds"].append(
            {
                "fold": fold,
                "heldout_subjects": len({row["subject_id"] for row in heldout}),
                "train_subjects": len({row["subject_id"] for row in train}),
                "heldout_rows": len(heldout),
                "train_rows": len(train),
                "heldout_label_type_counts": {str(k): int(v) for k, v in sorted(Counter(int(float(r["label_type"])) for r in heldout).items())},
                "train_label_type_counts": {str(k): int(v) for k, v in sorted(Counter(int(float(r["label_type"])) for r in train).items())},
            }
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", default="neuromm26_datasets/annotations/neuromm2026_train_val_all_labeled_train_only.csv")
    parser.add_argument("--output-dir", default="neuromm26_datasets/annotations/source_localization_folds")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--prefix", default="neuromm2026_task3_subject5fold")
    args = parser.parse_args()

    rows = _read_rows(Path(args.input_manifest))
    subject_counts = _subject_counts(rows)
    subject_to_fold = _assign_subject_folds(subject_counts, num_folds=int(args.num_folds), seed=int(args.seed))
    out_dir = Path(args.output_dir)
    for fold in range(int(args.num_folds)):
        _write_manifest(out_dir / f"{args.prefix}_fold{fold}.csv", rows, subject_to_fold, heldout_fold=fold)
    summary = _summarize(rows, subject_to_fold, num_folds=int(args.num_folds))
    summary["input_manifest"] = str(args.input_manifest)
    summary["subject_to_fold"] = subject_to_fold
    summary_path = out_dir / f"{args.prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "subject_to_fold"}, ensure_ascii=False, indent=2))
    print(f"[done] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
