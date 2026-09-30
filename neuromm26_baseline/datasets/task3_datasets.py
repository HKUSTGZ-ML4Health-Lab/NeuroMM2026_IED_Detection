"""Task3 datasets: 5-class seizure-type localization.

Filters rows from the underlying manifests where label_type > 0 (i.e. the
positive seizure samples) and returns the class index 0..4 as a torch.long
label suitable for CrossEntropyLoss.

Three classes:
- Task3EEGFeatureDataset      EEG-only
- Task3MultimodalFeatureDataset  EEG + video feature
- Task3VideoFeatureDataset    Video feature only (mean-pooled, for video MLP)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def _is_positive_row(row: dict) -> bool:
    v = (row.get("label_type", "") or "").strip()
    if v == "" or v == "0":
        return False
    try:
        return int(v) > 0
    except ValueError:
        return False


def _class_index(row: dict) -> int:
    """0-indexed class (0..4) for label_type 1..5."""
    if row.get("split") == "candidate":
        return -1  # Candidate labels are unavailable and never used for inference.
    return int(row["label_type"]) - 1


def _load_filtered(manifest_csv: str, split: str) -> list[dict]:
    with Path(manifest_csv).open("r", encoding="utf-8-sig") as h:
        rows = [r for r in csv.DictReader(h) if r.get("split") == split]
    if split != "candidate":
        rows = [r for r in rows if _is_positive_row(r)]
    if not rows:
        raise ValueError(f"No samples for split={split} in {manifest_csv}")
    return rows


class Task3EEGFeatureDataset(Dataset):
    """EEG-only multi-class dataset. Reuses the EEG normalization scheme
    from EEGFeatureDataset (29-channel raw -> 26-channel after ECG/EMG
    derivation, padded/cropped to (26, 2000))."""

    def __init__(
        self,
        manifest_csv: str,
        split: str,
        eeg_feature_root: str,
        preload_in_memory: bool = True,
        target_shape: tuple[int, int] = (26, 2000),
        eeg_preprocess: str = "baseline26",
        eeg_normalization: str = "none",
    ) -> None:
        self.records = _load_filtered(manifest_csv, split)
        self.eeg_feature_root = Path(eeg_feature_root)
        self.preload_in_memory = preload_in_memory
        self.target_shape = target_shape
        self.eeg_normalization = str(eeg_normalization)
        self.eeg_preprocess = str(eeg_preprocess)
        if self.eeg_preprocess not in {
            "baseline26",
            "car26",
            "raw23_direct",
            "car23_only",
            "car26_hf_suppress",
            "bipolar_laplacian26",
            "car26_low80",
            "car26_low200",
            "car26_hfa80",
            "vep_region_ref26",
        }:
            raise ValueError(f"Unsupported Task3 EEG preprocess: {self.eeg_preprocess!r}")
        self.labels = [_class_index(r) for r in self.records]
        self.sample_ids = [r["sample_id"] for r in self.records]

        self._cache: list[torch.Tensor] | None = None
        if preload_in_memory:
            self._cache = []
            desc = f"Preloading EEG {split} (task3)"
            for sid in tqdm(self.sample_ids, desc=desc):
                self._cache.append(self._load_eeg_tensor(sid))

    def _normalize(self, wave: np.ndarray) -> np.ndarray:
        wave = wave.astype(np.float32, copy=True)
        if self.eeg_preprocess == "raw23_direct":
            return wave[:23, ...]
        if wave.shape[0] >= 29:
            wave[:23, ...] = wave[:23, ...] / 1e-3
            if self.eeg_preprocess in {"car26", "car23_only", "car26_hf_suppress", "car26_low80", "car26_low200", "car26_hfa80"}:
                wave[:23, ...] = wave[:23, ...] - wave[:23, ...].mean(axis=0, keepdims=True)
            wave[23:, ...] = wave[23:, ...] * 1e-2
            heart_wave = wave[23, :] - wave[24, :]
            muscle_wave1 = wave[25, :] - wave[26, :]
            muscle_wave2 = wave[27, :] - wave[28, :]
            heart_muscle = np.stack([heart_wave, muscle_wave1, muscle_wave2], axis=0)
            if self.eeg_preprocess == "car23_only":
                return wave[:23, ...]
            if self.eeg_preprocess == "car26_hf_suppress":
                wave[:23, ...] = self._attenuate_high_frequency(wave[:23, ...], fs=1000.0, cutoff_hz=120.0, attenuation=0.35)
            if self.eeg_preprocess == "car26_low80":
                wave[:23, ...] = self._fft_bandpass(wave[:23, ...], fs=1000.0, low_hz=None, high_hz=80.0)
            if self.eeg_preprocess == "car26_low200":
                wave[:23, ...] = self._fft_bandpass(wave[:23, ...], fs=1000.0, low_hz=None, high_hz=200.0)
            if self.eeg_preprocess == "car26_hfa80":
                wave[:23, ...] = self._fft_bandpass(wave[:23, ...], fs=1000.0, low_hz=80.0, high_hz=None)
            if self.eeg_preprocess == "vep_region_ref26":
                eeg = wave[:23, ...]
                ref = eeg[[19, 20, 21, 22], ...].mean(axis=0, keepdims=True)
                region_ref = eeg - ref
                frontal = eeg[[0, 1, 2, 3, 16], ...].mean(axis=0)
                temporal_l = eeg[[10, 11, 12], ...].mean(axis=0)
                temporal_r = eeg[[13, 14, 15], ...].mean(axis=0)
                centro_parietal = eeg[[4, 5, 6, 7, 17, 18], ...].mean(axis=0)
                occipital = eeg[[8, 9], ...].mean(axis=0)
                region_contrasts = np.stack(
                    [
                        occipital - frontal,
                        temporal_l - temporal_r,
                        centro_parietal - frontal,
                    ],
                    axis=0,
                )
                return np.concatenate([region_ref, region_contrasts], axis=0)
            if self.eeg_preprocess == "bipolar_laplacian26":
                eeg = wave[:23, ...]
                bipolar = eeg[1:, ...] - eeg[:-1, ...]
                global_trace = eeg.mean(axis=0, keepdims=True)
                return np.concatenate([bipolar, global_trace, heart_muscle], axis=0)
            wave = np.concatenate([wave[:23, ...], heart_muscle], axis=0)
        elif self.eeg_preprocess in {"car26", "car23_only", "car26_hf_suppress", "car26_low80", "car26_low200", "car26_hfa80"} and wave.shape[0] > 1:
            eeg_channels = min(23, int(wave.shape[0]))
            wave[:eeg_channels, ...] = wave[:eeg_channels, ...] - wave[:eeg_channels, ...].mean(axis=0, keepdims=True)
            if self.eeg_preprocess == "car23_only":
                return wave[:eeg_channels, ...]
            if self.eeg_preprocess == "car26_hf_suppress":
                wave[:eeg_channels, ...] = self._attenuate_high_frequency(
                    wave[:eeg_channels, ...], fs=1000.0, cutoff_hz=120.0, attenuation=0.35
                )
            if self.eeg_preprocess == "car26_low80":
                wave[:eeg_channels, ...] = self._fft_bandpass(wave[:eeg_channels, ...], fs=1000.0, low_hz=None, high_hz=80.0)
            if self.eeg_preprocess == "car26_low200":
                wave[:eeg_channels, ...] = self._fft_bandpass(wave[:eeg_channels, ...], fs=1000.0, low_hz=None, high_hz=200.0)
            if self.eeg_preprocess == "car26_hfa80":
                wave[:eeg_channels, ...] = self._fft_bandpass(wave[:eeg_channels, ...], fs=1000.0, low_hz=80.0, high_hz=None)
        return wave

    @staticmethod
    def _attenuate_high_frequency(wave: np.ndarray, *, fs: float, cutoff_hz: float, attenuation: float) -> np.ndarray:
        if wave.shape[-1] < 4:
            return wave
        freqs = np.fft.rfftfreq(int(wave.shape[-1]), d=1.0 / float(fs))
        fft = np.fft.rfft(wave, axis=-1)
        fft[..., freqs >= float(cutoff_hz)] *= float(attenuation)
        return np.fft.irfft(fft, n=int(wave.shape[-1]), axis=-1).astype(np.float32, copy=False)

    @staticmethod
    def _fft_bandpass(wave: np.ndarray, *, fs: float, low_hz: float | None, high_hz: float | None) -> np.ndarray:
        if wave.shape[-1] < 4:
            return wave
        freqs = np.fft.rfftfreq(int(wave.shape[-1]), d=1.0 / float(fs))
        keep = np.ones_like(freqs, dtype=bool)
        if low_hz is not None:
            keep &= freqs >= float(low_hz)
        if high_hz is not None:
            keep &= freqs <= float(high_hz)
        fft = np.fft.rfft(wave, axis=-1)
        fft[..., ~keep] = 0.0
        return np.fft.irfft(fft, n=int(wave.shape[-1]), axis=-1).astype(np.float32, copy=False)

    def _normalize_input_scale(self, wave: np.ndarray) -> np.ndarray:
        mode = self.eeg_normalization
        if mode in {"none", None, ""}:
            return wave
        eps = 1e-6
        if mode == "per_channel_zscore":
            center = np.mean(wave, axis=1, keepdims=True)
            scale = np.std(wave, axis=1, keepdims=True)
            return (wave - center) / np.maximum(scale, eps)
        if mode == "per_channel_robust":
            center = np.median(wave, axis=1, keepdims=True)
            q25 = np.quantile(wave, 0.25, axis=1, keepdims=True)
            q75 = np.quantile(wave, 0.75, axis=1, keepdims=True)
            scale = q75 - q25
            return (wave - center) / np.maximum(scale, eps)
        raise ValueError(f"Unsupported task3 eeg_normalization: {mode}")

    def _pad_or_crop(self, wave: np.ndarray) -> np.ndarray:
        c, t = self.target_shape
        padded = np.zeros((c, t), dtype=np.float32)
        ch, ts = wave.shape
        padded[: min(ch, c), : min(ts, t)] = wave[: min(ch, c), : min(ts, t)]
        return padded

    def _load_eeg_tensor(self, sid: str) -> torch.Tensor:
        path = self.eeg_feature_root / f"{sid}.npy"
        wave = np.load(path)
        wave = self._normalize(wave)
        wave = self._normalize_input_scale(wave)
        wave = self._pad_or_crop(wave)
        return torch.from_numpy(wave.copy()).float()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        eeg = self._cache[idx] if self._cache is not None else self._load_eeg_tensor(self.sample_ids[idx])
        return {
            "sample_id": self.sample_ids[idx],
            "eeg": eeg,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
