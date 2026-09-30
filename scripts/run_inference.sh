#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${NEUROMM_DATA_ROOT:?Set NEUROMM_DATA_ROOT to official extracted data}"
output_root="${NEUROMM_OUTPUT_ROOT:-$repo_root/outputs}"
python_bin="${PYTHON_BIN:-python}"
device="${DEVICE:-cuda}"
batch_size="${BATCH_SIZE:-512}"
num_workers="${NUM_WORKERS:-0}"
anchor="$repo_root/weights/eeg_anchor.pt"
[[ -f "$anchor" ]] || { echo "Missing $anchor; run scripts/prepare_weights.py" >&2; exit 1; }
mkdir -p "$output_root/submissions" "$output_root/eeg_cache" "$output_root/video_cache"
cd "$repo_root"
diags=()
for fold in 0 1 2 3 4; do
  ckpt="$repo_root/weights/folds/fold_${fold}.pt"
  [[ -f "$ckpt" ]] || { echo "Missing $ckpt; run scripts/prepare_weights.py" >&2; exit 1; }
  "$python_bin" scripts/predict.py \
    --checkpoint "$ckpt" --eeg-anchor-checkpoint "$anchor" \
    --candidate-dir "$data_root/candidate" --output-dir "$output_root/submissions" \
    --output-prefix "fold_${fold}" --batch-size "$batch_size" --num-workers "$num_workers" \
    --device "$device" --amp-enabled --amp-dtype float16 --cache-on-device \
    --eeg-cache-root "$output_root/eeg_cache" \
    --video-cache-path "$output_root/video_cache/candidate_video.pt" \
    --write-diagnostics --overwrite
  diags+=("$output_root/submissions/fold_${fold}.diagnostics.csv")
done
"$python_bin" scripts/ensemble.py --diagnostics "${diags[@]}" \
  --output-root "$output_root" --output-prefix submission --overwrite
echo "Submission: $output_root/submissions/submission.zip"
