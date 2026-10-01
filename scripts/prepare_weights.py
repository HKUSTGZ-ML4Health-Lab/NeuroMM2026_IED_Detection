#!/usr/bin/env python3
"""Link the six official checkpoints under readable local names and verify hashes."""
from __future__ import annotations
import argparse
import csv
import hashlib
from pathlib import Path

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Extracted weights directory')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with (root / 'weights/MANIFEST.tsv').open(newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    for row in rows:
        # The Track 3-only archive uses the readable path. The glob also accepts
        # the older combined archive for users who already downloaded it.
        matches = [args.source / row['path'], *sorted(args.source.glob(row['archive_glob']))]
        valid = [p for p in matches if p.is_file() and sha256(p) == row['sha256']]
        if not valid:
            raise ValueError(f"No checksum-matching checkpoint found for {row['path']}")
        source = valid[0].resolve()
        target = root / 'weights' / row['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.resolve() != source:
            if target.is_symlink():
                target.unlink()
            elif target.exists():
                raise FileExistsError(f'Conflicting checkpoint: {target}')
            target.symlink_to(source)
        if args.verify and sha256(target) != row['sha256']:
            raise ValueError(f'Hash mismatch: {target}')
        print(f"[ok] {row['path']}")

if __name__ == '__main__':
    main()
