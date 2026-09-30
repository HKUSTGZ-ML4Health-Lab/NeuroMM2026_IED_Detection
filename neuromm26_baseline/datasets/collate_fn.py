"""Batch collation helpers."""

from __future__ import annotations

import torch


def neuromm_collate(batch):
    collated = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if all(isinstance(value, torch.Tensor) for value in values):
            collated[key] = torch.stack(values)
        elif all(value is None for value in values):
            collated[key] = None
        else:
            collated[key] = values
    return collated
