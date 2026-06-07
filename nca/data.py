from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_target(path: str | Path, size: int | None = None) -> torch.Tensor:
    image = Image.open(path).convert("RGBA")
    if size is not None:
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    data = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).float() / 255.0
    return data.permute(2, 0, 1).contiguous()


def make_seed(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    device: torch.device | str,
    y: int | None = None,
    x: int | None = None,
) -> torch.Tensor:
    state = torch.zeros(batch_size, channels, height, width, device=device)
    cy = height // 2 if y is None else y
    cx = width // 2 if x is None else x
    state[:, 3:, cy, cx] = 1.0
    return state


def pad_target(target: torch.Tensor, padding: int) -> torch.Tensor:
    if padding <= 0:
        return target
    return torch.nn.functional.pad(target, (padding, padding, padding, padding))
