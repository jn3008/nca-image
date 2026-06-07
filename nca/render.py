from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch


def state_to_rgba(x: torch.Tensor) -> np.ndarray:
    frame = x.detach().cpu()[0, :4].permute(1, 2, 0).clamp(0.0, 1.0).numpy()
    return (frame * 255).astype(np.uint8)


def composite_on_background(rgba: np.ndarray, color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    rgb = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    background = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    return (rgb * alpha + background * (1.0 - alpha)).astype(np.uint8)


def save_gif(frames: list[np.ndarray], path: str | Path, fps: int = 20) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, frames, duration=1000 / fps, loop=0)

