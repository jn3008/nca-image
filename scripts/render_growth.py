from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from nca.data import make_seed
from nca.model import NeuralCA
from nca.render import composite_on_background, save_gif, state_to_rgba


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/growth.gif"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--every", type=int, default=2)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = NeuralCA(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    target = checkpoint["target"]
    _, height, width = target.shape
    x = make_seed(1, model.channels, height, width, device)
    frames = []
    with torch.no_grad():
        for step in range(args.steps + 1):
            if step % args.every == 0:
                frames.append(composite_on_background(state_to_rgba(x)))
            x = model(x, steps=1)
    save_gif(frames, args.out, fps=args.fps)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


if __name__ == "__main__":
    main()
