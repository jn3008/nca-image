from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def flatten(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().cpu().contiguous().view(-1).tolist()


def export_checkpoint(checkpoint_path: Path, out_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint["model_config"]
    state = checkpoint["model_state"]

    channels = int(model_config["channels"])
    hidden_size = int(model_config["hidden_size"])
    perception = str(model_config["perception"])

    if channels != 16:
        raise ValueError(f"WebGPU app currently expects 16 channels, got {channels}")
    if hidden_size != 128:
        raise ValueError(f"WebGPU app currently expects hidden_size 128, got {hidden_size}")
    if perception != "sobel":
        raise ValueError(f"WebGPU app currently expects perception 'sobel', got {perception!r}")

    w1 = state["update_net.0.weight"].squeeze(-1).squeeze(-1)
    b1 = state["update_net.0.bias"]
    w2 = state["update_net.2.weight"].squeeze(-1).squeeze(-1)

    target = checkpoint.get("target")
    height = int(target.shape[-2]) if target is not None else 64
    width = int(target.shape[-1]) if target is not None else 64

    payload = {
        "format": "nca-webgpu-v1",
        "step": int(checkpoint.get("step", 0)),
        "modelConfig": model_config,
        "grid": {
            "width": width,
            "height": height,
            "channels": channels,
        },
        "weights": {
            "w1Shape": list(w1.shape),
            "b1Shape": list(b1.shape),
            "w2Shape": list(w2.shape),
            "w1": flatten(w1),
            "b1": flatten(b1),
            "w2": flatten(w2),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("webgpu-sim/public/model.json"))
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
