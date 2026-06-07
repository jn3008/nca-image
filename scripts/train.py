from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from tqdm import trange

from nca.damage import random_damage
from nca.data import load_target, make_seed, pad_target
from nca.model import NeuralCA


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def train(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    run_dir = args.out / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    target = pad_target(load_target(args.target, args.size), args.padding).to(device)
    _, height, width = target.shape
    target_batch = target.unsqueeze(0).repeat(args.batch_size, 1, 1, 1)

    model = NeuralCA(
        channels=args.channels,
        hidden_size=args.hidden_size,
        fire_rate=args.fire_rate,
        perception=args.perception,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[args.steps // 2, args.steps * 3 // 4], gamma=0.3
    )
    loss_fn = nn.MSELoss()

    pool = make_seed(args.pool_size, args.channels, height, width, device)
    losses: list[float] = []

    for name, param in model.named_parameters():
        print(name, param.shape, param.numel(), param.requires_grad)

    progress = trange(args.steps, desc="train")
    for step in progress:
        indices = torch.randint(args.pool_size, (args.batch_size,), device=device)
        x = pool[indices]

        if step < args.seed_steps or random.random() < args.seed_probability:
            x = make_seed(args.batch_size, args.channels, height, width, device)

        if args.damage and step >= args.damage_after and random.random() < args.damage_probability:
            x = random_damage(x, fraction=random.uniform(0.05, args.max_damage), shape=args.damage_shape)

        roll_steps = random.randint(args.min_roll_steps, args.max_roll_steps)
        x = model(x, steps=roll_steps)
        loss = loss_fn(x[:, :4], target_batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            pool[indices] = x.detach()
            if step % args.reset_worst_every == 0:
                errors = (pool[:, :4] - target.unsqueeze(0)).square().mean(dim=(1, 2, 3))
                worst = errors.argmax()
                pool[worst : worst + 1] = make_seed(1, args.channels, height, width, device)

        value = float(loss.detach().cpu())
        losses.append(value)
        progress.set_postfix(loss=f"{value:.5f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")

        if (step + 1) % args.checkpoint_every == 0 or step + 1 == args.steps:
            save_checkpoint(run_dir / "checkpoint.pt", model, args, target.cpu(), step + 1, losses)

    with open(run_dir / "losses.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])
        writer.writerows((i + 1, loss) for i, loss in enumerate(losses))


def save_checkpoint(
    path: Path,
    model: NeuralCA,
    args: argparse.Namespace,
    target: torch.Tensor,
    step: int,
    losses: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["target"] = str(config["target"])
    config["out"] = str(config["out"])
    torch.save(
        {
            "step": step,
            "model_config": model.config(),
            "model_state": model.state_dict(),
            "target": target,
            "train_config": config,
            "losses": losses,
        },
        path,
    )
    with open(path.parent / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, default=Path("assets/target.png"))
    parser.add_argument("--out", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--fire-rate", type=float, default=0.5)
    parser.add_argument("--perception", choices=["identity", "sobel", "sobel_laplace"], default="sobel_laplace")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed-steps", type=int, default=200)
    parser.add_argument("--seed-probability", type=float, default=0.1)
    parser.add_argument("--min-roll-steps", type=int, default=32)
    parser.add_argument("--max-roll-steps", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--reset-worst-every", type=int, default=25)
    parser.add_argument("--damage", action="store_true")
    parser.add_argument("--damage-after", type=int, default=500)
    parser.add_argument("--damage-probability", type=float, default=0.5)
    parser.add_argument("--max-damage", type=float, default=0.35)
    parser.add_argument("--damage-shape", choices=["circle", "rectangle"], default="circle")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
