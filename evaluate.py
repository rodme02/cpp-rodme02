"""Standalone evaluation helper.

Runs N episodes for one or more (size, model) pairs with fixed seeds and
prints a comparable metrics table. Optionally exports JSON for the
report.

Examples
--------
    python evaluate.py \
        --pair 5  data/ppo_cpp_5_3_100_500000_..._stage1.zip \
        --pair 10 data/ppo_cpp_10_12_400_1000000_..._stage2.zip \
        --pair 20 data/ppo_cpp_20_50_1600_2000000_..._stage3.zip \
        --episodes 100 --deterministic --out results/eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import List, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO

from gymnasium_env.grid_world_cpp import GridWorldCPPEnv


def set_global_seed(seed: int) -> None:
    """Seed every RNG that influences a PPO rollout.

    Without this, ``deterministic=False`` evaluations vary run-to-run
    because the action-sampling RNG is not pinned. Numbers in the report
    were collected with this set; reproducing them outside the script
    requires the same call.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


SIZE_DEFAULTS = {
    5:  {"obs_quantity": 3,  "max_steps": 100},
    10: {"obs_quantity": 12, "max_steps": 600},
    20: {"obs_quantity": 50, "max_steps": 2400},
}


def evaluate(model_path: str, size: int, episodes: int, seed: int,
             deterministic: bool) -> dict:
    set_global_seed(seed)
    cfg = SIZE_DEFAULTS.get(size) or {
        "obs_quantity": max(1, int(0.12 * size * size)),
        "max_steps": 4 * size * size,
    }
    env = GridWorldCPPEnv(
        size=size,
        obs_quantity=cfg["obs_quantity"],
        max_steps=cfg["max_steps"],
        render_mode="rgb_array",
    )
    model = PPO.load(model_path, device="cpu")

    coverages, steps_list = [], []
    full = 0
    for i in range(episodes):
        obs, info = env.reset(seed=seed + i)
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, info = env.step(int(action))
            steps += 1
        coverages.append(info["coverage"])
        steps_list.append(steps)
        if terminated and not truncated:
            full += 1
    env.close()

    cov = np.array(coverages)
    stp = np.array(steps_list)
    return {
        "size": size,
        "model": os.path.basename(model_path),
        "episodes": episodes,
        "deterministic": deterministic,
        "full_coverage_rate_pct": 100.0 * full / episodes,
        "coverage_mean_pct": 100.0 * cov.mean(),
        "coverage_std_pct": 100.0 * cov.std(),
        "coverage_min_pct": 100.0 * cov.min(),
        "coverage_max_pct": 100.0 * cov.max(),
        "steps_mean": float(stp.mean()),
        "steps_std": float(stp.std()),
        "steps_min": int(stp.min()),
        "steps_max": int(stp.max()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pair", action="append", nargs=2, metavar=("SIZE", "MODEL"),
                   required=True, help="size and model path; can be repeated.")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows: List[dict] = []
    for size_str, model_path in args.pair:
        size = int(size_str)
        print(f"\n>>> Evaluating size={size}  model={model_path}")
        row = evaluate(model_path, size, args.episodes, args.seed,
                       args.deterministic)
        rows.append(row)
        print(f"   full coverage : {row['full_coverage_rate_pct']:.2f}%")
        print(f"   avg coverage  : {row['coverage_mean_pct']:.2f}% "
              f"(±{row['coverage_std_pct']:.2f}%)")
        print(f"   avg steps     : {row['steps_mean']:.1f} "
              f"(±{row['steps_std']:.1f})")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nWrote {args.out}")

    print("\n=== Summary ===")
    print(f"{'Size':>5} {'Full %':>8} {'Cov %':>8} {'σ':>6} {'Steps':>8} {'σ':>6}")
    for r in rows:
        print(f"{r['size']:>5} {r['full_coverage_rate_pct']:>8.2f} "
              f"{r['coverage_mean_pct']:>8.2f} {r['coverage_std_pct']:>6.2f} "
              f"{r['steps_mean']:>8.1f} {r['steps_std']:>6.1f}")


if __name__ == "__main__":
    main()
