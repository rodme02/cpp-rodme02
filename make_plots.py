"""Geração de figuras para o relatório.

Subcomandos:
  curves    - curva de aprendizado de um treino (progress.csv)
  ablation  - barras comparando v1 / v2 / v3 (eval JSONs)
  bars      - barras de cobertura por tamanho do grid (eval JSON)
  all       - tudo de uma vez (chama os 3 acima)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIG_DIR = "results/figures"
RESULTS = "results"


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def cmd_curves(args: argparse.Namespace) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    for log_dir in args.log_dirs:
        csv_path = os.path.join(log_dir, "progress.csv")
        if not os.path.isfile(csv_path):
            print(f"[warn] sem progress.csv em {log_dir}")
            continue
        df = pd.read_csv(csv_path)
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        ax1.plot(df["time/total_timesteps"], df["rollout/ep_rew_mean"],
                 color="#1f77b4")
        ax1.set_xlabel("timesteps")
        ax1.set_ylabel("ep_rew_mean", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax1.grid(alpha=0.3)
        if "rollout/ep_len_mean" in df.columns:
            ax2 = ax1.twinx()
            ax2.plot(df["time/total_timesteps"], df["rollout/ep_len_mean"],
                     color="#d62728", alpha=0.7)
            ax2.set_ylabel("ep_len_mean", color="#d62728")
            ax2.tick_params(axis="y", labelcolor="#d62728")
        tag = os.path.basename(log_dir.rstrip("/"))
        plt.title(f"Curva de aprendizado — {tag}")
        fig.tight_layout()
        out = os.path.join(FIG_DIR, f"learning_curve_{tag}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"wrote {out}")


def cmd_bars(args: argparse.Namespace) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    rows = _load(args.eval_json)
    sizes = [r["size"] for r in rows]
    full = [r["full_coverage_rate_pct"] for r in rows]
    avg = [r["coverage_mean_pct"] for r in rows]
    std = [r["coverage_std_pct"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(sizes))
    w = 0.35
    ax.bar(x - w / 2, full, w, label="full coverage rate (%)", color="#2ca02c")
    ax.bar(x + w / 2, avg, w, yerr=std, capsize=4,
           label="cobertura média (%)", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}x{s}" for s in sizes])
    ax.set_ylim(0, 105)
    ax.set_ylabel("%")
    ax.set_title("Cobertura por tamanho de grid (100 eps, estocástico)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "coverage_bars.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def _v1_combined() -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    for size, fn in [(5, "v1_eval_stage1_stoch.json"),
                     (10, "v1_eval_stage2_stoch.json")]:
        path = os.path.join(RESULTS, fn)
        if os.path.isfile(path):
            out[size] = _load(path)[0]
    return out


def _v2_combined() -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    for size, fn in [(5, "v2_eval_stage1_stoch.json")]:
        path = os.path.join(RESULTS, fn)
        if os.path.isfile(path):
            out[size] = _load(path)[0]
    legacy = os.path.join(RESULTS, "eval_v2_stoch.json")
    if os.path.isfile(legacy):
        for row in _load(legacy):
            out[row["size"]] = row
    return out


def _by_size(path: str) -> Dict[int, dict]:
    if not os.path.isfile(path):
        return {}
    return {row["size"]: row for row in _load(path)}


def cmd_ablation(args: argparse.Namespace) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    data = {
        "v1": _v1_combined(),
        "v2": _v2_combined(),
        "v3": _by_size(os.path.join(RESULTS, "eval_final_stoch.json")),
    }
    sizes = sorted({s for v in data.values() for s in v.keys()}) or [5, 10, 20]
    versions = [v for v in ("v1", "v2", "v3") if data[v]]

    metrics = [
        ("full_coverage_rate_pct", "Full coverage rate (%)"),
        ("coverage_mean_pct", "Cobertura média (%)"),
    ]
    colors = {"v1": "#cbd5e8", "v2": "#67a9cf", "v3": "#1f6db8"}
    labels = {
        "v1": "v1: só local_map",
        "v2": "v2: + global_map",
        "v3": "v3: + frontier + shaping",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    n = max(1, len(versions))
    width = 0.8 / n
    for ax, (metric, title) in zip(axes, metrics):
        x = np.arange(len(sizes))
        for k, ver in enumerate(versions):
            offset = (k - (n - 1) / 2) * width
            vals = [data[ver].get(s, {}).get(metric, np.nan) for s in sizes]
            ax.bar(x + offset, vals, width, label=labels[ver],
                   color=colors[ver])
            for xi, v in zip(x + offset, vals):
                if not np.isnan(v):
                    ax.text(xi, v + 1, f"{v:.1f}", ha="center",
                            va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}x{s}" for s in sizes])
        ax.set_ylim(0, 108)
        ax.set_ylabel("%")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9, loc="lower left" if "média" in title else "lower right")

    fig.suptitle("Ablação dos três incrementos (100 eps estocástico)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "ablation_comparison.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def cmd_all(args: argparse.Namespace) -> None:
    cmd_curves(args)
    if args.eval_json and os.path.isfile(args.eval_json):
        cmd_bars(args)
    cmd_ablation(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("curves", help="curvas de aprendizado")
    pc.add_argument("--log-dirs", nargs="+", required=True)
    pc.set_defaults(func=cmd_curves)

    pb = sub.add_parser("bars", help="barras de cobertura")
    pb.add_argument("--eval-json", required=True)
    pb.set_defaults(func=cmd_bars)

    pa = sub.add_parser("ablation", help="ablação v1/v2/v3")
    pa.set_defaults(func=cmd_ablation)

    pall = sub.add_parser("all", help="curves + bars + ablation")
    pall.add_argument("--log-dirs", nargs="+", default=[])
    pall.add_argument("--eval-json", type=str, default=None)
    pall.set_defaults(func=cmd_all)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
