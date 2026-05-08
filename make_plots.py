"""Geração de figuras para o relatório.

Subcomandos:
  curves      - curva de aprendizado de um treino (progress.csv)
  bars        - barras de cobertura por tamanho do grid (eval JSON)
  plasticity  - curvas de dormant ratio, weight norm, stable rank (3 estágios)
  cross       - matriz de cross-evaluation (modelo × tamanho)
  all         - curves + bars + plasticity + cross
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


def cmd_plasticity(args: argparse.Namespace) -> None:
    """Plota dormant_total, weight_norm agregado e stable_rank vs timesteps,
    concatenando os 3 estágios do currículo numa figura. Marca as fronteiras
    entre estágios com linhas verticais. Métricas no log progress.csv via
    PlasticityCallback (Sokar 2023, Lyle 2024, Kumar 2021).
    """
    os.makedirs(FIG_DIR, exist_ok=True)
    log_dirs = args.log_dirs
    if not log_dirs:
        print("[warn] --log-dirs vazio; nada a plotar")
        return

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    cum_offset = 0
    boundaries = []
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for idx, log_dir in enumerate(log_dirs):
        csv_path = os.path.join(log_dir, "progress.csv")
        if not os.path.isfile(csv_path):
            print(f"[warn] sem progress.csv em {log_dir}")
            continue
        df = pd.read_csv(csv_path)
        if "time/total_timesteps" not in df.columns:
            continue
        x_local = df["time/total_timesteps"].to_numpy()
        x_global = x_local + cum_offset

        color = palette[idx % len(palette)]
        label = f"stage{idx+1}"

        # Painel 1: dormant ratio total
        if "plasticity/dormant_total" in df.columns:
            mask = df["plasticity/dormant_total"].notna()
            axes[0].plot(x_global[mask], df.loc[mask, "plasticity/dormant_total"],
                         color=color, label=label, linewidth=1.5)

        # Painel 2: weight norm — soma sobre todos blocos
        wnorm_cols = [c for c in df.columns if c.startswith("plasticity/wnorm_")]
        if wnorm_cols:
            total_norm = np.sqrt((df[wnorm_cols].fillna(0) ** 2).sum(axis=1))
            mask = (total_norm > 0)
            axes[1].plot(x_global[mask], total_norm[mask],
                         color=color, label=label, linewidth=1.5)

        # Painel 3: stable rank
        if "plasticity/stable_rank" in df.columns:
            mask = df["plasticity/stable_rank"].notna()
            axes[2].plot(x_global[mask], df.loc[mask, "plasticity/stable_rank"],
                         color=color, label=label, linewidth=1.5)

        if x_local.size:
            cum_offset += int(x_local.max())
            boundaries.append(cum_offset)

    # Boundary lines + axis labels
    for ax in axes:
        for b in boundaries[:-1]:
            ax.axvline(b, color="grey", linestyle="--", alpha=0.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    axes[0].set_ylabel("dormant ratio (total)")
    axes[0].set_title("Dormant neuron ratio — Sokar et al. 2023 (τ=0.025)")
    axes[1].set_ylabel("weight L2 norm (todos blocos)")
    axes[1].set_title("Weight L2 norm — preditor de plasticity loss (Klein 2024, Lyle 2024)")
    axes[2].set_ylabel("stable rank das features")
    axes[2].set_title("Stable rank — Kumar et al. ICLR 2021")
    axes[2].set_xlabel("timesteps cumulativos (currículo)")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "plasticity_curriculum.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def cmd_cross(args: argparse.Namespace) -> None:
    """Plota matriz de cross-evaluation (linhas = modelo, colunas = tamanho).
    Detecta forgetting (quedas) e negative transfer.
    """
    os.makedirs(FIG_DIR, exist_ok=True)
    rows = _load(args.cross_json)
    sizes = sorted({r["size"] for r in rows})
    by_model: Dict[str, Dict[int, float]] = {}
    for r in rows:
        by_model.setdefault(r["model"], {})[r["size"]] = r["full_coverage_rate_pct"]
    models = list(by_model.keys())

    matrix = np.array([[by_model[m].get(s, np.nan) for s in sizes] for m in models])

    fig, ax = plt.subplots(figsize=(7, 1.2 + 0.6 * len(models)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{s}×{s}" for s in sizes])
    ax.set_yticks(range(len(models)))
    short_names = [m.replace("ppo_cpp_", "").replace(".zip", "")[:30] for m in models]
    ax.set_yticklabels(short_names, fontsize=8)
    for i in range(len(models)):
        for j in range(len(sizes)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=10, color="black" if 30 <= v <= 80 else "white")
    fig.colorbar(im, ax=ax, label="full coverage rate (%)")
    ax.set_title("Cross-evaluation: modelo × tamanho de grid")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "cross_eval_matrix.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def cmd_all(args: argparse.Namespace) -> None:
    cmd_curves(args)
    if args.eval_json and os.path.isfile(args.eval_json):
        cmd_bars(args)
    if args.log_dirs:
        cmd_plasticity(args)
    if args.cross_json and os.path.isfile(args.cross_json):
        cmd_cross(args)


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

    pp = sub.add_parser("plasticity", help="curvas de plasticidade (3 estágios)")
    pp.add_argument("--log-dirs", nargs="+", required=True,
                    help="Pastas de log de cada estágio, em ordem")
    pp.set_defaults(func=cmd_plasticity)

    pcr = sub.add_parser("cross", help="matriz de cross-evaluation")
    pcr.add_argument("--cross-json", required=True)
    pcr.set_defaults(func=cmd_cross)

    pall = sub.add_parser("all", help="curves + bars + plasticity + cross")
    pall.add_argument("--log-dirs", nargs="+", default=[])
    pall.add_argument("--eval-json", type=str, default=None)
    pall.add_argument("--cross-json", type=str, default=None)
    pall.set_defaults(func=cmd_all)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
