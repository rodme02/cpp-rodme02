"""Geração de figuras para o relatório.

Subcomandos:
  curves      - curva de aprendizado de um treino (progress.csv)
  ablation    - barras comparando v1 / v2 / v3 (eval JSONs)
  bars        - barras de cobertura por tamanho do grid (eval JSON)
  variants    - comparação de variantes de recipe (full + steps)
  plasticity  - curvas de dormant ratio, weight norm, stable rank (3 estágios)
  cross       - matriz de cross-evaluation (modelo × tamanho)
  all         - curves + bars + ablation + plasticity + cross
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
        "v3": _by_size(os.path.join(RESULTS, "eval_p0p1p2_stoch.json")),
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


def cmd_variants(args: argparse.Namespace) -> None:
    """Bar chart das variantes de recipe (full coverage e steps por tamanho)."""
    os.makedirs(FIG_DIR, exist_ok=True)
    runs = [
        ("Recipe atual (wd=1e-4, F=8)", "results/eval_p0p1p2_stoch.json", "#1f77b4"),
        ("wd=5e-4",                     "results/eval_p3_wd5e4_stoch.json", "#2ca02c"),
        ("F=16",                        "results/eval_p3b_F16_stoch.json",  "#d62728"),
    ]
    sizes = [5, 10, 20]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    n = len(runs)
    w = 0.8 / n
    x = np.arange(len(sizes))

    for i, (label, path, color) in enumerate(runs):
        rows = _by_size(path)
        full = [rows.get(s, {}).get("full_coverage_rate_pct", np.nan) for s in sizes]
        steps = [rows.get(s, {}).get("steps_mean", np.nan) for s in sizes]
        offset = (i - (n - 1) / 2) * w
        axes[0].bar(x + offset, full, w, label=label, color=color)
        axes[1].bar(x + offset, steps, w, label=label, color=color)
        for xi, v in zip(x + offset, full):
            if not np.isnan(v):
                axes[0].text(xi, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        for xi, v in zip(x + offset, steps):
            if not np.isnan(v):
                axes[1].text(xi, v + max(steps) * 0.01, f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    for ax, (ylab, title) in zip(axes,
                                  [("Full coverage rate (%)", "Full coverage rate"),
                                   ("Steps médios por episódio", "Steps médios (menor = mais eficiente)")]):
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}×{s}" for s in sizes])
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylim(0, 110)

    fig.suptitle("Variantes da recipe (100 eps estocástico)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "comparison_4runs.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def cmd_all(args: argparse.Namespace) -> None:
    cmd_curves(args)
    if args.eval_json and os.path.isfile(args.eval_json):
        cmd_bars(args)
    cmd_ablation(args)
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

    pa = sub.add_parser("ablation", help="ablação v1/v2/v3")
    pa.set_defaults(func=cmd_ablation)

    pp = sub.add_parser("plasticity", help="curvas de plasticidade (3 estágios)")
    pp.add_argument("--log-dirs", nargs="+", required=True,
                    help="Pastas de log de cada estágio, em ordem")
    pp.set_defaults(func=cmd_plasticity)

    pcr = sub.add_parser("cross", help="matriz de cross-evaluation")
    pcr.add_argument("--cross-json", required=True)
    pcr.set_defaults(func=cmd_cross)

    pv = sub.add_parser("variants", help="comparação de variantes de recipe")
    pv.set_defaults(func=cmd_variants)

    pall = sub.add_parser("all", help="curves + bars + ablation + plasticity + cross")
    pall.add_argument("--log-dirs", nargs="+", default=[])
    pall.add_argument("--eval-json", type=str, default=None)
    pall.add_argument("--cross-json", type=str, default=None)
    pall.set_defaults(func=cmd_all)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
