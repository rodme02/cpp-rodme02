"""Avalia cada modelo em todos os tamanhos (matriz modelo × tamanho)."""
from __future__ import annotations

import argparse
import json
import os
from typing import List

from evaluate import evaluate


SIZES = [5, 10, 20]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True,
                   help="Paths to model zip files (typically 3: stage1/2/3)")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--sizes", type=int, nargs="+", default=SIZES,
                   help="Grid sizes to evaluate on (default: 5 10 20)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows: List[dict] = []
    for mp in args.models:
        for size in args.sizes:
            print(f"\n>>> Cross-eval: model={os.path.basename(mp)}  size={size}")
            row = evaluate(mp, size, args.episodes, args.seed,
                           args.deterministic)
            rows.append(row)
            print(f"   full coverage : {row['full_coverage_rate_pct']:.2f}%")
            print(f"   avg coverage  : {row['coverage_mean_pct']:.2f}% "
                  f"(±{row['coverage_std_pct']:.2f}%)")
            print(f"   repeat ratio  : {row.get('repeat_ratio_mean', 0.0):.3f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nWrote {args.out}")

    # Pretty-print a matrix: rows = models, cols = sizes
    print("\n=== Cross-eval matrix (full_coverage_rate %) ===")
    header = f"{'Model':<48}" + "".join(f"{s:>10}" for s in args.sizes)
    print(header)
    by_model: dict[str, dict[int, float]] = {}
    for r in rows:
        by_model.setdefault(r["model"], {})[r["size"]] = r["full_coverage_rate_pct"]
    for mname, sizes in by_model.items():
        line = f"{mname[:48]:<48}"
        for s in args.sizes:
            v = sizes.get(s)
            line += f"{v:>10.2f}" if v is not None else f"{'-':>10}"
        print(line)


if __name__ == "__main__":
    main()
