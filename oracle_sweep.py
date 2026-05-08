"""Sweep do teto estrutural por densidade de obstáculos no 20x20.

Roda apenas o connectivity check (mais barato que o oracle completo) sobre
100 layouts (sementes 10000-10099) para várias contagens de obstáculos.
Ajuda a calibrar uma distribuição de avaliação tratável sob visibilidade
parcial.
"""
from __future__ import annotations

import json
from collections import deque

from gymnasium_env.grid_world_cpp import GridWorldCPPEnv


def connectivity_ratio(env: GridWorldCPPEnv) -> float:
    obstacles = env._obstacles_set
    ax, ay = int(env._agent_location[0]), int(env._agent_location[1])
    size = env.size
    queue = deque([(ax, ay)])
    visited = {(ax, ay)}
    while queue:
        cx, cy = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if (0 <= nx < size and 0 <= ny < size
                    and (nx, ny) not in obstacles
                    and (nx, ny) not in visited):
                visited.add((nx, ny))
                queue.append((nx, ny))
    total_free = size * size - len(obstacles)
    if total_free == 0:
        return 1.0
    return len(visited) / total_free


def sweep(size: int, obstacle_counts: list[int],
          episodes: int = 100, seed: int = 10000):
    rows = []
    print(f"\n--- Sweep size={size}, episodes={episodes} ---")
    print(f"{'obs':>4} {'fully reachable':>18} {'mean reachable':>18}")
    for q in obstacle_counts:
        full = 0
        means = []
        for i in range(episodes):
            env = GridWorldCPPEnv(
                size=size, obs_quantity=q, max_steps=100,
                shaping_enabled=False,
            )
            env.reset(seed=seed + i)
            r = connectivity_ratio(env)
            means.append(r)
            if r >= 0.9999:
                full += 1
        full_pct = 100 * full / episodes
        mean_pct = 100 * sum(means) / len(means)
        print(f"{q:>4} {full_pct:>17.1f}% {mean_pct:>17.2f}%")
        rows.append({
            "size": size, "obstacles": q,
            "fully_reachable_pct": full_pct,
            "mean_reachable_pct": mean_pct,
        })
    return rows


if __name__ == "__main__":
    out = []
    out.extend(sweep(5,  [1, 2, 3, 4, 5]))
    out.extend(sweep(10, [6, 9, 12, 15, 18, 20]))
    out.extend(sweep(20, [25, 30, 35, 40, 45, 50, 55, 60]))
    with open("results/oracle_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote results/oracle_sweep.json")
