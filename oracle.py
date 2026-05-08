"""Oracle solver — verifica se o teto de cobertura é estrutural ou de treino.

Roda dois testes em 100 layouts (sementes 10000-10099, mesmo regime do
``evaluate.py``) por tamanho de grid:

1. **Connectivity check** (perfect-information): fração das células livres
   alcançáveis por BFS a partir da posição inicial do agente. Se < 1.0, há
   bolsões inalcançáveis -- limite estrutural intrínseco do MDP.

2. **Greedy frontier oracle** (partial-visibility): agente que conhece
   apenas ``_seen_obstacles`` e segue gulosamente para a fronteira mais
   próxima. Limite superior atingível por uma política sob visibilidade
   parcial.

Saída: full coverage rate dos dois testes; gap entre eles indica perda
por visibilidade parcial vs. estrutura do MDP.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from typing import Optional

from gymnasium_env.grid_world_cpp import GridWorldCPPEnv


SIZE_DEFAULTS = {
    5:  {"obs_quantity": 3,  "max_steps": 100},
    10: {"obs_quantity": 12, "max_steps": 600},
    20: {"obs_quantity": 50, "max_steps": 2400},
}


def connectivity_ratio(env: GridWorldCPPEnv) -> float:
    """Fração das células livres alcançáveis a partir da posição inicial,
    usando BFS sobre o mapa REAL (perfect information). Métrica estrutural
    do layout: se < 1.0, há bolsões inacessíveis.
    """
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


def oracle_action(env: GridWorldCPPEnv) -> Optional[int]:
    """Próxima ação na trajetória mais curta para a fronteira mais próxima
    (BFS sobre _seen_obstacles, mesma regra de visibilidade parcial do agente).

    Retorna ``None`` se não há fronteira (cobertura completa do conhecido) e
    ``-1`` se preso (sem fronteira mas há células não cobertas — happens em
    layouts com bolsões inalcançáveis sob a janela de visibilidade já vista).
    """
    ax, ay = int(env._agent_location[0]), int(env._agent_location[1])
    visited = env.visited
    seen_obstacles = env._seen_obstacles
    size = env.size

    # BFS com parent map para recuperar o primeiro passo da menor trajetória.
    parent = {(ax, ay): None}
    queue = deque([(ax, ay)])
    target = None

    while queue:
        cx, cy = queue.popleft()
        # É fronteira? (livre, não visitada, com vizinho visitado)
        if (cx, cy) not in visited and (cx, cy) not in seen_obstacles:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (cx + dx, cy + dy) in visited:
                    target = (cx, cy)
                    break
            if target is not None:
                break
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if (0 <= nx < size and 0 <= ny < size
                    and (nx, ny) not in seen_obstacles
                    and (nx, ny) not in parent):
                parent[(nx, ny)] = (cx, cy)
                queue.append((nx, ny))

    if target is None:
        return None

    # Trace-back: o passo desejado é o vizinho do agente na trajetória.
    cur = target
    while parent[cur] != (ax, ay) and parent[cur] is not None:
        cur = parent[cur]

    if parent[cur] is None:
        return -1  # target == agent (shouldn't happen)

    dx = cur[0] - ax
    dy = cur[1] - ay
    if dx == 1 and dy == 0:
        return 0
    if dx == 0 and dy == -1:
        return 1
    if dx == -1 and dy == 0:
        return 2
    if dx == 0 and dy == 1:
        return 3
    return -1


def oracle_episode(size: int, seed: int) -> tuple[float, int, bool]:
    """Roda um episódio com a política gulosa de fronteira. Retorna
    (coverage_final, steps, terminated_full)."""
    cfg = SIZE_DEFAULTS[size]
    env = GridWorldCPPEnv(
        size=size,
        obs_quantity=cfg["obs_quantity"],
        max_steps=cfg["max_steps"],
        shaping_enabled=False,  # Oracle não usa reward shaping
    )
    env.reset(seed=seed)

    terminated = False
    truncated = False
    steps = 0
    while not (terminated or truncated):
        a = oracle_action(env)
        if a is None or a < 0:
            # Não há fronteira conhecida — agente terminou o que sabe.
            # Escolhe ação aleatória para tentar descobrir mais (decreto).
            a = int(env.action_space.sample())
        _, _, terminated, truncated, info = env.step(int(a))
        steps += 1

    return info["coverage"], steps, bool(terminated and not truncated)


def run(sizes: list[int], episodes: int, seed: int, out_path: Optional[str]):
    summary = []
    for size in sizes:
        # Connectivity (estrutural)
        conn_full = 0
        conn_means = []
        for i in range(episodes):
            cfg = SIZE_DEFAULTS[size]
            env = GridWorldCPPEnv(
                size=size,
                obs_quantity=cfg["obs_quantity"],
                max_steps=cfg["max_steps"],
                shaping_enabled=False,
            )
            env.reset(seed=seed + i)
            r = connectivity_ratio(env)
            conn_means.append(r)
            if r >= 0.9999:
                conn_full += 1

        # Oracle (partial-vis greedy)
        ora_full = 0
        ora_covs = []
        ora_steps = []
        for i in range(episodes):
            cov, steps, full = oracle_episode(size, seed=seed + i)
            ora_covs.append(cov)
            ora_steps.append(steps)
            if full:
                ora_full += 1

        entry = {
            "size": size,
            "episodes": episodes,
            "seed_start": seed,
            "connectivity_perfect_pct": 100 * conn_full / episodes,
            "connectivity_mean_pct": 100 * sum(conn_means) / len(conn_means),
            "oracle_full_pct": 100 * ora_full / episodes,
            "oracle_cov_mean_pct": 100 * sum(ora_covs) / len(ora_covs),
            "oracle_steps_mean": sum(ora_steps) / len(ora_steps),
        }
        summary.append(entry)
        print(f"\n=== size={size} ===")
        print(f"  Connectivity perfect : {entry['connectivity_perfect_pct']:.1f}% "
              f"(mean reachable {entry['connectivity_mean_pct']:.2f}%)")
        print(f"  Oracle full coverage : {entry['oracle_full_pct']:.1f}% "
              f"(mean cov {entry['oracle_cov_mean_pct']:.2f}%, "
              f"avg steps {entry['oracle_steps_mean']:.0f})")

    if out_path:
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    run(args.sizes, args.episodes, args.seed, args.out)


if __name__ == "__main__":
    main()
