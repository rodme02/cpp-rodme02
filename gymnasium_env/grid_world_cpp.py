from collections import deque
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym

import pygame

# Coverage Path Planning environment.
#
# Observation (Dict, todos com shape fixo independente do tamanho do grid):
#   local_map  (3, 7, 7)   — janela egocêntrica one-hot: obstáculo/visitada/livre
#   global_map (2, 8, 8)   — memória pooleada: visitadas (max-pool) + posição do agente
#   coverage   (1,)        — fração de células livres visitadas
#   frontier   (3,)        — Δx, Δy, distância BFS (normalizadas) à fronteira mais próxima
#
# Reward base: +1 nova / -0.3 revisita / -0.5 colisão / -0.1 step / +10 cobertura completa / -5 truncamento.
# Reward shaping potential-based (Ng et al. 1999) com φ(s) = -d_BFS(agente, fronteira),
# ativado por ``shaping_enabled`` (default True): r' = r + γ φ(s') - φ(s).

class GridWorldCPPEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # Half-window of the egocentric local map. K = 2*WINDOW_RADIUS + 1.
    # Radius 3 -> 7x7 patch. Tunable via constructor argument.
    DEFAULT_WINDOW_RADIUS = 3

    # Fixed-resolution side of the global pooled memory map. Independent of
    # the grid size to preserve transfer between sizes.
    GLOBAL_MAP_SIDE = 8

    def __init__(
        self,
        render_mode: Optional[str] = None,
        size: int = 5,
        obs_quantity: int = 3,
        max_steps: int = 200,
        window_radius: int = DEFAULT_WINDOW_RADIUS,
        shaping_enabled: bool = True,
        shaping_gamma: float = 0.995,
    ):
        self.size = size
        self.window_size = 512
        self.obs_quantity = obs_quantity
        self.obstacles_locations = []
        self.count_steps = 0
        self.max_steps = max_steps

        self.window_radius = window_radius
        self.K = 2 * window_radius + 1  # local map side

        # Reward shaping (potential-based, Ng et al. 1999).
        self.shaping_enabled = shaping_enabled
        self.shaping_gamma = shaping_gamma

        # Track visited cells
        self.visited = set()

        self._agent_location = np.array([-1, -1], dtype=int)
        self._frontier_info = (0.0, 0.0, 0.0, 0)  # dx, dy, dist_norm, dist_raw
        self._potential_cached = 0.0

        # Observation:
        #   local_map  (3, K, K)        -> fine-grained egocentric view
        #   global_map (2, F, F)        -> coarse persistent memory at fixed F
        #   coverage   (1,)             -> overall coverage scalar
        self.F = self.GLOBAL_MAP_SIDE
        self.observation_space = gym.spaces.Dict({
            "local_map": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(3, self.K, self.K),
                dtype=np.float32,
            ),
            "global_map": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(2, self.F, self.F),
                dtype=np.float32,
            ),
            "coverage": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32,
            ),
            "frontier": gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(3,),
                dtype=np.float32,
            ),
        })

        # 4 actions: right, up, left, down
        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),   # right
            1: np.array([0, -1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, 1]),   # down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        # Pre-compute an obstacles set for O(1) membership tests in step().
        self._obstacles_set: set = set()

    @property
    def total_free_cells(self) -> int:
        return self.size * self.size - len(self.obstacles_locations)

    @property
    def coverage_ratio(self) -> float:
        return len(self.visited) / self.total_free_cells if self.total_free_cells > 0 else 1.0

    def _compute_frontier_info(self) -> Tuple[float, float, float, float]:
        """BFS from the agent to the nearest **frontier** cell.

        A frontier cell is an unvisited free cell that has at least one
        visited neighbor — exactly the cells the agent should head toward
        if it wants to expand its known region. BFS proceeds over free
        cells (visited cells are traversable; obstacles are blocked).

        Returns ``(dx_norm, dy_norm, dist_norm, dist_raw)``:
          - ``dx_norm``, ``dy_norm``: signed offsets to the target,
            normalized by ``size`` and clamped to [-1, 1].
          - ``dist_norm``: BFS distance, normalized by 2*size, clamped to 1.
          - ``dist_raw``: raw BFS distance in cells; ``inf`` if no frontier
            is reachable (full coverage or fully isolated).
        """
        if len(self.visited) >= self.total_free_cells:
            return 0.0, 0.0, 0.0, float("inf")

        size = self.size
        obstacles = self._obstacles_set
        visited = self.visited

        ax, ay = int(self._agent_location[0]), int(self._agent_location[1])

        queue = deque()
        queue.append((ax, ay, 0))
        seen = {(ax, ay)}

        target = None
        target_dist = 0
        # Adjacent visited cell makes the current cell a frontier (must also
        # be free and unvisited itself). We expand BFS over free terrain.
        while queue:
            cx, cy, d = queue.popleft()

            if (cx, cy) not in visited and (cx, cy) not in obstacles:
                # Frontier check: any of its 4-neighbors is in visited?
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    if (cx + dx, cy + dy) in visited:
                        target = (cx, cy)
                        target_dist = d
                        break
                if target is not None:
                    break

            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if (0 <= nx < size and 0 <= ny < size
                        and (nx, ny) not in obstacles
                        and (nx, ny) not in seen):
                    seen.add((nx, ny))
                    queue.append((nx, ny, d + 1))

        if target is None:
            return 0.0, 0.0, 0.0, float("inf")

        tx, ty = target
        dx_norm = max(-1.0, min(1.0, (tx - ax) / size))
        dy_norm = max(-1.0, min(1.0, (ty - ay) / size))
        dist_norm = min(1.0, target_dist / (2 * size))
        return dx_norm, dy_norm, dist_norm, float(target_dist)

    def _potential(self) -> float:
        """φ(s) = -d_BFS(agent, frontier). Negative because closer = better.

        Uses the cached ``self._frontier_info`` so callers must update it
        first. Returns 0 when no frontier exists (terminal/absorbing).
        """
        if not self.shaping_enabled:
            return 0.0
        dist_raw = self._frontier_info[3]
        if dist_raw == float("inf"):
            return 0.0
        return -dist_raw

    def _build_global_map(self) -> np.ndarray:
        """Pooled persistent memory at fixed resolution (2, F, F).

        Channel 0: visited mask, max-pooled — a pooled cell is 1 if any
                   source cell mapping to it has been visited.
        Channel 1: agent position, one-hot at the pooled cell currently
                   containing the agent.

        Source cells (sx, sy) are mapped to pooled coordinates via integer
        ratio: ti = sy * F // size, tj = sx * F // size. This gives a
        nearest-neighbor pooling that preserves "visited" flags and is
        fully size-invariant in output shape.
        """
        F = self.F
        size = self.size
        out = np.zeros((2, F, F), dtype=np.float32)

        for (sx, sy) in self.visited:
            ti = (sy * F) // size
            tj = (sx * F) // size
            out[0, ti, tj] = 1.0

        ax, ay = int(self._agent_location[0]), int(self._agent_location[1])
        out[1, (ay * F) // size, (ax * F) // size] = 1.0
        return out

    def _build_local_map(self) -> np.ndarray:
        """Egocentric KxK one-hot tensor centered on the agent.

        Out-of-bounds cells are encoded as obstacles (channel 0 = 1). Inside
        the grid, exactly one channel is 1 per cell — this provides a clean
        signal for the CNN and avoids the value-overloading issue of the
        previous {0, 1, 2} encoding.
        """
        K = self.K
        r = self.window_radius
        local = np.zeros((3, K, K), dtype=np.float32)

        ax, ay = int(self._agent_location[0]), int(self._agent_location[1])
        for di in range(-r, r + 1):       # row offset (y)
            for dj in range(-r, r + 1):   # col offset (x)
                i = di + r  # row index in patch
                j = dj + r  # col index in patch
                gx, gy = ax + dj, ay + di  # cell in world coords
                if not (0 <= gx < self.size and 0 <= gy < self.size):
                    local[0, i, j] = 1.0  # out of bounds -> obstacle channel
                elif (gx, gy) in self._obstacles_set:
                    local[0, i, j] = 1.0
                elif (gx, gy) in self.visited:
                    local[1, i, j] = 1.0
                else:
                    local[2, i, j] = 1.0
        return local

    def _get_obs(self) -> dict:
        dx, dy, dist_norm, _ = self._frontier_info
        return {
            "local_map": self._build_local_map(),
            "global_map": self._build_global_map(),
            "coverage": np.array([self.coverage_ratio], dtype=np.float32),
            "frontier": np.array([dx, dy, dist_norm], dtype=np.float32),
        }

    def _get_info(self) -> dict:
        return {
            "coverage": self.coverage_ratio,
            "visited_cells": len(self.visited),
            "total_free_cells": self.total_free_cells,
            "steps": self.count_steps,
            "size": self.size,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.count_steps = 0
        self.obstacles_locations = []
        self._obstacles_set = set()
        self.visited = set()

        # Place agent randomly
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        agent_tuple = (int(self._agent_location[0]), int(self._agent_location[1]))

        # Place obstacles (avoid agent and duplicates)
        attempts = 0
        max_attempts = 1000
        while len(self.obstacles_locations) < self.obs_quantity and attempts < max_attempts:
            cand = self.np_random.integers(0, self.size, size=2, dtype=int)
            cand_tuple = (int(cand[0]), int(cand[1]))
            if cand_tuple != agent_tuple and cand_tuple not in self._obstacles_set:
                self.obstacles_locations.append(cand)
                self._obstacles_set.add(cand_tuple)
            attempts += 1

        # Mark starting position as visited
        self.visited.add(agent_tuple)

        # Initial frontier + potential (used by step() for shaping baseline).
        self._frontier_info = self._compute_frontier_info()
        self._potential_cached = self._potential()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        old_location = self._agent_location.copy()

        # Move agent (clip to grid bounds)
        new_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # If the agent would step into an obstacle, stay in place.
        new_tuple = (int(new_location[0]), int(new_location[1]))
        if new_tuple in self._obstacles_set:
            self._agent_location = old_location
        else:
            self._agent_location = new_location

        self.count_steps += 1

        # --- CPP Reward Function ---
        current_pos = (int(self._agent_location[0]), int(self._agent_location[1]))
        is_new_cell = current_pos not in self.visited
        stayed_in_place = np.array_equal(self._agent_location, old_location)

        reward = -0.1  # base step penalty

        if stayed_in_place:
            # Hitting wall or obstacle (also covers the no-op edge case where
            # the agent picks a direction that lands on its own cell — only
            # possible at borders / obstacles, where it costs the collision).
            reward -= 0.5
        elif is_new_cell:
            reward += 1.0
            self.visited.add(current_pos)
        else:
            reward -= 0.3

        # Termination on full coverage
        full_coverage = len(self.visited) >= self.total_free_cells
        terminated = full_coverage
        if full_coverage:
            reward += 10.0

        # Truncation on max steps
        if self.count_steps >= self.max_steps and not terminated:
            truncated = True
            reward -= 5.0
        else:
            truncated = False

        # Update frontier info AFTER state changes; then apply potential-
        # based shaping (Ng et al. 1999):
        #     r_shaped = r + γ · φ(s') − φ(s)
        # φ(s_terminal) := 0 for absorbing states (preserves optimality).
        self._frontier_info = self._compute_frontier_info()
        if terminated or truncated:
            phi_next = 0.0
        else:
            phi_next = self._potential()
        if self.shaping_enabled:
            reward = reward + self.shaping_gamma * phi_next - self._potential_cached
        self._potential_cached = phi_next

        observation = self._get_obs()
        info = self._get_info()
        info["frontier_distance"] = self._frontier_info[3]

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size

        # Draw visited cells in light green
        for cell in self.visited:
            cell_arr = np.array(cell)
            pygame.draw.rect(
                canvas,
                (144, 238, 144),  # light green
                pygame.Rect(
                    pix_square_size * cell_arr,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Draw obstacles in black
        for obs in self.obstacles_locations:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    pix_square_size * obs,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Draw agent as blue circle
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Draw coverage info text
        font = pygame.font.SysFont(None, 24)
        coverage_text = font.render(
            f"Coverage: {self.coverage_ratio:.1%} | Steps: {self.count_steps}",
            True, (0, 0, 0)
        )
        canvas.blit(coverage_text, (5, 5))

        # Draw gridlines
        for x in range(self.size + 1):
            pygame.draw.line(canvas, 0, (0, pix_square_size * x),
                             (self.window_size, pix_square_size * x), width=3)
            pygame.draw.line(canvas, 0, (pix_square_size * x, 0),
                             (pix_square_size * x, self.window_size), width=3)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
