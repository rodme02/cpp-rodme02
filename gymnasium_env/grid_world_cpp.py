from collections import deque
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym

import pygame

# Coverage Path Planning environment.
#
# Visibilidade parcial: o agente só vê uma janela 5x5 egocêntrica e o que ele
# já viu antes (memória persistente). Em particular, todos os cálculos
# derivados (frontier BFS, global_map) usam APENAS o conjunto
# ``_seen_obstacles`` — obstáculos que entraram na janela do agente em algum
# step anterior — nunca o conjunto completo de obstáculos do grid.
#
# Observation (Dict, todos com shape fixo independente do tamanho do grid):
#   local_map  (3, 5, 5)   — janela egocêntrica one-hot: obstáculo/visitada/livre
#   global_map (2, 8, 8)   — memória pooleada: visitadas (max-pool) + posição do agente
#   coverage   (1,)        — fração de células livres visitadas
#   frontier   (3,)        — Δx, Δy, distância BFS (normalizadas) à fronteira mais próxima
#   progress   (1,)        — count_steps / max_steps (orçamento de tempo restante)
#   trail      (8, 2)      — últimas 8 posições normalizadas (-1 quando ainda sem histórico)
#
# Reward base: +1 nova / -0.3 revisita / -0.5 colisão / -0.1 step / +10 cobertura completa / -5 truncamento.
# Reward shaping potential-based (Ng et al. 1999) com φ(s) = -d_BFS(agente, fronteira),
# ativado por ``shaping_enabled`` (default True): r' = r + γ φ(s') - φ(s).

class GridWorldCPPEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # Half-window of the egocentric local map. K = 2*WINDOW_RADIUS + 1.
    # Radius 2 -> 5x5 patch (regra do exercício: agente vê grid 5x5 com ele no centro).
    DEFAULT_WINDOW_RADIUS = 2

    # Fixed-resolution side of the global pooled memory map (size-invariant).
    GLOBAL_MAP_SIDE = 8

    # Comprimento do trail (últimas K posições do agente expostas na obs).
    TRAIL_LEN = 8

    def __init__(
        self,
        render_mode: Optional[str] = None,
        size: int = 5,
        obs_quantity: int = 3,
        max_steps: int = 200,
        window_radius: int = DEFAULT_WINDOW_RADIUS,
        shaping_enabled: bool = True,
        shaping_gamma: float = 0.995,
        shaping_scale: float = 1.0,
        obs_quantity_range: Optional[Tuple[int, int]] = None,
        enforce_connectivity: bool = True,
    ):
        self.size = size
        self.window_size = 512
        self.obs_quantity = obs_quantity
        self.obs_quantity_range = obs_quantity_range
        # Quando True, descarta layouts com bolsões inalcançáveis no reset()
        # (rejection sampling) — solução para o teto estrutural de §4.4.
        # Quando False, comportamento idêntico ao upstream original (qualquer
        # layout aleatório é aceito) — usado para validar que a política
        # treinada com rejection ainda atinge o teto estrutural na distribuição
        # legacy.
        self.enforce_connectivity = enforce_connectivity
        self.obstacles_locations = []
        self.count_steps = 0
        self.max_steps = max_steps
        self.count_revisits = 0

        self.window_radius = window_radius
        self.K = 2 * window_radius + 1  # local map side

        # Reward shaping (potential-based, Ng et al. 1999).
        self.shaping_enabled = shaping_enabled
        self.shaping_gamma = shaping_gamma
        self.shaping_scale = shaping_scale

        # Track visited cells
        self.visited = set()

        # Trail: deque das últimas TRAIL_LEN posições do agente (mais recente
        # à direita). Permite a política reconhecer e quebrar ciclos curtos
        # de re-visitação no end-game sem precisar de recorrência.
        self.trail_len = self.TRAIL_LEN
        self._trail: deque = deque(maxlen=self.trail_len)

        self._agent_location = np.array([-1, -1], dtype=int)
        self._frontier_info = (0.0, 0.0, 0.0, 0)  # dx, dy, dist_norm, dist_raw
        self._potential_cached = 0.0

        # Observation:
        #   local_map  (3, K, K)        -> fine-grained egocentric view
        #   global_map (2, F, F)        -> coarse persistent memory at fixed F
        #   coverage   (1,)             -> overall coverage scalar
        #   frontier   (3,)             -> Δx, Δy, dist normalizados
        #   progress   (1,)             -> orçamento (count_steps / max_steps)
        #   trail      (TRAIL_LEN, 2)   -> últimas L posições normalizadas (-1 = ausente)
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
            "progress": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32,
            ),
            "trail": gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.trail_len, 2),
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

        # Conjunto completo de obstáculos (usado só pela física do step e pelo
        # render — nunca pela observação ou pelo BFS).
        self._obstacles_set: set = set()
        # Obstáculos que o agente JÁ viu (entraram na janela KxK em algum step).
        # Único conjunto consultado por _build_local_map e _compute_frontier_info.
        self._seen_obstacles: set = set()

    @property
    def total_free_cells(self) -> int:
        return self.size * self.size - len(self.obstacles_locations)

    @property
    def coverage_ratio(self) -> float:
        return len(self.visited) / self.total_free_cells if self.total_free_cells > 0 else 1.0

    def _update_seen_obstacles(self) -> None:
        """Adiciona ao ``_seen_obstacles`` todos os obstáculos dentro da janela
        KxK ao redor do agente. Chamado em ``reset`` e após cada movimento.

        É a única forma do agente "descobrir" obstáculos — antes de entrarem na
        janela em algum step, eles ficam ocultos para tudo que deriva da
        observação (BFS de fronteira, etc.).
        """
        r = self.window_radius
        ax, ay = int(self._agent_location[0]), int(self._agent_location[1])
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                gx, gy = ax + dj, ay + di
                if 0 <= gx < self.size and 0 <= gy < self.size and \
                        (gx, gy) in self._obstacles_set:
                    self._seen_obstacles.add((gx, gy))

    def _compute_frontier_info(self) -> Tuple[float, float, float, float]:
        """BFS do agente até a célula de **fronteira** mais próxima.

        Fronteira = célula livre não-visitada com pelo menos um vizinho
        visitado. A BFS expande sobre o "terreno conhecido" do agente:
        bloqueia apenas em obstáculos que o agente JÁ viu (``_seen_obstacles``).
        Células nunca observadas são tratadas como potencialmente livres
        (otimismo sob incerteza) — exatamente o que o enunciado exige ao
        proibir acesso ao mapa completo.

        Retorna ``(dx_norm, dy_norm, dist_norm, dist_raw)``.
        """
        if len(self.visited) >= self.total_free_cells:
            return 0.0, 0.0, 0.0, float("inf")

        size = self.size
        seen_obstacles = self._seen_obstacles  # Apenas obstáculos vistos.
        visited = self.visited

        ax, ay = int(self._agent_location[0]), int(self._agent_location[1])

        queue = deque()
        queue.append((ax, ay, 0))
        seen = {(ax, ay)}

        target = None
        target_dist = 0
        while queue:
            cx, cy, d = queue.popleft()

            if (cx, cy) not in visited and (cx, cy) not in seen_obstacles:
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
                        and (nx, ny) not in seen_obstacles
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
        return -self.shaping_scale * dist_raw

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
        """Tensor egocêntrico KxK one-hot centrado no agente.

        Out-of-bounds vira canal 0 (obstáculo). Dentro do grid, exatamente um
        canal é 1 por célula — encoding linearmente separável para a CNN.
        Obstáculos são lidos do mapa real (``_obstacles_set``) porque, por
        definição, o agente "vê" tudo dentro da janela KxK (sensor imediato).
        Mas como esses obstáculos também são adicionados a ``_seen_obstacles``
        em ``_update_seen_obstacles``, nenhuma informação adicional é exposta:
        o BFS e os outros cálculos consultam apenas a memória persistente.
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

    def _build_trail(self) -> np.ndarray:
        """Trail right-aligned das últimas TRAIL_LEN posições do agente.

        Posições normalizadas por ``size``. Slots vazios (deque ainda não
        cheia no início do episódio) ficam com -1 (sentinela fora do range
        [0,1]). A política pode ler trail[-1] como "posição corrente" e
        trail[-2..] como "trajetória recente", permitindo reconhecer ciclos
        curtos sem precisar de recorrência (LSTM).
        """
        L = self.trail_len
        size_norm = max(1, self.size)
        out = np.full((L, 2), -1.0, dtype=np.float32)
        n = len(self._trail)
        for i, (px, py) in enumerate(self._trail):
            slot = L - n + i  # right-align: mais recente em out[-1]
            out[slot, 0] = px / size_norm
            out[slot, 1] = py / size_norm
        return out

    def _get_obs(self) -> dict:
        dx, dy, dist_norm, _ = self._frontier_info
        progress = (self.count_steps / max(1, self.max_steps)) if self.max_steps > 0 else 0.0
        return {
            "local_map": self._build_local_map(),
            "global_map": self._build_global_map(),
            "coverage": np.array([self.coverage_ratio], dtype=np.float32),
            "frontier": np.array([dx, dy, dist_norm], dtype=np.float32),
            "progress": np.array([min(1.0, progress)], dtype=np.float32),
            "trail": self._build_trail(),
        }

    def _get_info(self) -> dict:
        return {
            "coverage": self.coverage_ratio,
            "visited_cells": len(self.visited),
            "total_free_cells": self.total_free_cells,
            "steps": self.count_steps,
            "size": self.size,
            "revisits": self.count_revisits,
            "repeat_ratio": (self.count_revisits / self.count_steps
                             if self.count_steps > 0 else 0.0),
        }

    def _all_free_cells_reachable(self, agent_tuple: Tuple[int, int]) -> bool:
        """BFS sobre o mapa real (perfect information): todas as células
        livres são alcançáveis a partir de ``agent_tuple``?

        Usado em ``reset`` para descartar (rejection sampling) layouts onde
        o sorteio aleatório de obstáculos cria bolsões inalcançáveis. Sob
        visibilidade parcial, esses layouts impõem um teto estrutural à
        cobertura — irrespectivo da política, nenhum agente pode atingir
        100%. Descartá-los na geração mantém o problema bem-posto.
        """
        size = self.size
        obstacles = self._obstacles_set
        total_free = size * size - len(obstacles)
        queue = deque([agent_tuple])
        seen = {agent_tuple}
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if (0 <= nx < size and 0 <= ny < size
                        and (nx, ny) not in obstacles
                        and (nx, ny) not in seen):
                    seen.add((nx, ny))
                    queue.append((nx, ny))
        return len(seen) >= total_free

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.count_steps = 0
        self.count_revisits = 0
        self._seen_obstacles = set()
        self.visited = set()
        self._trail.clear()

        # Domain randomization opcional: amostra obs_quantity em cada episódio
        # dentro de obs_quantity_range (inclusive). Diversifica distribuição de
        # treino e reduz overfit a uma densidade específica de obstáculos.
        if self.obs_quantity_range is not None:
            lo, hi = self.obs_quantity_range
            target_quantity = int(self.np_random.integers(lo, hi + 1))
        else:
            target_quantity = self.obs_quantity

        # Rejection sampling (quando enforce_connectivity=True): gera layouts
        # até encontrar um onde TODAS as células livres sejam alcançáveis a
        # partir da posição do agente. Layouts com bolsões impõem teto
        # estrutural (oracle perfect-info bate exatamente o mesmo platô que
        # o RL), então mantê-los na distribuição apenas adiciona ruído
        # indistinguível de overfit ao problema.
        # Quando enforce_connectivity=False, comportamento idêntico ao
        # upstream original (qualquer layout é aceito).
        max_layout_resamples = 200 if self.enforce_connectivity else 1
        agent_tuple = (-1, -1)
        for resample in range(max_layout_resamples):
            self.obstacles_locations = []
            self._obstacles_set = set()

            # Place agent randomly (re-sample em cada attempt para preservar
            # diversidade de starting positions sob aceitação por conectividade).
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            agent_tuple = (int(self._agent_location[0]), int(self._agent_location[1]))

            # Place obstacles (avoid agent and duplicates)
            attempts = 0
            max_attempts = 1000
            while len(self.obstacles_locations) < target_quantity and attempts < max_attempts:
                cand = self.np_random.integers(0, self.size, size=2, dtype=int)
                cand_tuple = (int(cand[0]), int(cand[1]))
                if cand_tuple != agent_tuple and cand_tuple not in self._obstacles_set:
                    self.obstacles_locations.append(cand)
                    self._obstacles_set.add(cand_tuple)
                attempts += 1

            if not self.enforce_connectivity:
                break
            if self._all_free_cells_reachable(agent_tuple):
                break
        # Se max_layout_resamples for esgotado (extremamente raro nos ranges
        # usados), aceita o último layout — é melhor que travar; o agente
        # apenas sofrerá o teto natural daquele layout.

        # Trail começa com a posição inicial do agente.
        self._trail.append(agent_tuple)

        # Mark starting position as visited
        self.visited.add(agent_tuple)

        # Atualiza obstáculos vistos com o que está na janela inicial do agente
        # ANTES de qualquer cálculo derivado (frontier BFS, etc.).
        self._update_seen_obstacles()

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
        self._trail.append((int(self._agent_location[0]),
                            int(self._agent_location[1])))

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
            self.count_revisits += 1

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

        # Após mover, atualiza obstáculos vistos com o que está na nova janela.
        self._update_seen_obstacles()

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

        # Coverage info overlay (skip silently if pygame.font is unavailable).
        try:
            font = pygame.font.SysFont(None, 24)
            coverage_text = font.render(
                f"Coverage: {self.coverage_ratio:.1%} | Steps: {self.count_steps}",
                True, (0, 0, 0)
            )
            canvas.blit(coverage_text, (5, 5))
        except (NotImplementedError, AttributeError):
            pass

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
