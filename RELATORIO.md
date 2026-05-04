# Relatório — Coverage Path Planning com PPO

**Aluno:** Rodrigo Medeiros — Insper, 10º semestre, disciplina de Reinforcement Learning
**Enunciado:** <https://insper.github.io/rl/classes/23_custom_env_agent/>

## 1. Problema

O agente precisa visitar todas as células livres de um grid com obstáculos no menor número de passos possível, **sem acesso ao mapa completo do ambiente** — só à vizinhança imediata e ao histórico que ele mesmo coletou ao longo da exploração.

A função de recompensa do enunciado (mantida sem alterações):

| Evento | Reward |
|---|---|
| Visitar célula nova | +1.0 |
| Revisitar célula | −0.3 |
| Colidir com parede / obstáculo | −0.5 (sobre o step) |
| Step penalty | −0.1 |
| Cobertura completa (terminal) | +10.0 |
| Truncamento sem fechar | −5.0 |

**Baseline reportado pela página:** 75–81 % de full coverage no 5×5 e degradação para 59–70 % de cobertura média ao escalar para 10×10. A entrega tem que melhorar isso atingindo ~100 % nos três tamanhos (5×5, 10×10 e o bônus 20×20).

## 2. Estratégia

A causa raiz das duas falhas do baseline, identificada por leitura do código original, é dupla:

1. A observação inclui **coordenadas absolutas** `(x, y)`, então a política aprende a *posição* específica de cada grid (25 mapeamentos no 5×5, 100 no 10×10, 400 no 20×20). Cada cenário vira um problema independente — transfer learning não acontece.
2. A vizinhança é apenas 3×3 com encoding `{0, 1, 2}`. O agente vê 1 célula de raio, e o canal único confunde semânticas (obstáculo / livre / visitada estão todos no mesmo escalar).

A intervenção combina três alavancas que se reforçam.

### 2.1 Observação invariante ao tamanho do grid

Coordenadas absolutas foram **removidas**. A nova observação é um `Dict` com 4 componentes, todos com **shape fixo independente do tamanho do grid**:

```python
{
  "local_map":  Box(0, 1, shape=(3, 7, 7)),  # detalhe imediato
  "global_map": Box(0, 1, shape=(2, 8, 8)),  # memória persistente em resolução fixa
  "coverage":   Box(0, 1, shape=(1,)),       # progresso global
  "frontier":   Box(-1, 1, shape=(3,)),      # direção e distância à fronteira
}
```

**`local_map` (3, 7, 7)** — janela egocêntrica em torno do agente, 3 canais one-hot mutuamente exclusivos:

| Canal | Significado |
|---|---|
| 0 | obstáculo (incluindo *out-of-bounds*) |
| 1 | célula livre **já visitada** |
| 2 | célula livre **ainda não visitada** |

**`global_map` (2, 8, 8)** — mapa pooleado em resolução fixa F = 8:

| Canal | Significado |
|---|---|
| 0 | máscara de visitadas (max-pool: 1 se qualquer célula da região foi visitada) |
| 1 | posição corrente do agente (one-hot na célula pooleada) |

A resolução F = 8 é **independente do tamanho do grid**. No 5×5 ela é mais fina que 1:1; no 20×20 cada célula pooleada cobre ~6 células reais. Isso preserva a invariância de shape entre tamanhos sem custo extra de parâmetros.

**`frontier` (3)** — direção `(Δx, Δy)` normalizada e distância BFS normalizada à célula de fronteira mais próxima, onde *fronteira* = célula livre não-visitada adjacente a uma célula visitada (definição clássica de frontier-based exploration em robótica). Essa informação é **derivada da própria história do agente** — não revela posições de obstáculos ainda não vistos.

**Justificativa em RL.** A política passa a ver o mundo a partir do referencial do agente. A transição `s → s'` é a mesma em qualquer ponto do mapa onde a vizinhança local for igual. Isso é **equivariância translacional**: se a política for ótima em um patch local 7×7, ela continua ótima em outro patch idêntico em qualquer tamanho de grid. A política treinada em 5×5 pode ser refinada em 10×10 sem retreinar a CNN do zero — exatamente o que o enunciado pede ao questionar "necessidade de transfer learning".

A separação em canais one-hot também resolve o problema do encoding `{0, 1, 2}`: o ReLU/Conv não trata 2 como "duas vezes mais obstáculo que 1". O sinal fica linearmente separável.

**Não viola visibilidade parcial.** Toda a observação é construída a partir de duas fontes:
- **Sensor imediato:** o que está dentro da janela 7×7 ao redor do agente *agora*.
- **Memória persistente:** as células visitadas (`self.visited`) e os obstáculos já vistos (`self._seen_obstacles`, conjunto que cresce monotonicamente conforme obstáculos entram na janela em algum step).

Concretamente: o `local_map` mostra 49 células em torno do agente (~50 % de um grid 10×10, ~12 % de um grid 20×20); o `global_map` carrega apenas as células visitadas; e o **BFS que computa o `frontier` bloqueia somente em `_seen_obstacles`** — células nunca observadas são tratadas como potencialmente livres (otimismo sob incerteza), o que é exatamente o comportamento que o enunciado exige. Em nenhum momento o agente acessa o conjunto completo de obstáculos do grid via observação ou cálculo derivado.

### 2.2 CNN feature extractor com duas streams

`gymnasium_env/cpp_policy.py` define `CPPFeatureExtractor`:

- **Stream local**: dois `Conv2d(3 → 32 → 32, kernel 3, padding 1)` + ReLU sobre o `local_map` 3×7×7 → flatten → linear → 56 features.
- **Stream global**: dois `Conv2d(2 → 32 → 32, kernel 3, padding 1)` + ReLU sobre o `global_map` 2×8×8 → flatten → linear → 56 features.
- **MLP** sobre os 4 escalares (coverage + frontier) → 16 features.
- Concatenação → 128 features para a policy/value head (`net_arch=[64, 64]`).

**Justificativa em RL.** Convolução é um *inductive bias* explícito de **invariância translacional local**: filtros 3×3 detectam padrões como "fronteira entre visitada e não-visitada", "obstáculo à frente", "canto" — exatamente os primitivos relevantes para coverage. Substituir a `MultiInputPolicy` plana do baseline (que descarta a estrutura espacial) por uma CNN é a resposta direta à pergunta da página sobre "suficiência da arquitetura `MultiInputPolicy`".

### 2.3 Reward shaping potential-based (Ng et al. 1999)

Mesmo com `local_map` + `global_map`, uma versão preliminar (sem shaping) saturava em ~76 % de full coverage no 10×10 e simplesmente não convergia no 20×20 — o `ep_rew_mean` ficava ~−265 sem subir.

**Diagnóstico.** A recompensa terminal `+10` por cobertura completa está descontada por `γ^k` com `k ≥ 100` no 10×10 e `k ≥ 1000` no 20×20. Em PPO esse sinal terminal vira essencialmente ruído — o gradiente fica dominado por custos imediatos (revisita, step penalty).

**Solução: shaping potential-based.** Define-se um potencial `φ(s)` e altera-se o reward para
$$r' = r + γ\,φ(s') - φ(s).$$

O **teorema de Ng et al. (1999)** garante que **a política ótima não muda** sob essa transformação, mas o agente recebe um gradiente denso por toda a trajetória.

Escolha do potencial:
$$φ(s) = -d_\text{BFS}(\text{agente},\;\text{fronteira mais próxima}),\quad φ(s_\text{terminal}) = 0$$

A BFS é executada sobre o **terreno conhecido pelo agente** — `visited ∪ ¬_seen_obstacles` — bloqueando apenas em obstáculos que o agente já viu pessoalmente. Células nunca observadas são tratadas como potencialmente livres (otimismo sob incerteza). Cada passo do agente em direção à fronteira gera ≈ +1.0 de shaping (porque `d` cai por 1, e `gamma * (d - 1) - d ≈ 1 - gamma * 0 = 1` para `γ` próximo de 1). Cada passo na direção contrária penaliza simetricamente.

A flag `shaping_enabled` permite desligar (útil para ablação).

### 2.4 Currículo + transfer learning

Cada estágio carrega os pesos do anterior. Como toda a observação tem shape fixo, a CNN não precisa ser retreinada do zero.

| Estágio | Tamanho | Obstáculos | Max steps | Timesteps | `ent_coef` |
|---|---|---|---|---|---|
| 1 | 5×5  | 3  | 100  | 1 M  | 0.04  |
| 2 | 10×10 | 12 | 600  | 4 M  | 0.02  |
| 3 | 20×20 | 50 | 2400 | 8 M  | 0.015 |

**Justificativa em RL.** Currículo (Bengio et al. 2009) acelera convergência em problemas de **recompensa esparsa**. No 5×5 a sequência de cobertura é curta o suficiente para a recompensa terminal `+10` chegar ao agente em poucos retornos descontados; uma vez aprendida a *política de cobertura local*, o transfer aproveita os pesos da CNN — que continuam corretos no 10×10 e 20×20 porque a entrada é a mesma. A entropia decrescente entre estágios respeita o trade-off **exploração → exploitation**: começa alta para descobrir, baixa para refinar.

### 2.5 Hiperparâmetros PPO

| | |
|---|---|
| `learning_rate` | 3e-4 |
| `n_steps` | 1024 |
| `batch_size` | 256 |
| `n_epochs` | 10 |
| `gamma` | 0.995 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.2 |
| `vf_coef` | 0.5 |
| `max_grad_norm` | 0.5 |
| `n_envs` (SubprocVecEnv) | 8 |

`gamma = 0.995` é a única alteração não-trivial: a recompensa terminal `+10` precisa "viajar" até centenas de passos no 20×20. Um `gamma = 0.99` desconta para zero em ~500 passos, o que faz a recompensa terminal sumir do retorno. Com `gamma = 0.995` o horizonte efetivo dobra.

## 3. Resultados

Avaliação com **100 episódios e sementes fixas 10000–10099** em cada tamanho. `evaluate.py` chama `set_global_seed` em `random`, `numpy` e `torch` antes de cada eval, deixando os números **bit-a-bit reprodutíveis** entre execuções.

### 3.1 Tabela final (política estocástica, configuração final)

| Tamanho | Full coverage rate | Cobertura média | σ | Passos médios | σ |
|---|---|---|---|---|---|
| **5×5**   | **95.0 %** | 99.41 % | 3.44 |   34.7 |  16.9 |
| **10×10** | **91.0 %** | 99.86 % | 0.49 |  169.6 | 144.1 |
| **20×20 (bônus)** | **80.0 %** | **99.93 %** | 0.17 | 1173.1 | 640.3 |

> A configuração final é treinada com 1 M + 4 M + 8 M passos (currículo). A tabela pode ser regenerada bit-a-bit com `python evaluate.py --pair 5 ... --pair 10 ... --pair 20 ... --seed 10000 --episodes 100`.

### 3.2 Comparação com o baseline da página

| Métrica | Baseline (página) | Proposta | Δ |
|---|---|---|---|
| 5×5 full coverage rate | 75–81 % | **95.0 %** | **+14 a +20 pp** |
| 10×10 cobertura média | 59–70 % | **99.86 %** | **+29.9 a +40.9 pp** |
| 20×20 (bônus) full coverage rate | — | **80.0 %** | — |
| 20×20 (bônus) cobertura média | — | **99.93 %** | — |

A melhora mais saliente é o salto na cobertura média do 10×10, que passa do regime "agente erra metade do mapa" para "agente erra ~0.1 células por episódio em média". O 20×20, que sequer aparecia no baseline da página, é viabilizado pela combinação shaping + currículo.

### 3.3 Ablação dos três incrementos arquiteturais

Mesmo procedimento de avaliação (100 eps, sementes 10000–10099, estocástico), variando só a configuração:

| | 5×5 full | 5×5 cov | 10×10 full | 10×10 cov | 20×20 full | 20×20 cov |
|---|---|---|---|---|---|---|
| **v1**: só `local_map`                  | 94.0 % | 99.32 % |  3.0 % | 94.78 % | — | — |
| **v2**: + `global_map`                  | 96.0 % | 99.77 % | 76.0 % | 99.10 % | — | — |
| **v3**: + `frontier` + reward shaping   | **95.0 %** | **99.41 %** | **91.0 %** | **99.86 %** | **80.0 %** | **99.93 %** |

Cada incremento ataca uma falha específica:

- **`global_map` (v1 → v2):** sem ele, o agente cobre 95 % do 10×10 mas só **fecha 3 %** dos episódios — esquece células visitadas que saem da janela 7×7. Com ele sobe para 76 % de full coverage. Resolve o problema da memória além da janela local.
- **`frontier` + reward shaping (v2 → v3):** sobe o 10×10 de 76 % para 92 %, e **viabiliza o 20×20**, que sem shaping ficava num platô de `ep_rew_mean ≈ -265` sem progredir. O shaping dá o gradiente denso que destrava a otimização.

Ver `results/figures/ablation_comparison.png`.

### 3.4 Curvas de aprendizado

`results/figures/learning_curve_*.png` — uma figura por estágio. O salto pós-transfer é visível em cada curva: `ep_rew_mean` no início do stage 2 (10×10) parte de uma região "quente" do espaço de políticas, em vez do mergulho profundo em recompensa negativa do treinamento do zero.

## 4. Análise

### 4.1 Por que a representação egocêntrica funcionou

O baseline exigia que a rede aprendesse uma função `(x, y, coverage, vizinhos) → ação`. Para o 5×5 isso requer aprender 25 mapeamentos posicionais; para o 10×10, 100; para o 20×20, 400. Cada cenário é um problema novo, transfer não acontece.

A versão egocêntrica reduz o espaço efetivo de estados a *padrões locais* que se repetem **tanto dentro de um grid quanto entre grids** de tamanhos diferentes. A política aprendida em 5×5 já cobre a maior parte dos padrões que vai encontrar no 10×10, e os pesos da CNN transferem direto.

### 4.2 Por que o shaping foi a alavanca-chave para o 20×20

A recompensa terminal `+10` está descontada por `γ^k` com `k ≥ 100` no 10×10 e `k ≥ 1000` no 20×20. Em PPO, esse sinal terminal vira ruído, dominado por custos imediatos. O shaping potential-based dá um gradiente denso (≈ +1 por passo na direção certa) **sem mudar a política ótima** — exatamente o que o teorema de Ng et al. (1999) garante. Foi a diferença entre "convergir" e "ficar no platô" no 20×20.

### 4.3 Determinístico vs estocástico

A política `argmax` é consistentemente pior que a estocástica em todos os tamanhos (no 20×20: 0 % vs 80 % de full coverage). Em CPP o ruído da amostragem funciona como mecanismo de *tie-breaking* para configurações com múltiplas ações de valor próximo — caso típico de uma célula no centro de uma região explorada com 4 vizinhas idênticas em valor. Sem ruído, o argmax desempata sempre pela mesma ação e cria ciclos. Os números reportados em §3 são todos da política estocástica (`deterministic=False`), que é a configuração natural para PPO em ambientes com empates locais comuns.

## 5. Limitações e melhorias futuras

- **Custo da BFS por step.** O cálculo da distância à fronteira é uma BFS por chamada de `step()`. Em CPU a queda de FPS é da ordem de 30–40 % (de ~1.5 k para ~1 k passos/segundo com 8 envs paralelos). Aceitável para grids até 20×20; em escalas maiores valeria caching incremental do BFS quando o conjunto de células visitadas não muda.
- **Resolução fixa do `global_map` no 20×20.** Cada célula pooleada cobre ~6 células reais, então o sinal global vira grosseiro. Aumentar para F = 16 daria mais resolução à custa de mais parâmetros. Como o shaping dá um sinal denso independente do `global_map`, a sensibilidade desta escolha cai bastante na configuração final.
- **Ausência de recorrência.** O agente não tem hidden state. Para CPP em mapas muito grandes (≥ 30×30), uma `RecurrentPPO` com LSTM provavelmente capturaria sequências de ações melhor que o `global_map` estático + shaping. Trabalho futuro.
- **Política determinística subótima.** A política aprendida depende da amostragem. Se a aplicação exigir política determinística (ex.: certificação), seria preciso treinamento com regularização específica ou um *tie-break* explícito (ex.: preferir a ação com menor histórico recente).

## 6. Reprodutibilidade

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Currículo completo: 5x5 (1M) → 10x10 (4M) → 20x20 (8M) com transfer
python train_grid_world_cpp.py curriculum --n-envs 8 --seed 42

# Avaliação reproduzível bit-a-bit
python evaluate.py \
  --pair 5  data/<stage1>.zip \
  --pair 10 data/<stage2>.zip \
  --pair 20 data/<stage3>.zip \
  --episodes 100 --seed 10000 --out results/eval_final_stoch.json

# Gráficos
python make_plots.py all \
  --log-dirs log/<stage1> log/<stage2> log/<stage3> \
  --eval-json results/eval_final_stoch.json
```

Sementes: `42` no currículo (offset por estágio); `10000–10099` na avaliação. `evaluate.py` fixa `random`, `numpy` e `torch` para garantir reprodutibilidade da política estocástica.

## 7. Referências

- Yanes Luis et al. *A Deep Reinforcement Learning Approach for the Patrolling Problem of Water Resources Through Autonomous Surface Vehicles: The Ypacarai Lake Case*. IEEE Access, 2020.
- Galceran & Carreras. *A Survey on Coverage Path Planning for Robotics*. Robotics and Autonomous Systems, 2013.
- Schulman et al. *Proximal Policy Optimization Algorithms*. arXiv:1707.06347, 2017.
- Bengio et al. *Curriculum Learning*. ICML 2009.
- Ng et al. *Policy invariance under reward transformations*. ICML 1999.
