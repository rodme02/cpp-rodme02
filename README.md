# Coverage Path Planning com PPO

Agente que aprende a cobrir todas as células livres de um grid com obstáculos, sob visibilidade parcial (janela 5×5 egocêntrica), usando PPO + CNN com dois streams (local + global pooleado) + currículo crescente em tamanho de grid (5×5 → 10×10 → 20×20) com transfer entre estágios, potential-based reward shaping, e rejection sampling de layouts conectados no `reset()`.

## Relatório

A descrição da estratégia, justificativa em conceitos de RL, resultados e análise estão em **[`RELATORIO.md`](RELATORIO.md)**. A revisão de literatura que sustenta a recipe atual está em [`RELATORIO_LITERATURA.md`](RELATORIO_LITERATURA.md).

## Resultados

100 episódios, sementes fixas 10000–10099, política estocástica.

| Tamanho | Full coverage (com rejection) | Steps | Full coverage (legacy, sem rejection) |
|---|---|---|---|
| 5×5 | **100.0 %** | 23.1 | 97.0 % (= teto estrutural) |
| 10×10 | **100.0 %** | 99.0 | 92.0 % (= teto estrutural) |
| 20×20 | **100.0 %** | 530.1 | 80.0 % (= teto estrutural) |

A coluna "legacy" mostra os mesmos checkpoints avaliados sem rejection sampling — distribuição idêntica à do upstream `gym_custom_env`. A política bate exatamente o teto estrutural identificado por oracle perfect-info (§4.4 do RELATORIO), confirmando que o ganho de 100/100/100 vem da remoção dos layouts irresolúveis, não de uma melhoria artificial.

## Estrutura

```
.
├── README.md                    # este arquivo
├── RELATORIO.md                 # relatório técnico
├── RELATORIO_LITERATURA.md      # revisão de literatura
├── requirements.txt
├── gymnasium_env/
│   ├── grid_world_cpp.py        # ambiente CPP custom (com rejection sampling)
│   └── cpp_policy.py            # CNN dual-stream feature extractor
├── train_grid_world_cpp.py      # train | curriculum | no-curriculum | test | run
├── plasticity_callback.py       # diagnósticos de plasticidade
├── evaluate.py                  # avaliação primária
├── evaluate_cross.py            # cross-evaluation (modelo × tamanho)
├── oracle.py                    # oracle frontier-following + connectivity check
├── oracle_sweep.py              # sweep do teto estrutural por densidade
├── make_plots.py                # curves | bars | plasticity | cross | all
├── data/                        # checkpoints (gerados no treino)
├── log/                         # tensorboard + CSV (gerados no treino)
└── results/
    ├── eval_*.json
    ├── oracle_*.json
    └── figures/*.png
```

## Como executar

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Currículo completo 5×5 (1M) → 10×10 (4M) → 20×20 (8M, com obstacle randomization)
python train_grid_world_cpp.py curriculum --n-envs 8 --seed 42

# Avaliação 100 episódios cada tamanho
STAGE1=data/ppo_cpp_5_3_100_1000000_20260508_001352_stage1.zip
STAGE2=data/ppo_cpp_10_12_600_4000000_20260508_003138_stage2.zip
STAGE3=data/ppo_cpp_20_50_2400_8000000_20260508_015056_stage3.zip

python evaluate.py \
  --pair 5  "$STAGE1" --pair 10 "$STAGE2" --pair 20 "$STAGE3" \
  --episodes 100 --seed 10000 --out results/eval_runA_stoch.json

# Avaliação na distribuição legacy (sem rejection sampling) — transparência
python evaluate.py \
  --pair 5  "$STAGE1" --pair 10 "$STAGE2" --pair 20 "$STAGE3" \
  --episodes 100 --seed 10000 --no-enforce-connectivity \
  --out results/eval_runA_legacy_dist.json

# Cross-evaluation (cada modelo em todos os tamanhos, detecta forgetting)
python evaluate_cross.py \
  --models "$STAGE1" "$STAGE2" "$STAGE3" \
  --episodes 100 --seed 10000 --out results/cross_eval_runA.json

# Oracle (sanity check do teto estrutural)
python oracle.py --sizes 5 10 20 --episodes 100 --seed 10000

# Gráficos
python make_plots.py all \
  --log-dirs log/ppo_cpp_5_*_stage1 log/ppo_cpp_10_*_stage2 log/ppo_cpp_20_*_stage3 \
  --eval-json results/eval_runA_stoch.json \
  --cross-json results/cross_eval_runA.json
```

Para iteração rápida use `--total-multiplier 0.25` (1/4 dos timesteps).

Para visualizar um episódio do agente treinado:

```bash
python train_grid_world_cpp.py run --size 10 --model data/<modelo>.zip --deterministic
```

## Renderização

- **Verde claro**: células já visitadas
- **Azul (círculo)**: posição atual do agente
- **Preto**: obstáculos
- **Branco**: células livres ainda não visitadas
- **Texto no topo**: cobertura atual e número de passos
