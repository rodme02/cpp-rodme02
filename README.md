# Coverage Path Planning com PPO

Agente que aprende a cobrir todas as células livres de um grid com obstáculos, sob visibilidade parcial, usando PPO + CNN egocêntrica + currículo crescente em tamanho de grid (5×5 → 10×10 → 20×20) com transfer entre estágios e potential-based reward shaping.

## Relatório

A descrição da estratégia, justificativa em conceitos de RL, resultados e análise estão em **[`RELATORIO.md`](RELATORIO.md)**. A revisão de literatura que sustenta a recipe atual está em [`RELATORIO_LITERATURA.md`](RELATORIO_LITERATURA.md).

## Resultados (resumo)

100 episódios, sementes fixas 10000–10099, política estocástica.

| Tamanho | Full coverage | Cobertura média | Steps médios |
|---|---|---|---|
| 5×5 | **97.0 %** | 99.82 % | 30.5 |
| 10×10 | **92.0 %** | 99.89 % | 153.0 |
| 20×20 | **81.0 %** | 99.93 % | 957.7 |

## Estrutura

```
.
├── README.md                    # este arquivo
├── RELATORIO.md                 # relatório técnico
├── RELATORIO_LITERATURA.md      # revisão de literatura
├── requirements.txt
├── gymnasium_env/
│   ├── grid_world_cpp.py        # ambiente CPP custom
│   └── cpp_policy.py            # CNN feature extractor
├── train_grid_world_cpp.py      # train | curriculum | no-curriculum | test | run
├── plasticity_callback.py       # diagnósticos de plasticidade
├── evaluate.py                  # avaliação primária
├── evaluate_cross.py            # cross-evaluation (modelo × tamanho)
├── make_plots.py                # curves | bars | ablation | plasticity | cross | all
├── data/                        # checkpoints (gerados no treino)
├── log/                         # tensorboard + CSV (gerados no treino)
└── results/
    ├── eval_*.json
    └── figures/*.png
```

## Como executar

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Currículo completo 5×5 (1M) → 10×10 (4M) → 20×20 (8M)
python train_grid_world_cpp.py curriculum --n-envs 8 --seed 42

# Avaliação 100 episódios cada tamanho
python evaluate.py \
  --pair 5  data/<stage1>.zip \
  --pair 10 data/<stage2>.zip \
  --pair 20 data/<stage3>.zip \
  --episodes 100 --seed 10000 --out results/eval_final_stoch.json

# Cross-evaluation (cada modelo em todos os tamanhos, detecta forgetting)
python evaluate_cross.py \
  --models data/<stage1>.zip data/<stage2>.zip data/<stage3>.zip \
  --episodes 100 --seed 10000 --out results/cross_eval.json

# Gráficos
python make_plots.py all \
  --log-dirs log/<stage1> log/<stage2> log/<stage3> \
  --eval-json results/eval_final_stoch.json
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
