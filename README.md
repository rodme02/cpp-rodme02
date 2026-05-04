# Coverage Path Planning com PPO

APS final de Reinforcement Learning (Insper, 10º semestre, prof. Fabrício Barth). Enunciado: <https://insper.github.io/rl/classes/23_custom_env_agent/>.

> **📄 Relatório completo:** [`RELATORIO.md`](RELATORIO.md). O relatório descreve a estratégia adotada, a justificativa em conceitos de RL, os resultados obtidos e a análise.

## Resultados (resumo)

100 episódios, sementes fixas 10000–10099, política estocástica. Detalhes e tabelas completas em [`RELATORIO.md`](RELATORIO.md).

| Tamanho | Full coverage | Cobertura média |
|---|---|---|
| 5×5 | **95.0 %** | 99.41 % |
| 10×10 | **91.0 %** | 99.86 % |
| 20×20 (bônus) | **80.0 %** | **99.93 %** |

## Estrutura

```
.
├── README.md                    # este arquivo
├── RELATORIO.md                 # relatório técnico completo
├── requirements.txt
├── gymnasium_env/
│   ├── grid_world_cpp.py        # ambiente CPP custom
│   └── cpp_policy.py            # CNN feature extractor (custom)
├── train_grid_world_cpp.py      # train | curriculum | test | run
├── run_grid_world_cpp.py        # demo do env com agente aleatório
├── evaluate.py                  # avaliação reproduzível
├── make_plots.py                # curves | bars | ablation | all
├── data/                        # checkpoints (gitignored, gerados no treino)
├── log/                         # tensorboard + CSV (gitignored)
└── results/
    ├── eval_final_{stoch,det}.json
    └── figures/*.png
```

## Como executar

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Currículo completo: 5×5 (1M) → 10×10 (4M) → 20×20 (8M) com transfer entre estágios
python train_grid_world_cpp.py curriculum --n-envs 8 --seed 42

# Avaliação dos 3 modelos em 100 episódios (seeds fixas, reproduzível bit-a-bit)
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

Para iteração rápida use `--total-multiplier 0.25` no `curriculum` (1/4 dos timesteps).

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
