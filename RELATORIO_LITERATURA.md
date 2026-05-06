# Relatório de Literatura — CPP com PPO + Currículo

Síntese da pesquisa em seis temas que tocam o projeto (PPO + CNN egocêntrica, currículo 5×5 → 10×10 → 20×20 com transfer entre estágios, PBRS via BFS de fronteira), com referências centrais e como cada uma conecta com decisões do código.

---

## 1. Loss of plasticity em deep RL

**Dohare, Hernandez-Garcia, Lan, Rahman, Mahmood, Sutton (2024). "Loss of plasticity in deep continual learning." *Nature* 632:768–774.** Mostram que treino sequencial em tarefas que mudam (incluindo PPO no Ant com fricção variável) leva a perda de plasticidade: gradientes encolhem, neurônios morrem, rede vira efetivamente shallow. Mitigações eficazes: L2 leve, continual backpropagation (reinicialização periódica de unidades pouco usadas), Shrink-and-Perturb. Adam tunado sozinho ajuda mas não resolve.

**Conexão:** currículo é um caso particular de continual learning. A run completa do projeto (13M passos, 3 mudanças de tarefa) está estruturalmente no cenário do paper.

### Diagnósticos modernos

- **Sokar et al. (ICML 2023). "The Dormant Neuron Phenomenon in Deep RL."** [arXiv 2302.12902](https://arxiv.org/abs/2302.12902). Define τ-dormant neuron; ratio cresce monotonamente em DRL e prevê colapso. Métrica padrão para CNNs ReLU.
- **Kumar et al. (ICLR 2021). "Implicit Under-Parameterization."** [arXiv 2010.14498](https://arxiv.org/abs/2010.14498). Effective rank das features colapsa sob bootstrapping mesmo sem dormant units visíveis.
- **Lyle et al. (ICML 2023; CoLLAs 2025).** [arXiv 2303.01486](https://arxiv.org/abs/2303.01486), [arXiv 2402.18762](https://arxiv.org/abs/2402.18762). Plasticity loss tem múltiplos mecanismos independentes (saturação, weight growth, curvatura); nenhuma intervenção isolada cobre todos.

### Mitigações relevantes para PPO+CNN

- **AdamW (Loshchilov & Hutter, ICLR 2019).** [arXiv 1711.05101](https://arxiv.org/abs/1711.05101). L2 desacoplado do segundo momento do Adam — contramedida cheap para weight-norm growth.
- **LayerNorm + weight projection (Lyle et al. NeurIPS 2024).** [arXiv 2407.01800](https://arxiv.org/abs/2407.01800). Previne decay do learning rate efetivo causado por crescimento de pesos.
- **Klein et al. (NeurIPS 2024). "A Study of Plasticity Loss in On-Policy Deep RL."** [arXiv 2405.19153](https://arxiv.org/abs/2405.19153). Resultado central específico para PPO+CNN: **regenerative regularization + LayerNorm é o melhor combo**; ReDo e last-layer reset *pioram* em PPO (apesar de ajudarem em DQN).

**Implicações aplicadas no projeto:** AdamW (`weight_decay=1e-4`), LayerNorm/GroupNorm pós-conv e pós-MLP, plasticity callback instrumentando dormant ratio + weight norm + stable rank.

---

## 2. Curriculum Learning em RL

**Bengio et al. (ICML 2009). "Curriculum Learning."** [PDF](https://ronan.collobert.com/pub/2009_curriculum_icml.pdf). Tese central: ordenar exemplos de fácil a difícil acelera convergência e pode atingir mínimos locais melhores em objetivos não-convexos. **Hacohen & Weinshall (ICML 2019)** mostram empiricamente que o efeito pode ser pequeno ou negativo fora de regimes específicos.

**Narvekar et al. (JMLR 2020). "Curriculum Learning for RL Domains: A Framework and Survey."** [arXiv 2003.04960](https://arxiv.org/abs/2003.04960). Taxonomia em três eixos: task generation, sequencing, transfer. **O projeto cai na célula mais simples e bem-comportada da taxonomia: paramétrico-manual + ordem fixa + weight-init transfer.** O survey nota que essa célula é uma escolha forte quando o knob de dificuldade é parametrizável.

### Currículos automáticos / UED (não aplicados, mas registrados)

- POET (Wang et al. 2019), PAIRED (Dennis et al. NeurIPS 2020), PLR (Jiang et al. ICML 2021), ACCEL (Parker-Holder et al. ICML 2022). Substituem decisão humana ("5×5 é mais fácil") por sinal de aprendizado (regret, TD-error). **Não justificáveis para 3 estágios paramétricos** — Narvekar 2020 e Theile et al. confirmam que manual continua dominante quando o knob é claro.

### Riscos de currículo

- **Catastrophic forgetting** (van de Ven et al. 2024, [arXiv 2403.05175](https://arxiv.org/abs/2403.05175)). Mitigação: avaliar nos estágios anteriores. **Aplicado:** `evaluate_cross.py`.
- **Negative transfer** (Anand & Precup 2024, [arXiv 2403.05066](https://arxiv.org/abs/2403.05066)). Sanity check possível: 20×20 from-scratch vs. com transfer; o subcomando `train_grid_world_cpp.py no-curriculum` está disponível, mas a comparação não foi reportada nesta entrega — o sinal indireto vem da cross-eval (stage 2 atinge 81 % no 20×20 sem treino direto).
- **Plasticity loss** (Dohare 2024) — tratado em §1.

### Prior art em CPP-RL com currículo de tamanho

- **Theile et al.** ([arXiv 2003.02609](https://arxiv.org/abs/2003.02609), [arXiv 2010.06917](https://arxiv.org/abs/2010.06917)). DDQN/PPO com map centering + dual local/global. Referência mais direta para a arquitetura de observação do projeto.
- **Jonnarth, Zhao, Felsberg (2023). "Learning Coverage Paths in Unknown Environments with Deep RL."** [arXiv 2306.16978](https://arxiv.org/abs/2306.16978). PPO + ego-mapas multi-escala + tiered curriculum. Mainstream metodológico.

---

## 3. Reward shaping (PBRS)

**Ng, Harada & Russell (ICML 1999).** Toda política ótima em M' = M + F é ótima em M sse F é potential-based: F(s,a,s') = γΦ(s') − Φ(s). Em PPO horizonte infinito, gradientes idênticos em expectativa.

**Devlin & Kudenko (AAMAS 2012). "Dynamic PBRS."** [PDF](https://eprints.whiterose.ac.uk/id/eprint/75121/2/p433_devlin.pdf). Generalizam para Φ(s, t); invariância preservada. **Justificativa formal direta** para Φ_t(s) = −d_BFS(s, fronteira_t) do projeto, que é dinâmico por construção.

**Eck et al. (JAAMAS 2015).** PBRS estende-se a POMDPs se Φ depende do *belief* (= histórico observado). Compatível com a escolha de fazer BFS sobre `visited ∪ ¬_seen_obstacles`.

**Behboudian et al. (NCAA 2022). "Policy Invariant Explicit Shaping."** Mesmo PBRS válido pode enviesar via *exploration bias* em learners não-tabulares (PPO!) quando |Φ| é grande relativo a R.

**Frontier-distance potential — literatura recente:** **Caraballo et al. (2025)** [arXiv 2504.11907](https://arxiv.org/html/2504.11907v2) usam exatamente Φ por proximidade à fronteira em RL para exploração. Validação independente do design.

**No código:** `Φ(s_T) = 0` em `terminated or truncated` (cumpre requisito do teorema). `shaping_scale` exposto para sweep futuro.

---

## 4. Coverage Path Planning com deep RL

**Galceran & Carreras (RAS 2013). "A survey on coverage path planning for robotics."** Métodos clássicos (boustrophedon, Morse) assumem mapa conhecido. Em mapa desconhecido sob observabilidade parcial, mapping + planning + cobertura viram problema sequencial — nicho de RL.

**Receita dominante na literatura recente** (Theile 2020/2021, Jonnarth 2023, Niroui 2019): CNN sobre stack de canais de mapa + observação egocêntrica + dois streams (local fino + global pooleado fixo) + PPO/DDQN. **Arquitetura do projeto está exatamente nessa receita.**

**Egocêntrico vs alocêntrico** (Mishkin et al. 2019, [arXiv 1901.10915](https://arxiv.org/abs/1901.10915)): políticas egocêntricas generalizam melhor a layouts novos; combinar ego (ação local) + alo (objetivo global) é o ótimo. Justifica o design: local 7×7 + global 8×8 pooleado + frontier vector.

**Memória estática basta para CPP estático** (Parisotto & Salakhutdinov ICLR 2018, Gupta et al. CVPR 2017): mapa explícito como canal extra dispensa LSTM quando não há dinâmica temporal além do mapa.

**Métricas padrão em CPP-RL:** full coverage rate, cobertura média, steps to coverage, **repeat ratio** (Theile 2020, Jonnarth 2023). Repeat ratio adicionado em `evaluate.py`.

---

## 5. Transfer learning entre estágios

**Taylor & Stone (JMLR 2009)** e **Zhu, Lin, Zhou (TPAMI 2023, [arXiv 2009.07888](https://arxiv.org/abs/2009.07888))**. Setup do projeto = "same-domain transfer with scaled state space" + "initialization transfer" — categorias mais favoráveis das taxonomias.

**Riscos em RL** (vs. fine-tuning supervisionado puro):

1. Optimizer state resetado (momentum perde calibração).
2. **Value head dessincronizada** com nova distribuição de retornos.
3. Distribuição de estados diferente.

**Igl et al. (ICLR 2021, [arXiv 2006.05826](https://arxiv.org/abs/2006.05826))** e **Wolczyk et al. (ICML 2024, [arXiv 2402.02868](https://arxiv.org/abs/2402.02868))**: reset seletivo no fine-tuning melhora estabilidade. **Klein 2024 cautela:** *full* last-layer reset hurts PPO. Compromisso aplicado: resetar **só o value head**, mantendo features e policy head.

**Alternativas mais robustas (não aplicadas, registradas):** Progressive Networks (Rusu et al. 2016), Successor Features (Barreto et al. NeurIPS 2017), Meta-RL (MAML/RL²/PEARL). Justificativa para não usar: overkill para 3 estágios paramétricos.

---

## 6. Exploração com recompensa esparsa

PPO com recompensa terminal `+10` longe sofre de credit assignment em horizonte longo. Entropia da policy + ε-greedy não bastam quando o random walk dificilmente fecha cobertura no 20×20.

- **Intrinsic motivation** (ICM, RND, NGU) — em grid determinístico, **redundante**: contagem de visitadas é mais barata e bem calibrada.
- **Count-based** (Tang et al. NeurIPS 2017): em CPP a contagem é trivial (índice (x,y)). Substituir +1/0 por β/√N(x,y) é alternativa para ablação futura.
- **Frontier shaping com PBRS** (Yamauchi 1997 + Ng 1999 + Caraballo 2025): a alavanca principal aplicada no projeto.
- **Currículo como exploração disfarçada** (Theile 2025): no 5×5 random walk cobre tudo, agente aprende estrutura do reward, e quando o mapa cresce já tem priors úteis.
- **Achado de Jonnarth 2023:** rewards adicionais esparsos para "coverage completo" não ajudam quando já existe reward denso por nova célula. **Implicação:** o `+10` terminal pode ser mais simbólico que funcional.

**HRL/options só fazem sentido em mapas ≥ 30×30**, fora do escopo.

---

## 7. Síntese — o que sustenta as decisões do código

### Validado pela literatura
- Currículo manual paramétrico (Narvekar 2020; Theile, Jonnarth, Kyaw em CPP-RL).
- Observação egocêntrica + global pool fixo (Theile 2021; Mishkin 2019).
- CNN feed-forward sem LSTM para CPP estático (consenso Theile/Jonnarth).
- PBRS dinâmico Φ = −d_BFS(fronteira) (Ng 1999 + Devlin 2012 + Eck 2015 + Caraballo 2025).
- Initialization transfer entre estágios (Taylor-Stone; Zhu 2023).

### Identificado como risco e mitigado
- Plasticity loss (Dohare 2024 + Klein 2024): AdamW + LayerNorm + plasticity callback.
- Value head desalinhada (Igl 2021 + Wolczyk 2024): reset entre estágios.
- Catastrophic forgetting: cross-eval dos modelos.

### Confirmado empiricamente nas runs (§3 do RELATORIO.md)
- Sem catastrophic forgetting (stage 3 mantém 96 % no 5×5).
- Plasticidade *parcialmente* mitigada: dormant ratio sob controle (<25 %, longe dos 30–60 % catastróficos), mas stable rank cai 1.62 → 1.13 (Kumar 2021) e weight norm cresce 2.5× (Lyle 2024) — coerente com a tese de Lyle 2024 de múltiplos mecanismos independentes.
- Variantes mais agressivas (`weight_decay=5e-4`, `F=16`) melhoraram diagnósticos de plasticidade durante o treino mas não moveram o teto de full coverage.
- Teto de full coverage no 20×20 (~80 %) **não cede** a aumento de weight_decay nem a F=16 — provavelmente estrutural do MDP.

### O que não fazer (literatura é clara)
- BatchNorm em PPO (instabilidade train/eval).
- Last-layer reset / ReDo em PPO (Klein 2024 mostra que pioram).
- ICM/RND em grid determinístico.
- Meta-RL para 3 estágios.
