# Three Levels of Learning — Paper Outline

**状态**：initial outline draft（still evolving）

**Target length**：short paper（15-20 pages）—— 不是 magnum opus，是**清晰的重构性理论论文**

**Target venue**：候选——arXiv preprint 优先；后续投期刊可选 Neural Computation / Cognitive Science / Behavioral and Brain Sciences

---

## Title candidates

- "Three Levels of Learning: Dynamics, Readout, and Shaping"
- "Learning as Alignment, Not Encoding: A Three-Level Framework"
- "Reservoir Computing and Deep Learning are Levels, Not Rivals"

（等 outline 稳定后决定）

## Abstract（draft, ~150 words）

For twenty years, reservoir computing and deep learning have been treated as competing paradigms: one keeps recurrent dynamics fixed and trains only a readout; the other trains everything end-to-end. We propose that this is a false dichotomy. Both are cases of a unified three-level framework: **Dynamics** (rich state trajectories), **Readout** (task-relevant recovery from those states), and **Dynamics Shaping** (reorganizing dynamics for compact, composable representation). Reservoir computing uses only Levels 1-2; standard deep learning adds Level 3. Neither is fundamentally right — they occupy different points on a spectrum. Using a minimal Level 2 case study (a 192-dim GRU with frozen recurrent weights that develops self-representation via only readout training), we argue that "learning" is not knowledge creation but ontological alignment of internal dynamics with world structure. Level 3 shaping is refinement of alignment, not creation of knowledge. This reframing yields concrete predictions about which architectures can develop self-representation and where reservoir approaches suffice.

## Structure

### Section 1. Introduction: A Twenty-Year Debate

- Reservoir Computing (Jaeger 2001, Maass 2002) vs Deep Learning (Hinton, LeCun, Bengio)
- Two positions:
  - RC: "random recurrent networks are already rich enough; only train readout"
  - DL: "train everything to learn task-optimal representations"
- Both work in different regimes. Why?
- **Thesis**: they occupy different levels of a unified framework; the debate is misframed

### Section 2. The Three Levels

**§2.1 Level 1: Dynamics**

- Definition: a system that produces state trajectories via input + recurrence
- Mathematical form: $h_{t+1} = f(h_t, x_t)$, $f$ fixed or random
- Properties: boundedness, contractive/expansive, echo state property (Jaeger 2001)
- Examples: any RNN with random weights, physical dynamical systems, weather

**§2.2 Level 2: Readout**

- Definition: a trainable projection from state to task-relevant output
- Mathematical form: $y_t = g_\theta(h_t)$, $g_\theta$ learned
- Key property: **selection/discovery**, not creation
- CSPP (constraint selection through prediction power) view
- Examples: reservoir computing, linear probes, echo state networks

**§2.3 Level 3: Dynamics Shaping**

- Definition: training also modifies $f$ (recurrent weights), not just $g$
- Mathematical form: $h_{t+1} = f_\theta(h_t, x_t)$ with $f_\theta$ learned
- Effect: reorganizing state space so task-relevant representations become **compact, stable, composable, easier to read out**
- **Not "creating knowledge" but "improving alignment"**
- Examples: standard RNN training, LSTM, Transformer (attention as dynamic shaping)

**§2.4 Levels are cumulative**

- L2 requires L1 (need dynamics to read from)
- L3 requires L1+L2 (shape dynamics to be readable)
- The choice of stopping level is architectural

### Section 3. A Minimal Level 2 Case Study: Self-Representation in a Frozen GRU

- Reference to Paper 1 ("From Prediction to Self", arXiv:2606.05605)
- Setup: 192-dim GRU with random-frozen weights + causal action loop + trainable prediction heads
- Key finding: self-representation emerges **without training the recurrent dynamics**
- Attribution:
  - Reservoir dynamics propagate constraint information from world → obs → h
  - Readout learns to select h subspaces that carry self-relevant info
  - Adding a 1D proprioceptive channel (trace) opens the pathway for readable self-representation
- **This is L2 in action** — no dynamics shaping, yet Level 3 self-representation (in CET terms) emerges

### Section 4. Standard Deep Learning as Level 3

- Why RC alone doesn't scale to modern tasks (millions of concepts, hierarchical semantics)
- **Random dynamics can *expand* world, but cannot *organize* world**
- Deep learning does organize:
  - Gradient descent creates task-aligned feature clusters
  - Cats near cats, dogs near dogs, animals form a hierarchy
- **This organization is what Level 3 shaping accomplishes**
- Reference to representation learning literature (Bengio et al. 2013, etc.)

### Section 5. Learning as Ontological Alignment

**§5.1 The philosophical reframe**

- Traditional view: model **encodes** world (world → data → model → internal representation)
- Three-level view: model **is** a dynamical system that can **align** with world's dynamical structure
- **World is dynamics; RNN is dynamics; alignment is possible**
- Symbol AI (GOFAI) failed because discrete symbols cannot align with continuous world

**§5.2 The operational consequence**

- "Training a model" is no longer a vague positive term
- Precise definition: **reshape the model's internal dynamics such that they align with target dynamics structure**
- Assessment shifts from "loss went down" to "how well aligned are the internal dynamics with the target?"

**§5.3 Connection to enactivism and predictive processing**

- Varela, Thompson, Rosch: cognition as sensorimotor coupling
- Friston, Clark: predictive processing / active inference
- Three-level framework provides a concrete mechanism for these philosophical positions

### Section 6. Predictions and Testable Hypotheses

**§6.1 Prediction 1: Stateless architectures cannot form self-representation via this mechanism**

- Rationale: no Level 1 (no persistent state) = no dynamical alignment possible
- Failure predicted: vanilla Transformer, MLP, feedforward networks
- Success possible: RNN, Mamba, KV-cached Transformer, RWKV, xLSTM
- Experimental design: replicate Paper 1's setup with stateless architecture; check for encoding gap analog

**§6.2 Prediction 2: L2 vs L3 self-representation differs in structure**

- L2 (reservoir): self-representation is a flat readable direction in state space
- L3 (shaped): self-representation should be hierarchical, compositional, transferable
- Experimental design: same probe methodology on L2 vs L3 versions, measure structure properties (not just recall)

**§6.3 Prediction 3: Shaping necessity scales with task complexity**

- Simple dynamics prediction (sinusoid, Lorenz) → L2 sufficient
- Hierarchical concepts (language, vision) → L3 required
- Provides principled criterion for when to add shaping

### Section 7. Discussion

**§7.1 Not disputing deep learning**

- L3 is real, useful, and often necessary
- The claim is not "reservoir is enough"; it's "the three levels are separate and different systems occupy different points"

**§7.2 Limitations**

- Three levels are analytical framework, not clean architectural boundaries
- Attention mechanisms in Transformers do both shaping and readout
- Practical systems often blur the levels
- **"Compression" is too narrow** — training is **structural reshaping** (compression, expansion, separation, aggregation)

**§7.3 Implications for AI safety and alignment**

- If training is alignment, then "misalignment" has a specific dynamical meaning
- Understanding L3 shaping mechanisms could inform alignment research
- (Cautious speculation, not overreach)

### Section 8. Conclusion

- Three levels: Dynamics, Readout, Shaping
- Reservoir and DL differ in level, not fundamentally
- Learning is alignment, not encoding
- Concrete predictions offered; falsifiable

## References（初步 seeds）

- Jaeger, H. (2001). Echo state networks.
- Maass, W., Natschläger, T., Markram, H. (2002). Liquid state machines.
- Bengio, Y., Courville, A., Vincent, P. (2013). Representation learning.
- Tishby, N., Zaslavsky, N. (2015). Deep learning and information bottleneck.
- Varela, F., Thompson, E., Rosch, E. (1991). The Embodied Mind.
- Clark, A. (2016). Surfing Uncertainty.
- Friston, K. (2010). Free energy principle.
- Chen, T.Q. et al. (2018). Neural Ordinary Differential Equations.
- Rao, R., Ballard, D. (1999). Predictive coding in the visual cortex.
- **Ye, E. (2026). From Prediction to Self.** (arXiv:2606.05605) — 主 case study

## 待完成的写作前工作

1. ☐ **文献调研**：查过去 5 年 reservoir computing 和 deep learning 有没有类似"统一框架"的提议——可能已经有人做了类似工作
2. ☐ **实验数据**：至少要跑一个 Prediction 1 或 2 的 minimal 实验，为论文提供 empirical anchor
3. ☐ **Related Work 定位**：与 information bottleneck / neural ODE / world models 等相关工作的关系
4. ☐ **术语确认**：Level / Dynamics / Readout / Shaping / Alignment 等术语是否与既有文献冲突
5. ☐ **图示设计**：三层结构的直观图；L2 vs L3 的对比图；本体论对齐 vs 编码的示意图

## 写作策略

**Short paper 定位**：15-20 页，focus on reframe 而不是 exhaustive results

**关键要**：
- 一个清晰的 conceptual 骨架（三层）
- 一个 concrete case study（Paper 1 作为 L2 minimal case）
- 至少一个 empirical prediction 的 minimal test（如 stateless-fails）
- 不需要"打赢"reservoir vs DL 的辩论——需要**给出让两者各得其所的框架**
