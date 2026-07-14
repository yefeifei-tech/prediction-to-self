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

## Abstract（draft v2, ~180 words, incorporating critique revisions）

For twenty years, reservoir computing and deep learning have been treated as competing paradigms: one keeps recurrent dynamics fixed and trains only a readout; the other trains everything end-to-end. We propose that this is a false dichotomy. Both are cases of a unified three-level framework: **Dynamics** (rich state trajectories), **Readout** (task-relevant recovery from those states), and **Dynamics Shaping** (reorganizing dynamics for compact, composable representation). Reservoir computing uses only Levels 1-2; standard deep learning adds Level 3. Neither is fundamentally right — they occupy different points on a spectrum. Using a minimal Level 2 case study (a 192-dim GRU with frozen recurrent weights that develops **implicit causal utilization** via only readout training), we argue that learning can be viewed as **progressive alignment between internal dynamics and world dynamics** — an ontological alignment complementing the traditional encoding view. Level 3 shaping refines alignment (making representations more compact and composable), rather than being the sole source of representational meaning. The framework yields three testable predictions about which architectures can support such alignment (specifically requiring persistent internal state) and how alignment scales with task complexity.

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

### Section 3. A Minimal Level 2 Case Study: Implicit Causal Utilization in a Frozen GRU

（**修正**：不能说 L2 alone develops full self-representation——那和 Paper 1 exp2 严格边界打架）

- Reference to Paper 1 ("From Prediction to Self", arXiv:2606.05605)
- Setup: 192-dim GRU with random-frozen weights + causal action loop + trainable prediction heads
- **Key finding**: **implicit causal utilization** emerges without training the recurrent dynamics
  - System's readout learns to compensate for own-action effects via the reservoir's h
  - Behavioral evidence (spike test, recovery) confirms causal dependence
  - **This is what L2 alone accomplishes**
- **What L2 alone does NOT accomplish**:
  - **Readable self-representation** requires an additional pathway (Paper 1 exp4 adds 1D proprioceptive trace)
  - The encoding gap between L2's implicit utilization and L2-plus-pathway's readable representation is Paper 1's central conceptual finding
- Attribution:
  - Reservoir dynamics propagate constraint information from world → obs → h
  - Readout learns to select h subspaces useful for prediction
  - Additional 1D pathway (trace) enables readable readout of "recently acted" state

### Section 4. Standard Deep Learning as Level 3

- Why RC alone doesn't scale to modern tasks (millions of concepts, hierarchical semantics)
- **Random dynamics can *expand* world, but cannot *organize* world**
- Deep learning does organize:
  - Gradient descent creates task-aligned feature clusters
  - Cats near cats, dogs near dogs, animals form a hierarchy
- **This organization is what Level 3 shaping accomplishes**
- Reference to representation learning literature (Bengio et al. 2013, etc.)

### Section 5. Learning as Ontological Alignment

**§5.1 The philosophical reframe**（tighten wording per critique）

- Traditional view: learning creates task-optimized representations via gradient descent
- Three-level reframe: **learning can be viewed as progressive alignment between internal dynamics and world dynamics**
  - Not denying creation happens at parameter level
  - Reframing what makes the created structures meaningful: alignment with world structure
- **Both world and RNN belong to the same mathematical class: dynamical systems**
  - (State-and-transition, not physical resonance metaphor)
- **One possible interpretation of GOFAI's limits**: symbolic systems lack an intrinsic continuous dynamical substrate; alignment with continuously-varying world dynamics becomes structurally difficult (multiple factors likely contribute — combinatorial explosion, symbol grounding, uncertainty handling — this is one interpretation among several)

**§5.1a L0 Constraint as the deeper substrate**（新增 subsection）

- Underlying L1-L3 is a deeper substrate: **the constraint structure of the world** — what learning aligns *to*
- Developed in companion theoretical work on Constraint Emergence Theory [Ye 2026+ CET]
- For this paper, we take world dynamics as given and focus on how internal dynamics can align with them at L1-L3
- L0 is not itself a learning layer; it's the target of alignment. L1-L3 are where learning operates

**§5.2 The operational consequence**

- "Training a model" is no longer a vague positive term
- Precise definition: **reshape the model's internal dynamics such that they align with target dynamics structure**
- Assessment shifts from "loss went down" to "how well aligned are the internal dynamics with the target?"

**§5.3 Connection to enactivism and predictive processing**

- Varela, Thompson, Rosch: cognition as sensorimotor coupling
- Friston, Clark: predictive processing / active inference
- Three-level framework provides a concrete mechanism for these philosophical positions

### Section 6. Predictions and Testable Hypotheses

**§6.1 Prediction 1: Architectures without persistent internal state show limitations**（tighten wording per critique）

- Rationale: no Level 1 (persistent state) = no dynamical alignment possible
- Failure predicted on tasks requiring sustained temporal integration: **pure feed-forward architectures without persistent internal state** (careful: don't blanket-attack "Transformer" — KV cache, attention over context, and stateful Transformer variants are partial states)
- Success possible: RNN, Mamba, KV-cached Transformer, RWKV, xLSTM, Neural ODEs
- Experimental design: replicate Paper 1's setup with truly stateless feed-forward architecture; check for encoding gap analog

**§6.2 Prediction 2: L2 vs L3 representation differs in structure**

- L2 (reservoir): representation flat readable direction; task-specific, less transferable
- L3 (shaped): representation should be hierarchical, compositional, transferable
- **Concrete measurable metrics** (must specify or prediction is unfalsifiable):
  - **Linear separability**: class separation in h space
  - **CKA (Centered Kernel Alignment)** / **SVCCA**: cross-layer/cross-model similarity
  - **Mutual Information gap**: hierarchical representations should show ancestor-concept containing descendant-concept information
  - **Compositionality tests** (systematic generalization benchmarks)
  - **Transfer learning performance**: hierarchical L3 should transfer better
- Experimental design: same probe methodology on L2 vs L3 versions; report structure metrics, not just recall

**§6.3 Prediction 3: Shaping necessity scales with task complexity (a form of scaling law)**

- Simple dynamics prediction (sinusoid, Lorenz) → L2 sufficient (frozen reservoir OK)
- Hierarchical concepts (language, vision) → L3 required (must shape dynamics)
- **This is effectively a scaling law**: task complexity ↑ → L3 marginal benefit ↑
- Experimental design: sweep task complexity axis (simple → hierarchical), plot L2 vs L3 performance gap
- **Falsifiable prediction**: gap should widen with complexity; if gap stays constant, framework needs revision

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
