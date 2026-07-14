---
name: Paper 1 as CET §13.8 minimal empirical demonstration
description: Canonical statement of how Paper 1 maps into CET framework — Paper 1 is the minimal empirical anchor for §13.8's condition set expansion theory, not a standalone "encoding gap discovery" paper
type: project
originSessionId: cet-alignment-session
---

**Central claim**: Paper 1 = **minimal empirical demonstration of CET §13.8 (Condition Set Expansion)**. Its scientific value is not "we discovered encoding gap" (that's the necessary consequence of §13.8's second condition missing — theoretically expected). Its value is:

1. **Proving §13.8's two conditions** (info value + architectural pathway) **are reproducible in a minimal artificial system**
2. **Quantifying the minimum sufficient pathway** for each Level transition (specifically: 1D EMA of |a| for Level 2→3)
3. **Providing selection-effect evidence** (§14.3) for Self-persistence via exp4b (not intentional-agent framing)
4. **Cross-verifying constraints via CET §9.8's two probes** (spike/recovery/counterfactual span D_KL^int; pred gap = D_KL^obs)
5. **Giving developmental order as identifying assumption** (§6.4-6.5 global causality) via exp5

## Correspondence table

| Paper 1 element | CET reference | What it demonstrates |
|-----------------|-------------|---------------------|
| 4 sufficient conditions | §13.8 | Architectural pathways for Level 0→3 |
| exp1 attractor | §7 (manifold as constraint geometry) | C's geometric expression via random reservoir + input |
| exp2 spike test | §9.8 D_KL^int | Interventional probe of C's causal side (obs layer) |
| exp2 recovery | §9.8 D_KL^int sustained | Structured vs statistical model at obs layer |
| exp2 Control | §9.8 cross-check | Ruling out purely-observational absorption |
| exp3 low probe recall | §13.8 second condition missing | Level 2→3 stall without pathway |
| exp4 60%+ probe recall | §13.8 pathway opened | Level 2→3 transition when 1D trace added |
| exp4b self-maintenance | §6.4-6.5 global + §14.3 selection | Loop persistence via selection effect (NOT Pearl) |
| exp5 async awakening | §6.4-6.5 global | Temporal identifying assumption |
| exp6 counterfactual | Pearl Layer 3 + §9.8 h-layer D_KL^int | Full counterfactual on action input dim |

## What Paper 1 does NOT claim (per CET framework)

- **Value / Goal / Preference** (§15) — requires viable set V distress signals; Paper 1 stays below this level
- **Constraint creation** (§11) — Paper 1's agents only utilize existing C, don't create persistent new constraints
- **Multi-layer V** (§16) — no hierarchical viable sets; Paper 1's V is single-level (body-scale only)
- **Level 3→4 recursive self-model** — Paper 1 stops at Level 3; exp7 (metacog_level4) targets Level 4 but is paused

## Key insight: encoding gap is theoretically expected, not novel

Common inflation: "Paper 1's central discovery is that predictive competence and self-representation are dissociated."

**More precise CET statement**: "This dissociation is the necessary consequence of §13.8's second condition (architectural pathway) missing. Paper 1's original contribution is:
1. Demonstrating this theoretical prediction in a minimal artificial system
2. Quantifying the exact minimum architectural bridge (1D EMA of |a|)
3. Grounding this bridge to biological proprioception (trace ≈ muscle spindle feedback)"

## The A vs B question (from CET §9.8 lens)

exp2 leaves ambiguous:
- **A** (representational decomposition): h has both D_KL^obs and D_KL^int absorbed at h-layer
- **B** (retained self-correlated variable): only D_KL^obs absorbed at h-layer

Recovery test is obs-layer D_KL^int — doesn't touch h-layer. Cannot close A vs B.

**Full Self claim requires FIVE conjunctive conditions** (not any single experiment):
1. D_KL^obs absorption (exp2 pred gap)
2. D_KL^int absorption at obs layer (exp2 recovery + spike)
3. h-layer readable representation (exp4 probe recall)
4. Loop-level selection effect (exp4b self-maintenance)
5. Temporal identifying assumption (exp5 async)

## Correction of my earlier confusions (documented for future sessions)

Through Q&A with Evan, several imprecise framings emerged and were corrected:

- **"Pearl Layer 2 covers spike + recovery + self-maintenance"** — WRONG. Self-maintenance is CET §6.4-6.5 global, not Pearl-inside-C.
- **"Encoding gap is Paper 1's central discovery"** — OVERCLAIMED. It's the §13.8 second-condition-missing prediction; ML-common-knowledge. Novel = the minimal bridge.
- **"Control learned nothing"** — WRONG. Control learned *statistical* utilization (AR(1) autocorrelation). Its partial success is NECESSARY for the contrast argument.
- **"Recovery proves causal decomposition"** — OVERCLAIMED. Recovery is obs-layer D_KL^int; cannot discriminate A (representational) from B (correlational) at h-layer.
- **"Framework is ORI"** — WRONG name. Framework is CET (Constraint Emergence Theory); ORI (Observer-Relative Information) is companion paper. Same core quantity, different emphasis.

## Future directions (CET-driven)

1. **h-layer D_KL^int**: direct do on h dimensions to close A vs B ambiguity
2. **Level 3→4 recursive self-model**: exp7 metacog_level4 is first attempt (paused)
3. **Constraint creation (§11)**: agents that make persistent new constraints, not just utilize
4. **Multi-layer V (§16)**: agents that identify with different viable-set layers
5. **Test the stateless-fails prediction**: run same causal-loop setup on vanilla Transformer / MLP (no persistent state); CET predicts no self-representation emerges via this mechanism

## Reservoir insight — the deeper implication of "GRU frozen"

**Empirical fact (verified)**: GRU weights delta = 0 in exp1/exp2/exp6. Only pred_A/pred_B/W_action-like readouts train.

**Deep implication**: AG > 0 is NOT trained structure — it's the fixed reservoir dynamics naturally producing structure that the readout discovers.

**Complete causal chain**:
```
World constraint C  →  structured obs distribution  →  obs into GRU (blind reservoir)
  →  h_multi encoding of obs history  →  pred_A/B readout discovers useful subspaces  →  AG > 0
```

**Attribution**:
- **Reservoir doesn't "know" constraint** — it's a blind dynamical system
- **Info in h is not learned** — it's propagated via "constraint → obs → fixed GRU"
- **pred_A/B are discovery tools, not information sources** — 192→4 linear projections extracting what's already in h
- **Learning happens at readout, not at dynamics** — concrete CSPP (CET §1.5) implementation

**Three CET arguments this strengthens**:

1. **Dual-head is measurement tool, not information source**: sharpens Paper 1 §6.1 claim. pred_A cannot create info; can only read from h. AG entirely attributable to reservoir + input structure.

2. **Axiom 1 (persistent state) more fundamental than "learning capability"**: revised CET claim: "Persistent dynamical system (no learning needed) + trainable readout (necessary) = minimum architecture for Level 1→2 transition". Learning necessary but only at readout, not at dynamics.

3. **Stateless architectures predicted to fail**: vanilla Transformer/MLP lack Axiom 1 (no persistent updating state); cannot form self-representation via this mechanism. Stateful architectures (RNN, Mamba, KV-cached Transformer, RWKV, xLSTM) satisfy the prerequisite; success depends on other CET conditions. Untested prediction — needs ablation experiment.

**Implicit correction to CET §13.8's phrasing**:

- **Original wording implies "system actively expands condition set"** — but Paper 1 shows system doesn't autonomously expand. Architecture is experimenter-provided; system's readout learns to utilize it.
- **Stricter phrasing**: "When experimenter/environment provides architectural pathway AND informational condition holds, system's readout under optimization pressure discovers and utilizes the pathway, so representation emerges in the 'being-utilized' sense."
- **This shifts the subject from "system autonomously expanding" to "environment provides, readout discovers"** — more accurately reflects what Paper 1 demonstrates.

**Three-column impact table**:

| Contribution | Content | Impact |
|--------------|---------|--------|
| Attribution precision | AG > 0 fully attributed to reservoir + input, not training | "dual-head is measurement" claim mechanized |
| Axiom 1 positioning | Persistent state + trainable readout = minimum architecture | Learning at readout (selection), not dynamics (shaping) |
| Subject clarification | Environment provides, readout discovers | Contracts "system autonomously expands" overclaim |

## Cross-references

- Full manual re-reading: [`docs/paper1_解读手册.md`](../../docs/paper1_解读手册.md) §12 "用 CET 重读 Paper 1"
- 8-step evidence chain (dual-framed): [[project_paper1_methodology_chain]]
- exp2 strict boundary: [[project_exp2_strict_boundary]]
- Architecture causal ladder: [[project_architecture_causal_ladder]]
