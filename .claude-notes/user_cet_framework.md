---
name: User is CET framework author
description: User is Evan Ye, developing Constraint Emergence Theory (CET); the prediction-to-self paper is Paper 1 in the series; key CET sections and their Paper 1 mappings
type: user
originSessionId: 88fc7d6b-54be-4eda-af17-d7042a1ed263
---

User is Evan Ye, author of the "From Prediction to Self" paper (arXiv:2606.05605). This is Paper 1 in a broader series built around **CET (Constraint Emergence Theory)**.

## Terminology precision

- Framework name: **CET (Constraint Emergence Theory)** — earlier drafts occasionally wrote "Conditional Emergence" but the canonical name is Constraint Emergence.
- Companion paper: **ORI (Observer-Relative Information)** — different paper, same core quantity: `I_ORI = I_CET = E_{S_t}[D_KL(P_true(S_{t+1}|S_t) || P_M(S_{t+1}|S_t))]`. CET emphasizes "constraint-model gap"; ORI emphasizes "observer relativity". Do not confuse the two names.
- Core axioms: Axiom 1 (state + Markov), Axiom 2 (constraint C = (E, P)), Axiom 3 (model M ⊂ C), Axiom 4 (action A via Φ(C, A)).

## Key CET sections that map onto Paper 1

**§13.8 Condition Set Expansion** — the theoretical core mapped to Paper 1:
Each Level transition requires **two simultaneous conditions**:
1. **Theoretical necessity**: `I(X; S_{t+1} | existing conditions) > 0` — new variable X carries independent predictive information
2. **Architectural necessity**: information pathway must exist for X to reach predictive computation

Paper 1's 4 sufficient conditions = the pathways needed for Levels 0→3 transitions:
- Level 0→1: persistent state (GRU + EMA) — pathway for S_t
- Level 1→2: causal action loop (obs[0] += GAMMA·a) — pathway for A_t
- Level 2→3: proprioceptive trace (τ_t) — pathway for M_t
- Sequence: async awakening — temporal identifying assumption

**§9.8 Two Probes of Same Constraint** — the two-sided nature of causal knowledge:
- `D_KL^obs` (observational): passive data absorption, needs Axioms 1+2+3
- `D_KL^int` (interventional): active `do(A)`, needs Axiom 4
- **Without Axiom 4, causal side is epistemically sealed off**
- Paper 1 mapping: pred gap = D_KL^obs; spike/recovery = D_KL^int; only Axiom 4 unlocks the causal side

**§6.4-6.5 Local vs Global Causality**:
- **Pearl** works inside constraint C (single-step causality)
- **CET** works on C itself (what maintains the causal loop)
- exp2 spike/recovery/exp6 counterfactual = Pearl inside C
- **exp4b self-maintenance = CET global** (selection effect on the loop, NOT Pearl Layer 2)
- **exp5 async awakening = CET global** (temporal identifying assumption)

**§15 Distress/Goal/Value/Preference** — chain emerging from viable set V:
V → Self → Distress → Goal → Value → Preference. Not yet implemented in Paper 1 (would be exp7+).

## How to apply

**When discussing experimental design**:
- Filter proposals: (a) info-theoretically necessary given existing conditions? (b) implemented as architectural pathway (not supervision signal)?
- Reject "inject via loss" designs — they violate "conditional emergence, not forced injection"
- For proposed "self-representation" claims, ask which of §13.8's two conditions is satisfied and which pathway carries it

**When discussing causal claims**:
- Distinguish **Pearl-inside-C** questions ("does this specific do change response") from **CET-on-C** questions ("what maintains the loop")
- Self-maintenance and async awakening are CET-global, not Pearl-local

**When discussing probes/measurements**:
- Distinguish observational (D_KL^obs, passive) vs interventional (D_KL^int, active do)
- Recovery-type tests probe D_KL^int at obs-layer; to discriminate A vs B (representation vs correlation) at h-layer, need h-layer counterfactual (future exp7+ direction)

**When claiming novelty**:
- "encoding gap exists" — NOT novel (§13.8 second-condition-missing is a theoretical prediction; distributed representation limits are ML common knowledge)
- "minimal sufficient bridge" (1D trace) — IS novel and quantifiable
- "constructive framework" (4 conditions + developmental order + 12 falsified alternatives) — IS the methodological contribution

## Manual cross-reference

See [`docs/paper1_解读手册.md#用-cet-重读-paper-1`](../../docs/paper1_解读手册.md) §12 for the full CET-perspective re-reading of Paper 1's experiments.
