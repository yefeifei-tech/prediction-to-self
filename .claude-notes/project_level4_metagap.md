---
name: Level 3→4 meta-encoding-gap experiment (planned)
description: Extension experiment testing whether a recursive encoding gap exists at the meta-self-representation level, using ρ_t as a 6th GRU input dimension
type: project
originSessionId: 88fc7d6b-54be-4eda-af17-d7042a1ed263
---
**Planned experiment**: Test whether the encoding gap is recursive — specifically whether Level 3 (self-representation "I am acting") → Level 4 (meta-self-representation "I know my self-model is reliable") shows the same architectural gap that Level 2 → Level 3 showed.

**Design (per CET §13.8, D-direction / recursive trace)**:
- Level 4 pathway: `ρ_t = β·ρ_{t-1} + (1-β)·1[P1(h_t) correct]` — EMA of Level-1 probe's correctness
- GRU input: 4d obs + 1d τ (Level 3 trace) + 1d ρ (Level 4 trace) = 6d
- Meta-probe P2 reads from detached h_multi, predicts "is P1 currently correct?"
- Comparison: exp4 baseline (5d, no ρ) vs +ρ_t (6d) vs +ρ_t on AR(1) control group

**CET prediction**:
- If baseline P2 recall is low AND jumps with +ρ_t → recursive gap confirmed, each level needs its own architectural channel
- If baseline P2 recall is already high → multi-scale EMA already encodes meta-info (also interesting)

**Why**: Directly tests CET §13.8's core claim that condition-set expansion is recursive and each level requires simultaneous satisfaction of (information value + information pathway).

**How to apply**:
- Location: new folder (e.g., `experiments_ext/metacog_level4/`), not `experiments/`
- Model variant: inline in the experiment file, do not modify `core/model.py`
- Terms to recognize in future conversations: "Level 4", "meta-probe", "ρ_t", "recursive encoding gap", "meta-self-representation"
- Open design questions (not yet resolved with user): (1) freeze P1 vs online P1, (2) ρ_t's dependence on ground-truth labels — pure internal vs partially exogenous, (3) whether ρ_t should be lagged to avoid circular dependency
