---
name: Three Levels of Learning — planned paper
description: Independent theoretical paper in development, arguing that reservoir computing and deep learning are levels of a unified three-level framework (Dynamics, Readout, Shaping) rather than competing paradigms; core reframe = "learning is alignment, not encoding"
type: project
originSessionId: three-levels-emergence-session
---

**Status**: pre-outline, working notes stage (2026-07)

**Location**: [`papers/three_levels/`](../../papers/three_levels/) in the private repo

## Central claim

Reservoir computing and deep learning are **not competing paradigms**. Both are cases of a unified three-level framework:

- **L1 Dynamics**: system produces rich state trajectories (fixed or random)
- **L2 Readout**: trainable projection from state to task-relevant output (selection, not creation)
- **L3 Dynamics Shaping**: training also modifies recurrent dynamics (reorganizes state space for readable, composable representation)

**Reservoir uses L1+L2; standard DL adds L3**. Neither is fundamentally right — they occupy different points on a spectrum.

## The deep reframe

**Learning ≠ knowledge creation**. **Learning = ontological alignment** — reshaping the model's internal dynamics to align with the world's dynamical structure.

**Why possible**: world IS dynamics (weather, stock, language token sequences all evolve continuously); RNN IS dynamics; therefore alignment/resonance is possible. Symbol AI (GOFAI) failed because discrete symbols cannot align with continuous world.

## Relation to Paper 1

Paper 1 ("From Prediction to Self") is used as **minimal L2 case study**: 192-dim GRU with **frozen** recurrent weights + trainable prediction heads develops self-representation. This is L2 in action — no dynamics shaping, yet Level 3 self-representation (in CET terms) emerges.

**Paper 1 stopping at L2 is deliberate** (attribution clarity), not because L2 is optimal.

## Three testable predictions

1. **Stateless architectures fail**: vanilla Transformer, MLP cannot form self-representation via this mechanism (no L1). Stateful (RNN, Mamba, KV-cached Transformer) can.
2. **L2 vs L3 self-representation differ in structure**: L2 = flat readable direction; L3 = hierarchical, compositional, transferable.
3. **Shaping necessity scales with task complexity**: simple dynamics → L2 sufficient; hierarchical concepts → L3 required.

## Key refinements identified so far

- "Learning = compression" is too narrow. More general: **training = structural reshaping** (compression, expansion, separation, aggregation depending on task). Compression is one instance (information bottleneck view) but not exclusive.
- The three levels are analytical framework, not clean architectural boundaries — modern DL blurs them (attention does both shaping and readout).

## Relation to CET

**The three-level framework is CET's observer perspective at the mechanism level**:

| CET | Three-level |
|-----|-------------|
| Constraint (objective C) | L1 Dynamics's raw representation space |
| Information (subjective I) | L2 Readout's discovery |
| Learning (constraint absorption) | L3 Shaping — reorganize for readability |

**"Information doesn't create constraint, recovers it" (CET) ↔ "Representation isn't created, read from dynamics" (three-level)** — same philosophical stance, different level of expression.

## Files

- [`papers/three_levels/notes.md`](../../papers/three_levels/notes.md) — running observations from discussions
- [`papers/three_levels/outline.md`](../../papers/three_levels/outline.md) — paper structure draft

## Where this came from

Emerged from 2026-07 Paper 1 study session. Evan pushed on the "GRU frozen only pred trained" fact until it crystallized into a general theoretical stance. Key inflection moments:

- Realization that pred is **selection/readout**, not creation
- Recognition that "training weights" needs to be distinguished from "shaping dynamics"
- The ontological alignment insight — "world is dynamics, GRU is dynamics, so they resonate"

## Next steps (candidates)

1. **Literature review**: check if similar unified framework already proposed (last 5 years)
2. **Minimal empirical test**: run Prediction 1 (stateless fails) as concrete evidence
3. **Term consolidation**: ensure Level/Dynamics/Readout/Shaping terminology doesn't conflict with existing literature
4. **Diagram design**: three-level structure visualization; L2 vs L3 comparison; alignment vs encoding illustration
5. **Draft writing**: start with Section 1 (Introduction) and Section 5 (Ontological Alignment) — the most novel parts

## How to apply this memory in future sessions

- When user asks about reservoir computing vs deep learning, invoke three-level framework
- When user asks about "learning" in general, distinguish shaping (L3) from readout (L2)
- When user discusses Paper 1 in context of theory, place Paper 1 as L2 minimal case
- When user proposes new experiments, categorize which level(s) they test
- **Do not treat this paper as endorsed by Paper 1 authors as CET-final** — this is Evan's working synthesis, may be revised
