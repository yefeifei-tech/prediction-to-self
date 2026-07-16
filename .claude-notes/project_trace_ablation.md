---
name: Trace ablation experiment — pending
description: Planned ablation to strengthen Paper 1 exp4's "1D trace is minimally sufficient" claim by testing if random noise or arbitrary 1D signal produces the same recall boost; currently paused
type: project
originSessionId: trace-ablation-session
---

**Status**: pre-implementation, design draft only. Paused (2026-07).

**Location**: [`experiments_ext/trace_ablation/`](../../experiments_ext/trace_ablation/) with README containing full design.

## Central question

**Paper 1 exp4 shows**: adding 1D trace (EMA of |a|) makes trailing recall jump from 12% → 60%+.

**What Paper 1 does NOT test**: whether the boost is **trace-specific** (must be action-derived signal) or **generic-1D-channel** (any extra input dimension gives similar boost).

**Distinguishing these is important because**:
- If trace-specific → "1D trace is minimally sufficient" claim holds; proprioception analogy is strict
- If generic-1D → claim weakens to "any extra input helps"; encoding gap is more about capacity than specificity

## Three ablations planned

1. **Random noise channel**: replace trace with iid random noise → does recall stay at 60%+?
2. **AR(1) noise channel**: replace trace with AR(1) noise (has temporal structure but not action-derived) → separates "temporal structure" from "action content"
3. **β sweep**: keep trace as EMA of |a| but vary EMA time constant β ∈ {0.5, 0.8, 0.9, 0.95, 0.99} → tests whether trace timescale needs to match trailing window

## Expected outcomes

- **Best case for Paper 1**: only actual trace works → validates trace-specificity claim
- **Worst case for Paper 1**: any 1D channel works → forces revision of "trace is minimally sufficient" to "1D channel is minimally sufficient"
- **Interesting middle**: AR(1) works partially → temporal structure matters + action content contributes independently

## Implementation

- Fork [experiments/exp4_proprioception.py](../../experiments/exp4_proprioception.py)
- Change only the `update_trace` method (one place)
- Keep everything else identical (same seed, same hyperparams, same training)
- Estimated runtime: ~3 hours for full 8-run sweep

## Relationship to other pending work

- **experiments_ext/metacog_level4/**: Level 3→4 recursive self-model (paused). If restarted, trace_ablation is a natural companion cleanup.
- **papers/three_levels/**: results would clarify what "L2 architecture sufficient condition" means precisely — is it "1D channel of any kind" or "trace-specific"?

## When to trigger

- Evan has ~3 hour work session available
- Or specific writing/review context needs this ablation
- Or exp7 metacog_level4 restart

## Priority

Medium — doesn't block Paper 1 published content but would substantially strengthen rigor discussions (especially for ML audience).

## Cross-references

- Full design: [experiments_ext/trace_ablation/README.md](../../experiments_ext/trace_ablation/README.md)
- Related deep read: [docs/deep_reads/exp3_exp4_encoding_gap.md](../../docs/deep_reads/exp3_exp4_encoding_gap.md) §3.4
- Origin discussion: 2026-07 exp3/exp4 study session (user pushback on "trace makes probe trivial")
