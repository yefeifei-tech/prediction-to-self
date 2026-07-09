# Level 3→4 Meta-Encoding-Gap (CET §13.8 extension)

Extension experiment to Paper 1 (*From Prediction to Self*). **Not part of Paper 1.**
Tests whether the encoding gap is recursive — whether a Level 3→4 architectural gap
exists analogous to Paper 1's Level 2→3 gap (exp3 vs exp4).

## Framework

CET §13.8 states that each level of emergence requires two conditions simultaneously:

1. **Information value**: `I(X; S_{t+1} | existing conditions) > 0`
2. **Architectural pathway**: an information channel must exist for X to reach h

Paper 1 established Level 2→3 via **τ_t** (action trace) as an endogenous channel.
This experiment tests Level 3→4 via **ρ_t** (self-representation confidence trace).

## Design

### ρ_t definition (fully endogenous, mirrors τ_t structure)

```
ρ_t = β · ρ_{t-1} + (1 − β) · conf(P1, h_t)
conf(P1, h_t) = |sigmoid(P1(h_t.detach())) − 0.5| × 2   ∈ [0, 1]
```

Where P1 is a **frozen** Level-1 probe trained on the exp4-style base model.
`ρ_t` requires no ground-truth labels — it is computed from `h_t` alone via a
fixed function (frozen P1). This is the recursive analog of τ_t (which is
computed from `a_t` alone).

### Three groups

| Group | GRU input | ρ_t source | Tests |
|-------|-----------|------------|-------|
| Baseline | 5d (obs + τ) | — | Does Level 4 come "for free" via multi-scale EMA? |
| Causal + ρ | 6d (obs + τ + ρ) | live P1 on current h | Does the pathway enable Level 4? |
| Shuffled ρ | 6d (obs + τ + ρ) | causal ρ trajectory, shuffled | Rules out "any extra dim helps" |

### Metric A: Meta-probe (future P1 correctness)

For each future horizon `k ∈ {10, 30, 50}`, train a separate probe `P2_k`:

```
P2_k(h_t) → 1[P1(h_{t+k}) == is_trailing_{t+k}]
```

Since `h_{t+k}` is not derivable from `h_t` (k new obs/actions intervene),
predicting future P1 correctness requires `h_t` to carry **meta-info about
"how stable is my self-model right now"**.

We avoid predicting current-step P1 correctness because that target is a
deterministic function of `h_t` (both `P1(h_t)` and `is_trailing_t` — the
latter via Level 3 encoding — are readable from `h_t`), which would give
trivial probe accuracy regardless of ρ_t and produce a same-tautology gap.

Balanced accuracy is reported to control for class imbalance (P1 typically
correct on ~70-80% of steps).

### Metric B: Prediction MSE

Mean obs-prediction MSE during the eval window. Corresponds directly to CET
§13.8's first condition: `I(ρ_t; S_{t+1} | h_t) > 0` iff adding ρ_t reduces
prediction MSE below baseline.

**Both metrics must show `causal > baseline` AND `causal > shuffled`** to claim
a Level 3→4 encoding gap.

## CET predictions

### Metric A (meta-probe balanced accuracy at k=30)

| Group | Expected | Reason |
|-------|----------|--------|
| Baseline | modest above chance | multi-scale EMA slow layer gives some trend info |
| Causal + ρ | significantly higher | ρ_t EMA structure provides explicit stability signal |
| Shuffled ρ | ≈ Baseline | temporal shuffle destroys info; extra dim alone doesn't help |

Also: causal's advantage should **persist across k = 10, 30, 50** (EMA
extends the temporal horizon of prediction), while shuffled advantage
should not.

### Metric B (prediction MSE)

| Group | Expected | Reason |
|-------|----------|--------|
| Baseline | reference | |
| Causal + ρ | lower than baseline | ρ_t carries `I(ρ_t; S_{t+1} \| h_t) > 0` |
| Shuffled ρ | ≈ baseline or higher | shuffled ρ is noise → may hurt slightly |

### Outcomes

- **Both metrics separate causal from baseline AND shuffled** → recursive
  encoding gap confirmed; every level of self-representation requires its own
  architectural channel.
- **Only Metric A separates them** → structural pathway exists but ρ_t doesn't
  actually help prediction; interpretation shifts toward "readable but
  unused meta-info".
- **Only Metric B separates them** → ρ_t helps prediction but doesn't create a
  readable meta-representation; interpretation shifts toward "useful signal,
  not explicit reflection".
- **Neither separates them** → multi-scale EMA already carries meta-info at
  this depth; the CET framework's recursion prediction needs revision.

## Files

- `exp7_meta_gap.py` — main experiment
- `README.md` — this file

## Run

```bash
python -m experiments_ext.metacog_level4.exp7_meta_gap
python -m experiments_ext.metacog_level4.exp7_meta_gap --quick
```

## Design decisions log

1. **P1 frozen** — otherwise ρ_t encodes "P1 training progress" instead of
   "self-model quality" (violates CET's information-purity requirement).
2. **ρ_t endogenous (confidence-based, not label-based)** — Paper 1's τ_t is
   endogenous; a label-based ρ_t would silently open an external-observation
   shortcut and demote the experiment from Level 3→4 to "extra input helps".
3. **ρ_t lagged one step** — avoids circular dependency (matches τ_t handling).
4. **Shuffled control instead of AR(1)** — the hypothesis to rule out is "any
   extra dim helps", not "is causal loop needed" (Paper 1 already handled that).
