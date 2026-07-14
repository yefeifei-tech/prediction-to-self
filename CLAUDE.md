# Project context for Claude Code

Code repository for **Paper 1**: *From Prediction to Self: Developmental Conditions for Agency in Minimal Neural Systems* ([arXiv:2606.05605](https://arxiv.org/abs/2606.05605)).

> **Read this file first.** It carries the persistent project context that lets you (any Claude Code instance) pick up work without a warm-up conversation.
>
> **On the first session on a new machine**: the memory system at `~/.claude/projects/<encoded-path>/memory/` starts empty. A frozen snapshot of the previous machine's memory lives in [`.claude-notes/`](.claude-notes/) (git-tracked). Read those files to catch up on past design decisions and experimental history. If the user asks you to "activate" them, copy each `.claude-notes/*.md` file (except `MEMORY.md`, which is the index) into `~/.claude/projects/<encoded-path>/memory/` and rebuild the local `MEMORY.md` index — then the memory system will pick them up on subsequent turns.

---

## Author & framework

- **Author**: Evan Ye
- Developing **CET (Conditional Emergence Theory)** — this paper is **Paper 1** of the series
- CET §13.8 core rule: each level of emergence requires **two conditions simultaneously**:
  1. **Information value**: `I(X; target | existing conditions) > 0`
  2. **Architectural pathway**: information channel must exist for X to reach the predictive computation
- When discussing extensions or design, filter proposals through this lens. Reject designs that "inject via loss" rather than "provide via input channel" — they violate CET's *conditional emergence, not forced injection* principle.

---

## Primary reference documents

Load these when the user asks about the paper's methodology or an experiment's purpose:

| Document | Purpose |
|----------|---------|
| [`docs/paper1_解读手册.md`](docs/paper1_解读手册.md) | **Authoritative** Chinese-language guide: operational definition of "self", 8-step evidence chain, per-experiment details, common misreadings, code navigation |
| [`README.md`](README.md) | English overview + installation |
| [`experiments_ext/metacog_level4/README.md`](experiments_ext/metacog_level4/README.md) | Extension experiment framework (currently paused) |
| [`papers/three_levels/`](papers/three_levels/) | In-development independent paper: three-level framework unifying reservoir computing and deep learning (notes + outline stage) |
| [`.claude-notes/`](.claude-notes/) | Frozen memory snapshot from previous machine — read on first session for design-decision history |

---

## Working rules (hard constraints)

1. **Never modify** files under [`experiments/`](experiments/) or [`core/`](core/). These are Paper 1's published code and must remain reproducible as-is.
2. **New experiments go in `experiments_ext/<name>/`**. Model variants must be defined inline in the extension file, not by editing `core/model.py`.
3. **New independent papers go in `papers/<name>/`** (e.g., `papers/three_levels/`). Use for in-development theoretical or experimental work that will become its own arxiv preprint / publication. Distinct from `docs/` (interpretation of Paper 1) and `experiments_ext/` (experimental code).
4. **Extensions belong to Paper 2+ or exploratory work**, not to Paper 1. Do not blur the line.

---

## Paper 1 in one screen

**Central claim**: A minimal 192-dim GRU develops "self-representation" only when **four conditions are jointly and sequentially satisfied**:

```
Persistent state → Causal action loop → Proprioceptive trace → Async awakening
```

**"Self" is defined operationally, not ontologically.** It is *not* a location in `h`, not a neuron, not a symbol. It is `h`'s functional relationship to the world, characterized by four measurable properties:

1. **Causal**: Agency Gain `A = Err_B − Err_A > 0`
2. **Readable**: linear probe from detached `h` reaches ≥60% trailing recall (exp4)
3. **Self-sustaining**: retention ≥90% after aux-loss ablation (exp4b, causal 94.9%)
4. **Developmental**: emerges via a specific curriculum order that cannot be permuted (exp5)

Full details in [`docs/paper1_解读手册.md`](docs/paper1_解读手册.md).

**Common misreadings to avoid**:
- ❌ "self is a thing in h" → self is a *pattern of use*, not a location
- ❌ "Paper 1 only reaches Pearl Layer 2" → exp6 counterfactual is Layer 3
- ❌ "Causal vs Control differ by AG" → Paper 1 never reports this; the diff is in long-disconnect recovery (exp2) and self-maintenance retention (exp4b)
- ❌ "encoding gap = low probe recall" → it's a conceptual claim: *causal use ≠ self-representation*
- ❌ Symbol grounding, "70%", "80.1% BA", "'I' token" → not Paper 1, belongs to v10.3 / Paper 2

---

## Repository structure

```
prediction-to-self/
├── CLAUDE.md                        # this file
├── README.md                        # public overview
├── requirements.txt                 # pip dependencies
├── train.py                         # training entry
├── core/                            # Paper 1 model + world (DO NOT MODIFY)
│   ├── model.py                     # AgencyModel: GRU + EMA + dual heads + W_action
│   ├── world.py                     # SineSignal + LorenzSignal
│   └── lorenz.py
├── experiments/                     # Paper 1 experiments (DO NOT MODIFY)
│   ├── exp1_perception.py           # §3.1 stable attractors
│   ├── exp2_causal.py               # §3.2 causal budding + Control (AR(1))
│   ├── exp3_encoding_gap.py         # §3.3 encoding gap (12% recall)
│   ├── exp4_proprioception.py       # §3.4 breakthrough (60%+ recall)
│   ├── exp4b_self_maintenance.py    # §3.2 ablation (causal 94.9% vs control 53.9%)
│   ├── exp5_async_awakening.py      # §3.5 async time-ordering
│   └── exp6_measurement.py          # §3.6 agency gain + counterfactual spike
├── figures/                         # figure generation
├── docs/                            # extended documentation
│   └── paper1_解读手册.md            # authoritative methodology guide (Chinese)
└── experiments_ext/                 # extensions beyond Paper 1
    └── metacog_level4/              # Level 3→4 meta-encoding-gap (paused)
```

---

## Environment

```bash
pip install -r requirements.txt
```

**Windows Unicode note**: The console defaults to GBK. For scripts with Unicode output (τ, ρ, →, Chinese text), invoke with:
```bash
PYTHONIOENCODING=utf-8 python -X utf8 -m experiments.exp1_perception --quick
```

**Running experiments**:
```bash
python -m experiments.exp1_perception --quick
python -m experiments.exp6_measurement --trace --lorenz --delay
python -m experiments_ext.metacog_level4.exp7_meta_gap --quick
```

---

## Session tips (for future Claude Code instances)

- When the user asks about "self", the correct framing is the **operational four-part definition**. Do not slip into metaphysical language.
- When the user says "CET" or "condition-set expansion", they mean §13.8 — filter design proposals through it.
- The paper reaches **Pearl Layer 3** (exp6 counterfactual). Do not undersell this.
- If the user references "v10.3", "symbol grounding", or "Paper 2", these are downstream work — Paper 1 doesn't cover them.
- Numbers to internalize: **12% (exp3 gap) → 60%+ (exp4 breakthrough) → 94.9% vs 53.9% (exp4b retention)**. If the user cites different numbers ("70%", "65.4%"), they may be from later versions; verify before agreeing.
