---
name: project-architecture-causal-ladder
description: "Analytical lens — each Paper 1 experiment's Pearl causal ceiling is fixed by its architecture dimensions, especially whether the prediction head takes action as an explicit input; use when discussing what causal question an experiment can/can't ask"
metadata: 
  node_type: memory
  type: project
  originSessionId: fa5a5ffe-5fdc-4de1-9e8d-44c091045032
---

**Core lens**: In Paper 1, an experiment's Pearl causal ceiling is written into its architecture — read it off the dimensions, don't infer it from results.

**The decisive dimension: does the prediction head take `action` as an explicit input?**
- **Single head `192→4`** (action NOT in prediction) → can only intervene on the **world side** (cut `obs[0] += γ·a`) → **Pearl Layer 2**. In this case any action-compensation is *implicit*, flowing through h alone (because action = W_action(h) is a deterministic function of h, so `pred(h)` can reconstruct `sine + γ·W_action(h)` without an action input). Experiments: exp2, exp4b.
- **Dual head, `pred_A: 193→4`** (action concatenated as the 193rd input dim) + `pred_B: 192→4` → can also intervene on the **input side** (falsify the 193rd dim while holding h and world fixed) → **Pearl Layer 3 counterfactual**. Experiments: exp5, exp6. Without that explicit action-input slot the counterfactual is architecturally impossible — there is nothing to swap.

**Per-experiment dims (from source):**
- exp1 `PerceptionModel`: GRU 4d, `pred 192→4` single, no action, no W_action → Layer 0 prerequisite (h stable?)
- exp2 `SingleHeadModel`: GRU 4d, `pred 192→4` single, W_action `192→1` learned → Layer 2 + confounder control
- exp3 `Model`: GRU **4d (no trace)**, `pred 192→4` single, **no W_action** (action is external burst), AuxHead 192→3 + BinaryProbe 192→1 on detached h → encoding gap, recall 12%
- exp4 `ProprioceptiveModel`: GRU **5d (obs+trace)**, else identical to exp3 → encoding breakthrough, recall 60%+
- exp4b `Model`: GRU 3d, `pred_sensory 192→3` single, W_action `192→1` **fixed buffer** (motor subset), GatedBinaryHead `[h+1]→1` on detached h → functional validation, ablation retention 94.9% vs 53.9%
- exp5 `FullModel`: GRU 5d, dual head (`pred_A 193→4`+`pred_B 192→4`), W_action learned (freeze/unfreeze) → temporal necessity, async vs sync
- exp6 `AgencyModel` (core/model.py): GRU 4d/5d (`--trace`), dual head, W_action learned → Layer 1 (AG) + Layer 3 (counterfactual spike)

**Two structural regularities worth remembering:**
1. exp3 vs exp4 differ ONLY by the 5th GRU input dim (trace). Both have NO W_action — they test *readability of h*, not the causal loop; hence they occupy the encoding-separation step, not a Pearl layer.
2. W_action has three forms mapping to three intents: **absent** (exp1/exp3/exp4, no causal loop tested) / **learned `192→1`** (exp2/exp5/exp6, action from h, tests agency) / **fixed buffer** (exp4b, frozen policy, tests self-maintenance).

The one-sentence payoff of this lens ("the 193rd action-input dim is what architecturally enables the Pearl Layer-3 counterfactual; single-head experiments are capped at Layer 2") is folded into the `Pearl 分层 ≠ 证据链分层` section of [`docs/paper1_解读手册.md`](../../docs/paper1_解读手册.md). The fuller per-experiment dimension table was NOT added to the doc — the remote's own rework (Pearl-vs-evidence-chain split + `训练模式对照：哪些权重在动` table + weight-vs-state sections) already covers that ground more finely, so this memory file is the canonical home for the full dimension table above. Complements the [[project-paper1-methodology-chain]] 8-step chain (that gives the argument; this gives the architectural reason each step lands where it does).
