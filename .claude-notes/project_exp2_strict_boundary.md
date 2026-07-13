---
name: project-exp2-strict-boundary
description: "exp2's Recovery test strictly licenses \"Implicit Causal Utilization\", NOT \"Self-World Decomposition\"; use when discussing what exp2 proves, the utilization-vs-representation distinction, or naming/claims for exp2"
metadata: 
  node_type: memory
  type: project
  originSessionId: fa5a5ffe-5fdc-4de1-9e8d-44c091045032
---

**Evan's own critique, endorsed:** exp2's long-disconnect Recovery test (Causal ~74.8% vs Control ~57.2%) proves a **weaker** claim than the exp2 docstring/README wording ("Implicit Self-World Decomposition"). The strict, data-aligned claim is **Implicit Causal Utilization**.

**Shared vs discriminating evidence (don't misattribute):** channel-specific **spike** (Causal ~13.8x, Control ~26x) and **ch0 prediction quality** are had by BOTH groups — Control's action also goes through `obs[0] += GAMMA·a`. They only establish the existence-level phenomenon "system learned action affects ch0"; they do NOT distinguish causal from statistical. **Both groups do implicit utilization.** The ONLY discriminator — the only thing that earns the word "causal" (vs Control's "statistical") — is **long-disconnect recovery**.

**What Recovery rules out vs leaves open:**
- Control (matched statistics: obs0 = signal + AR(1), same autocorr/variance) recovers worse → **rules out C = "any statistically-matched signal works"** (pure statistical correlation). This is exactly what the matched-stats Control is for.
- Remaining ambiguity is **A (representational decomposition: system internally distinguishes "caused by me" vs "caused by world") vs B (merely retained a self-correlated, predictively-useful internal variable)**. Recovery **cannot** separate A from B — both fit the data. (Note: the common three-way A/B/C listing is imprecise — C is already killed by Control; the live residual is A-vs-B.)

**Two equivocations that inflate "utilization" into "decomposition":**
1. The axis exp2 actually decomposes is **"predictable-from-h vs not-predictable"**. Action lands on the predictable side ONLY because the experimenter wired `action = W_action(h)` (a deterministic function of internal state). Labeling the predictable component "self" is done by the **experimenter**, not discovered by the system — the system draws no self/world boundary.
2. **Mechanistic** decomposition (pred.weight splits the prediction into signal-part + action-part) ≠ **representational/readable** decomposition (a readable "I am acting" dim in h). exp2 has at most the former.

**Code reinforcement:** `long_disconnect_test`'s disconnect phase ([exp2_causal.py:250](../../experiments/exp2_causal.py#L250)) does NOT call `get_action()` — no internal action is generated during the recovery measurement. So the recovery gap is a property of the trained `pred.weight` structure + state drift, not "internal action continuity." This *strengthens* the deflationary (utilization) reading and shows the author's "internal action still generated during disconnect" narrative is imprecise.

**Paper is internally consistent with the weaker reading:** exp3's trailing recall ≈ 12% is quantitative proof that exp2-style causal utilization is NOT readable self-representation. Structural argument (Evan's): if exp2 already proved decomposition, exp3/exp4/exp5 would be redundant — their existence reflects that exp2 did not reach self-representation.

**Preferred phrasing (experiment-conclusion aligned):** "The predictive system retains an *internally regenerable* dependence on self-generated action, beyond what matched statistics alone can explain." — internal/regenerable dependence only; no self-representation, no causal decomposition.

**Naming ladder:** exp2 = Implicit Causal Utilization → exp3 = Encoding Gap (utilization ≠ readable representation) → exp4 = Readable Self-Signal → exp4b = Self-Maintenance. Reserve "self-world decomposition / self-representation" for exp4+. The fix is terminological (downgrade exp2's wording), not a change to exp2's role — it actually sharpens why exp3/4/5 are necessary.

**Caveat kept:** "IBE not proof" applies to all empirical work; the real question is tightness. Control makes it a fairly tight IBE with one residual (A-vs-B). Nothing load-bearing breaks because the paper's formal 4-part self definition (causal + readable + self-sustaining + developmental) does not rest on exp2 — exp2 only touches the "causal" part.

Written up as the "⚠ exp2 的严格边界" box in [`docs/paper1_解读手册.md`](../../docs/paper1_解读手册.md) (exp2 section, before exp3). Complements [[project-paper1-methodology-chain]] (that chain lists exp2 as steps 2+3; this pins the epistemic ceiling of what those steps license). Related architecture lens: [[project-architecture-causal-ladder]].
