---
name: Paper 1 methodology — 8-step evidence chain (causal-inference frame)
description: Corrected characterization of Paper 1's experimental logic under a Pearl causal-inference framing; use when discussing Paper 1's methodology, arguing about proof structure, or extending to Paper 2/v10.3
type: project
originSessionId: 88fc7d6b-54be-4eda-af17-d7042a1ed263
---
Paper 1 ("From Prediction to Self") uses **interventional + confounder-controlled comparisons** to peel "self" apart from statistical correlation. The full evidence chain has **8 steps**, not 5. Common summaries collapse it — do not.

## 8-step chain

1. **Construction (setup, not a claim)**
   `obs[0] += GAMMA * action` imposed by experimenter. The system doesn't know this; it must discover it.

2. **AG > 0 (observational scalar, not a causal proof)**
   `A = Err_B − Err_A`. Since pred_A gets action as input and pred_B doesn't, AG > 0 is almost tautological given the construction. It serves as a **quantitative meter** for developmental progress, not as evidence of causal use.

3. **Pearl Layer 2 — exp2 world-side disconnect**
   Cut `obs[0] += GAMMA*a` at test time. Only ch0 spikes (>2x), other channels ~1x. Proves system learned **channel-specific** causal use, not global correlation.

4. **Confounder control — exp2 Control (AR(1))**
   Match action's autocorrelation and variance, but generate from external noise instead of h. Long-disconnect recovery: Causal recovers (regenerates action prediction from h), Control doesn't. Rules out "any signal with matching statistics would work".

5. **Encoding separation — exp3 vs exp4 encoding gap**
   Same setup, only difference: GRU input 4d (obs only) vs 5d (obs + τ_t trace). Trailing recall: ~12% vs 60%+. **Conceptual finding**: causal utilization (pred_A weights compensate for action) ≠ readable self-representation (h contains "I am acting" as a state dim). Trace bridges this specific gap.

6. **Pearl Layer 3 — exp6 counterfactual on pred_A input**
   Feed pred_A action=0 or −action while world received real action. Spike proves pred_A uses the action VALUE (not just the action being present). This is a Layer-3 counterfactual query, executed inside Paper 1 itself — Paper 1 is not limited to Layer 2.

7. **Functional validation — exp4b self-maintenance ablation**
   After aux supervision removed, Causal retains 94.9% recall, Control collapses to 53.9%. Self-representation is **self-sustaining only when causally useful**, not merely externally forced.

8. **Temporal necessity — exp5 async vs sync**
   Even with all architectural pieces present, simultaneous perception+action training fails. Async (perception first, then action, with LR drops) succeeds. **Time-order is an identifying assumption**, not a hyperparameter.

## Corrections to common misreadings

- **AG-based Causal vs Control is not reported in Paper 1**. Control-vs-Causal appears as: exp2 long-disconnect recovery gap, and exp4b self-maintenance retention (94.9% vs 53.9%). Do not claim "Causal AG high, Control AG low".
- **Paper 1 already reaches Pearl Layer 3** via exp6 counterfactual (zero/wrong action). Later work systematizes this, does not invent it.
- **Encoding gap is a conceptual finding**, not just "low probe recall". The claim is `causal use ≠ self-representation`, a categorical distinction.
- **Symbol grounding is NOT in Paper 1**. It belongs to v10.3 / Paper 2. Do not include it in Paper 1's chain.

## Probe philosophy

BinaryProbe reads `detached h`. This is a deliberate measurement choice: it asks "does h already contain this info" (measurement) as opposed to aux head's "force h to contain this info" (induction). Never conflate the two roles.

## Numbers to remember (Paper 1)

- exp3 trailing recall ≈ 12% (encoding gap)
- exp4 trailing recall ≈ 60%+ (breakthrough)
- exp4b retention: causal 94.9% vs control 53.9%
- exp2 channel-specific spike > 2x on ch0

Numbers like "70% → 65.4%" or "BA 80.1% vs 64.4%" are NOT Paper 1 — they belong to later versions or symbol-grounding work.
