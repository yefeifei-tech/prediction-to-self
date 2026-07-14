---
name: Paper 1 methodology — 8-step evidence chain (Pearl + CET dual framing)
description: Corrected characterization of Paper 1's experimental logic under BOTH Pearl causal-inference framing AND CET (Constraint Emergence Theory) framing; use when discussing methodology, arguing about proof structure, or extending to Paper 2+
type: project
originSessionId: 88fc7d6b-54be-4eda-af17-d7042a1ed263
---

Paper 1 ("From Prediction to Self") uses **interventional + confounder-controlled comparisons** to peel "self" apart from statistical correlation. The full evidence chain has **8 steps**, not 5. Common summaries collapse it — do not.

## 8-step chain (with dual Pearl / CET annotation)

1. **Construction (setup, not a claim)**
   `obs[0] += GAMMA * action` imposed by experimenter. System doesn't know this; must discover it.

2. **AG > 0 (observational scalar, not a causal proof)**
   `A = Err_B − Err_A`. AG > 0 is almost tautological given pred_A gets action input. Serves as **quantitative meter** for developmental progress, not causal evidence.
   - **CET**: obs-level D_KL^obs — the correlational side of C absorbed by M

3. **Pearl Layer 2 — exp2 world-side disconnect (spike test)**
   Cut `obs[0] += GAMMA*a` at test time. Only ch0 spikes; others ~1x. Proves **channel-specific** causal use.
   - **CET**: obs-level D_KL^int — first interventional probe of C's causal side (Axiom 4 required)

4. **Confounder control — exp2 Control (AR(1)) + Long-disconnect recovery**
   Match action autocorrelation and variance from external noise. Causal recovery ~74.8%, Control ~57.2%. Rules out "any statistically-matched signal works".
   - **CET**: sustained D_KL^int at obs layer + Control comparison. Recovery gap = "structured causal model vs statistical association". BUT recovery cannot discriminate A (representational decomposition) from B (retained self-correlated variable) at h-layer.

5. **Encoding separation — exp3 vs exp4 encoding gap**
   Same setup, GRU input 4d vs 5d (only trace differs). Trailing recall: ~12% vs 60%+.
   - **CET §13.8**: Level 2→3 needs (a) I(M_t; S_{t+1} | S_t, A_t) > 0 AND (b) architectural pathway. exp3 has (a) but lacks (b) — encoding gap = second condition missing. exp4 opens the pathway.
   - **NOT novel as isolated finding** (distributed representation limits are ML common knowledge). Novel: the minimum sufficient bridge (1D trace).

6. **Pearl Layer 3 — exp6 counterfactual on pred_A input**
   Feed pred_A action=0 or −action while world received real action. Proves pred_A uses action VALUE.
   - **CET**: h-side D_KL^int (counterfactual on the action input dim). Paper 1 does reach Pearl Layer 3 here — but only for pred_A subsystem, not for full self-world decomposition claim.

7. **Functional validation — exp4b self-maintenance ablation**
   After aux supervision removed, Causal retains 94.9% recall, Control collapses to 53.9%.
   - **CET §6.4-6.5**: This is **GLOBAL causality**, NOT Pearl Layer 2. It's a selection-effect argument about what maintains the causal loop, not a single-step do query. Do not attribute to Pearl.

8. **Temporal necessity — exp5 async vs sync**
   Even with all architectural pieces present, simultaneous perception+action training fails. Async succeeds.
   - **CET §6.4-6.5**: GLOBAL — time-order as identifying assumption for the loop. NOT Pearl Layer 2 (not a query about single-step effect).

## Pearl Layer attribution — corrections

Earlier framings put steps 3, 6, 7 all at "Pearl Layer 2". This is imprecise:
- **Step 3 (spike)**: Pearl Layer 2 ✓
- **Step 6 (exp6 counterfactual)**: Pearl Layer 3 (counterfactual on input) ✓
- **Step 7 (self-maintenance)**: CET GLOBAL — selection effect, not Pearl at all
- **Step 8 (async awakening)**: CET GLOBAL — identifying assumption, not Pearl

Pearl operates INSIDE the constraint C; CET operates ON C itself. Self-maintenance and async-awakening ask "what maintains the loop", which is CET-global.

## A vs B ambiguity (CET §9.8 language)

Recovery test (step 4) leaves ambiguous:
- **A (representational decomposition)**: h-layer D_KL^obs AND D_KL^int both absorbed
- **B (retained self-correlated variable)**: only D_KL^obs absorbed at h-layer

Recovery is obs-layer D_KL^int — doesn't distinguish A from B at h-layer. To close: need **h-layer D_KL^int probe** (direct do on h dimensions, then measure downstream). Paper 1 doesn't have this; future exp7+ direction.

## Corrections to common misreadings

- **AG-based Causal vs Control is not reported in Paper 1**. Control-vs-Causal appears as: exp2 long-disconnect recovery gap, and exp4b self-maintenance retention. Do not claim "Causal AG high, Control AG low".
- **Paper 1 already reaches Pearl Layer 3** via exp6 counterfactual (zero/wrong action). Later work systematizes this, does not invent it.
- **Encoding gap is a conceptual finding**, not just "low probe recall". Also NOT novel in isolation — it's the necessary consequence of §13.8's second condition missing. Novel = the minimum sufficient bridge.
- **Symbol grounding is NOT in Paper 1**. Belongs to v10.3 / Paper 2.
- **exp2 licenses "Implicit Causal Utilization", NOT "Self-World Decomposition"**. The strong "self-world decomposition" claim requires exp4's readable representation to close A vs B ambiguity. See [[project_exp2_strict_boundary]].
- **Both Causal and Control learn implicit utilization** — Causal learns *causal* utilization (action = W_action(h), deterministic from h), Control learns *statistical* utilization (AR(1) autocorrelation, 95% predictable from h). Control's partial success is NECESSARY for exp2's argument to be non-trivial.

## Probe philosophy

BinaryProbe reads `detached h`. Deliberate measurement choice: asks "does h already contain this info" (measurement) vs aux head's "force h to contain this info" (induction). Never conflate.

pred and probe are BOTH linear readouts of h. When probe fails and pred succeeds on same h, the asymmetry means info is present but not organized as a linearly-recoverable dimension — this is the encoding gap.

## Key numbers (Paper 1)

- exp3 trailing recall ≈ 12% (encoding gap)
- exp4 trailing recall ≈ 60%+ (breakthrough)
- exp4b retention: causal 94.9% vs control 53.9%
- exp2 spike (v0.3.2 setup, peak-of-5): Causal ch0 13.8x, Control ch0 ~26x (both pass channel-specificity — Control higher because AR(1) is less predictable)
- exp2 recovery: Causal 74.8% vs Control 57.2%
- exp2 current released code (mean-over-2000 with state reset): ch0 spike ≈ 1x (not shock-capturing, but recovery gap holds)

Numbers like "70% → 65.4%" or "BA 80.1% vs 64.4%" are NOT Paper 1 — later versions or symbol-grounding work.

## Cross-references

- Full CET-perspective re-reading: [`docs/paper1_解读手册.md`](../../docs/paper1_解读手册.md) §12
- Architecture causal ladder (per-experiment Pearl ceiling from architecture): [[project_architecture_causal_ladder]]
- exp2 strict boundary: [[project_exp2_strict_boundary]]
