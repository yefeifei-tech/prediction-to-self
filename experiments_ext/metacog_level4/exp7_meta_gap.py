"""
Experiment 7: Level 3→4 Meta-Encoding-Gap (CET §13.8 extension)
================================================================
Extension to Paper 1's exp3/exp4. Tests whether the encoding gap is recursive.

Framework (CET §13.8):
  Each level of emergence requires two conditions simultaneously:
    1. Information value: I(X; S_{t+1} | existing conditions) > 0
    2. Architectural pathway: information channel from X to h
  Paper 1 established Level 2→3 via τ_t (action trace).
  This tests Level 3→4 via ρ_t (self-representation confidence trace).

Design:
  ρ_t = β · ρ_{t-1} + (1 − β) · conf(P1, h_t)
  conf(P1, h_t) = |sigmoid(P1(h_t.detach())) − 0.5| × 2
  P1 is a frozen Level-1 probe trained on the exp4-style base model.
  ρ_t is fully endogenous — no ground-truth labels required.

Three groups compared on two orthogonal metrics:

  Metric A (meta-probe, future P1 correctness):
    P2_k(h_t) → 1[P1(h_{t+k}) == is_trailing_{t+k}]
    Since h_{t+k} is not derivable from h_t (k new obs/actions intervene),
    predicting future P1 correctness requires h_t to carry meta-info about
    "how stable is my self-model right now". Tested at k ∈ {10, 30, 50}.

  Metric B (prediction MSE):
    Mean obs-prediction MSE during eval window.
    Corresponds directly to CET §13.8 condition 1: I(ρ_t; S_{t+1} | ...) > 0
    iff adding ρ_t reduces MSE.

  Both metrics must show causal > baseline AND causal > shuffled to claim
  Level 3→4 encoding gap.

Three groups:
  1. Baseline (5d, no ρ):     tests whether Level 4 emerges "for free"
  2. Causal + ρ (6d):          tests whether the pathway enables Level 4
  3. Shuffled ρ (6d):          rules out "any extra dim helps"

Run:
  python -m experiments_ext.metacog_level4.exp7_meta_gap
  python -m experiments_ext.metacog_level4.exp7_meta_gap --quick
"""

import sys, os, time, math, random, copy
from collections import deque
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch
import torch.nn as nn

QUICK = "--quick" in sys.argv
SCALE = 5 if QUICK else 1

# ---- Setup phase (mirrors exp4 curriculum) ----
SETUP_P1_STEPS = 100_000 // SCALE   # perception only
SETUP_P2_STEPS =  80_000 // SCALE   # aux head + P1 probe

# ---- Level 4 test phase ----
L4_STEPS       =  60_000 // SCALE

LR_SETUP  = 1e-3
LR_L4     = 1e-4
AUX_LR    = 5e-4
P1_LR     = 5e-4
P2_LR     = 5e-4

GAMMA         = 2.0
ACTION_RANGE  = 2.0
TRACE_BETA    = 0.95
RHO_BETA      = 0.95

AUX_WEIGHT             = 0.1
TRAILING_CLASS_WEIGHT  = 5.0
TRAILING_STEPS         = 50 // SCALE
SEED                   = 42
LOG_EVERY              = 5_000 // SCALE

BURST_MEAN = 400 // SCALE
BURST_MIN  = 100 // SCALE
BURST_MAX  = 1000 // SCALE

CLASS_ACTIVE   = 0
CLASS_TRAILING = 1
CLASS_QUIET    = 2

P2_EVAL_FRACTION = 0.5   # accumulate P2 accuracy over last fraction of L4 phase
KS = [10, 30, 50]        # future horizons for meta-probe (Metric A)


# ============================================================
# BurstGate (identical to exp4)
# ============================================================
class BurstGate:
    def __init__(self, mean_p=400, min_p=100, max_p=1000, seed=456):
        self.rng = random.Random(seed)
        self.mean_p, self.min_p, self.max_p = mean_p, min_p, max_p
        self.active = False
        self.next_switch = 0
        self._prev = False

    def step(self, t):
        self._prev = self.active
        if t >= self.next_switch:
            self.active = not self.active
            iv = int(self.rng.expovariate(1.0 / self.mean_p))
            self.next_switch = t + max(self.min_p, min(iv, self.max_p))
        return self.active

    @property
    def just_off(self):
        return self._prev and not self.active


# ============================================================
# MetaModel: 5d input (obs+τ) or 6d input (obs+τ+ρ)
# ============================================================
class MetaModel(nn.Module):
    def __init__(self, obs_dim=4, hidden_dim=192, n_scales=4, use_rho=False):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.use_rho = use_rho

        gru_input = obs_dim + 1 + (1 if use_rho else 0)
        self.gru  = nn.GRUCell(gru_input, hidden_dim)
        self.pred = nn.Linear(hidden_dim, obs_dim)

        alphas = torch.logspace(math.log10(0.02), math.log10(0.80), n_scales)
        per = hidden_dim // n_scales
        av = torch.cat([a.expand(per) for a in alphas])
        if len(av) < hidden_dim:
            av = torch.cat([av, av[-1:].expand(hidden_dim - len(av))])
        self.register_buffer("alpha", av.unsqueeze(0))

        self.h_multi = torch.zeros(1, hidden_dim)
        self.h_gru   = torch.zeros(1, hidden_dim)
        self.trace   = 0.0
        self.rho     = 0.0   # only used when use_rho=True

    def update_trace(self, action_val):
        self.trace = TRACE_BETA * self.trace + (1 - TRACE_BETA) * abs(action_val)

    def update_rho(self, conf_val):
        self.rho = RHO_BETA * self.rho + (1 - RHO_BETA) * conf_val

    def _gru_input(self, obs):
        parts = [obs.unsqueeze(0),
                 torch.tensor([[self.trace]], dtype=torch.float32)]
        if self.use_rho:
            parts.append(torch.tensor([[self.rho]], dtype=torch.float32))
        return torch.cat(parts, dim=1)

    def step_frozen(self, obs):
        pred  = self.pred(self.h_multi)
        x     = self._gru_input(obs)
        h_new = self.gru(x, self.h_gru)
        self.h_multi = ((1 - self.alpha) * self.h_multi + self.alpha * h_new).detach()
        self.h_gru   = h_new.detach()
        return pred.squeeze(0)

    def step_live(self, obs):
        pred   = self.pred(self.h_multi)
        x      = self._gru_input(obs)
        h_new  = self.gru(x, self.h_gru)
        h_live = (1 - self.alpha) * self.h_multi + self.alpha * h_new
        self.h_gru = h_new.detach()
        return pred.squeeze(0), h_live

    def commit(self, h_live):
        self.h_multi = h_live.detach()


class AuxHead(nn.Module):
    def __init__(self, hidden_dim=192):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 3)
    def forward(self, h):
        return self.fc(h)


class BinaryProbe(nn.Module):
    def __init__(self, hidden_dim=192):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, h):
        return self.fc(h).squeeze(-1)


# ============================================================
# SETUP: train exp4-style base model + train P1 to freeze
# ============================================================
def train_setup(seed=SEED, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    from core.world import make_signal
    signal = make_signal("sine")

    model = MetaModel(use_rho=False)
    aux   = AuxHead()
    p1    = BinaryProbe()
    burst = BurstGate(BURST_MEAN, BURST_MIN, BURST_MAX)

    cw       = torch.tensor([1.0, TRAILING_CLASS_WEIGHT, 1.0])
    ce_loss  = nn.CrossEntropyLoss(weight=cw)
    bce_loss = nn.BCEWithLogitsLoss()

    if verbose:
        print(f"  Setup Phase 1: Perception ({SETUP_P1_STEPS} steps)")

    opt_setup1 = torch.optim.Adam(model.parameters(), lr=LR_SETUP)
    sched      = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_setup1, T_max=SETUP_P1_STEPS, eta_min=1e-5)

    t0 = time.time()
    for step in range(SETUP_P1_STEPS):
        obs_raw    = signal.get()
        action_val = np.random.uniform(-ACTION_RANGE, ACTION_RANGE)
        model.update_trace(action_val)

        obs = obs_raw.clone()
        obs[0] = obs[0] + GAMMA * action_val

        pred = model.step_frozen(obs)
        loss = nn.functional.mse_loss(pred, obs)
        opt_setup1.zero_grad(); loss.backward(); opt_setup1.step(); sched.step()

        if verbose and step > 0 and step % LOG_EVERY == 0:
            print(f"    step {step:>6d} | MSE={loss.item():.6f} "
                  f"| trace={model.trace:.3f} | {time.time()-t0:.1f}s")

    if verbose:
        print(f"\n  Setup Phase 2: Aux head + P1 probe ({SETUP_P2_STEPS} steps)")

    opt_setup2 = torch.optim.Adam([
        {"params": model.gru.parameters(),  "lr": LR_L4},
        {"params": model.pred.parameters(), "lr": LR_L4},
        {"params": aux.parameters(),        "lr": AUX_LR},
    ])
    opt_p1 = torch.optim.Adam(p1.parameters(), lr=P1_LR)

    trailing_remaining = 0
    p1_correct = 0
    p1_total   = 0

    for step in range(SETUP_P2_STEPS):
        global_step = SETUP_P1_STEPS + step
        obs_raw = signal.get()

        ba = burst.step(global_step)
        if burst.just_off:
            trailing_remaining = TRAILING_STEPS

        if trailing_remaining > 0:
            cur_class = CLASS_TRAILING
            is_trailing = True
            trailing_remaining -= 1
        elif ba:
            cur_class = CLASS_ACTIVE
            is_trailing = False
        else:
            cur_class = CLASS_QUIET
            is_trailing = False

        action_val = np.random.uniform(-ACTION_RANGE, ACTION_RANGE) if ba else 0.0
        model.update_trace(action_val)

        obs = obs_raw.clone()
        gamma_eff = GAMMA if ba else 0.0
        obs[0] = obs[0] + gamma_eff * action_val

        pred, h_live = model.step_live(obs)
        loss_pred = nn.functional.mse_loss(pred, obs)
        logits    = aux(h_live)
        loss_aux  = AUX_WEIGHT * ce_loss(
            logits, torch.tensor([cur_class], dtype=torch.long))
        opt_setup2.zero_grad()
        (loss_pred + loss_aux).backward()
        opt_setup2.step()
        model.commit(h_live)

        h_det = model.h_multi.detach()
        logit = p1(h_det)
        opt_p1.zero_grad()
        bce_loss(logit, torch.tensor([1.0 if is_trailing else 0.0])).backward()
        opt_p1.step()

        if is_trailing:
            with torch.no_grad():
                p1_correct += int((torch.sigmoid(logit) > 0.5).item())
            p1_total += 1

        if verbose and step > 0 and step % LOG_EVERY == 0:
            tr = p1_correct / max(1, p1_total) * 100
            print(f"    step {step:>6d} | MSE={loss_pred.item():.6f} "
                  f"| P1 trailing={tr:.1f}% | {time.time()-t0:.1f}s")

    p1_recall = p1_correct / max(1, p1_total) * 100
    return model, p1, p1_recall


# ============================================================
# LEVEL 4 TEST: fine-tune with (optional) ρ channel + train P2
# ============================================================
def extend_to_6d(model_base):
    """Copy 5d model weights into a fresh 6d model; init ρ column to zero."""
    model = MetaModel(use_rho=True)
    with torch.no_grad():
        model.gru.weight_ih[:, :5] = model_base.gru.weight_ih.clone()
        model.gru.weight_ih[:, 5].zero_()
        model.gru.weight_hh.copy_(model_base.gru.weight_hh)
        model.gru.bias_ih.copy_(model_base.gru.bias_ih)
        model.gru.bias_hh.copy_(model_base.gru.bias_hh)
        model.pred.weight.copy_(model_base.pred.weight)
        model.pred.bias.copy_(model_base.pred.bias)
    model.h_multi = model_base.h_multi.clone()
    model.h_gru   = model_base.h_gru.clone()
    model.trace   = model_base.trace
    model.rho     = 0.0
    return model


def _balanced_accuracy(stats):
    """stats: {'tp','fp','tn','fn'}. Returns (acc, bal_acc, positive_rate)."""
    tp, fp, tn, fn = stats["tp"], stats["fp"], stats["tn"], stats["fn"]
    pos = tp + fn
    neg = tn + fp
    total = pos + neg
    if total == 0:
        return 0.0, 0.0, 0.0
    recall_pos = tp / max(1, pos)
    recall_neg = tn / max(1, neg)
    acc = (tp + tn) / total
    bal = (recall_pos + recall_neg) / 2
    return acc * 100, bal * 100, pos / total


def train_level4(model_base, p1_frozen, mode="baseline",
                 rho_traj_source=None, seed=SEED, verbose=True):
    """
    mode:
      'baseline' — 5d input (no ρ)
      'causal'   — 6d input, ρ from live P1(h)
      'shuffled' — 6d input, ρ drawn from rho_traj_source

    Returns dict with:
      metric_A[k]  = {'acc', 'bal_acc', 'p_pos'} for k in KS
      metric_B     = mean prediction MSE (eval window)
      rho_traj     = list of ρ values (causal mode only)
      p1_stability = frozen-P1 trailing recall on this fine-tuned model
    """
    torch.manual_seed(seed + 100); np.random.seed(seed + 100); random.seed(seed + 100)
    from core.world import make_signal
    signal = make_signal("sine")

    if mode == "baseline":
        model = copy.deepcopy(model_base)
    else:
        model = extend_to_6d(model_base)

    p1 = copy.deepcopy(p1_frozen)
    for p in p1.parameters():
        p.requires_grad_(False)

    aux = AuxHead()

    # One meta-probe P2_k per future horizon
    p2_probes = {k: BinaryProbe() for k in KS}
    p2_opts   = {k: torch.optim.Adam(p2_probes[k].parameters(), lr=P2_LR) for k in KS}
    # Running class prior estimate for P2's target (P1 correctness at t+k).
    # Used to weight BCE per-sample so P2 doesn't collapse to majority class.
    class_prior = {k: 0.7 for k in KS}
    PRIOR_EMA   = 0.005

    cw       = torch.tensor([1.0, TRAILING_CLASS_WEIGHT, 1.0])
    ce_loss  = nn.CrossEntropyLoss(weight=cw)
    bce_loss_p2 = nn.BCEWithLogitsLoss(reduction="none")   # weighted per-sample

    opt_model = torch.optim.Adam([
        {"params": model.gru.parameters(),  "lr": LR_L4},
        {"params": model.pred.parameters(), "lr": LR_L4},
        {"params": aux.parameters(),        "lr": AUX_LR},
    ])
    burst  = BurstGate(BURST_MEAN, BURST_MIN, BURST_MAX, seed=seed + 200)

    trailing_remaining = 0
    rho_traj = []
    eval_start = int(L4_STEPS * (1 - P2_EVAL_FRACTION))
    max_k = max(KS)

    # For metric A: buffer of (h_at_step, was_step_in_eval_window)
    # We store BinaryProbe predictions computed at step t; when step t+k arrives
    # we retrieve them and compare to the current step's P1 correctness.
    h_hist = deque(maxlen=max_k + 1)

    # Per-k accumulators
    a_stats = {k: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for k in KS}

    # Metric B: prediction MSE during eval window
    mse_sum = 0.0
    mse_n   = 0

    # P1 stability
    p1_correct_trail, p1_total_trail = 0, 0

    t0 = time.time()

    for step in range(L4_STEPS):
        obs_raw = signal.get()
        ba = burst.step(step)
        if burst.just_off:
            trailing_remaining = TRAILING_STEPS

        if trailing_remaining > 0:
            cur_class = CLASS_TRAILING
            is_trailing = True
            trailing_remaining -= 1
        elif ba:
            cur_class = CLASS_ACTIVE
            is_trailing = False
        else:
            cur_class = CLASS_QUIET
            is_trailing = False

        action_val = np.random.uniform(-ACTION_RANGE, ACTION_RANGE) if ba else 0.0
        model.update_trace(action_val)

        # Update ρ BEFORE the model step (mirrors trace handling)
        if mode == "causal":
            with torch.no_grad():
                p1_logit_pre = p1(model.h_multi.detach())
                p1_prob_pre  = torch.sigmoid(p1_logit_pre).item()
                conf         = abs(p1_prob_pre - 0.5) * 2.0
            model.update_rho(conf)
            rho_traj.append(model.rho)
        elif mode == "shuffled":
            model.rho = float(rho_traj_source[step % len(rho_traj_source)])

        obs = obs_raw.clone()
        gamma_eff = GAMMA if ba else 0.0
        obs[0] = obs[0] + gamma_eff * action_val

        pred, h_live = model.step_live(obs)
        loss_pred = nn.functional.mse_loss(pred, obs)
        logits    = aux(h_live)
        loss_aux  = AUX_WEIGHT * ce_loss(
            logits, torch.tensor([cur_class], dtype=torch.long))
        opt_model.zero_grad()
        (loss_pred + loss_aux).backward()
        opt_model.step()
        model.commit(h_live)

        # Record prediction MSE (Metric B) — no-grad copy of the value
        if step >= eval_start:
            mse_sum += loss_pred.item()
            mse_n   += 1

        # Read frozen P1 on the post-commit h — this is P1 correctness at CURRENT step
        with torch.no_grad():
            p1_logit = p1(model.h_multi.detach())
            p1_pred  = (torch.sigmoid(p1_logit) > 0.5).item()
        p1_correct_now = (p1_pred == is_trailing)

        # Push current h_multi into history (used as h_{t-k} by later steps)
        h_hist.append(model.h_multi.detach().clone())

        # For each k: train P2_k using h from k steps ago, target = current correctness
        target_val = 1.0 if p1_correct_now else 0.0
        for k in KS:
            if len(h_hist) > k:
                # Update running prior estimate for balanced-weighted BCE
                class_prior[k] = ((1 - PRIOR_EMA) * class_prior[k]
                                  + PRIOR_EMA * target_val)
                p = min(0.99, max(0.01, class_prior[k]))
                sample_weight = 0.5 / (p if target_val > 0.5 else (1 - p))

                h_past  = h_hist[-(k + 1)]
                target  = torch.tensor([target_val])
                p2_log  = p2_probes[k](h_past)
                loss_p2 = bce_loss_p2(p2_log, target) * sample_weight
                p2_opts[k].zero_grad()
                loss_p2.backward()
                p2_opts[k].step()

                with torch.no_grad():
                    p2_pred = (torch.sigmoid(p2_log) > 0.5).item()

                if step >= eval_start:
                    if p1_correct_now:
                        if p2_pred: a_stats[k]["tp"] += 1
                        else:       a_stats[k]["fn"] += 1
                    else:
                        if p2_pred: a_stats[k]["fp"] += 1
                        else:       a_stats[k]["tn"] += 1

        if is_trailing:
            p1_correct_trail += int(p1_pred)
            p1_total_trail   += 1

        if verbose and step > 0 and step % LOG_EVERY == 0:
            p1_r  = p1_correct_trail / max(1, p1_total_trail) * 100
            _, ba30, _ = _balanced_accuracy(a_stats[30]) if 30 in a_stats else (0, 0, 0)
            rho_dbg = f"ρ={model.rho:.3f}" if mode != "baseline" else ""
            print(f"      [{mode:<8s}] step {step:>6d} | "
                  f"P1 recall={p1_r:.1f}% | P2_k30 bal={ba30:.1f}% "
                  f"{rho_dbg} | {time.time()-t0:.1f}s")

    # Finalize
    metric_A = {}
    for k in KS:
        acc, bal, pp = _balanced_accuracy(a_stats[k])
        metric_A[k] = {"acc": acc, "bal_acc": bal, "p_pos": pp}
    metric_B = mse_sum / max(1, mse_n)
    p1_stab  = p1_correct_trail / max(1, p1_total_trail) * 100

    return {
        "metric_A":     metric_A,
        "metric_B":     metric_B,
        "rho_traj":     rho_traj,
        "p1_stability": p1_stab,
    }


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("  Experiment 7: Level 3→4 Meta-Encoding-Gap (CET §13.8)")
    print("  Extension to Paper 1 — NOT part of the paper's core results")
    if QUICK:
        print("  *** QUICK MODE ***")
    print("=" * 60)

    # ---------- Setup ----------
    print("\n  === SETUP: train exp4-style base + freeze P1 ===")
    t_setup = time.time()
    model_base, p1_frozen, p1_setup_recall = train_setup(seed=SEED)
    print(f"\n  Setup done in {time.time()-t_setup:.1f}s")
    print(f"  Level 3 sanity: P1 trailing recall = {p1_setup_recall:.1f}%")

    if p1_setup_recall < 30.0:
        print(f"\n  WARNING: P1 recall low ({p1_setup_recall:.1f}%).")
        print(f"           ρ_t will be near-constant; results may be uninformative.")

    # ---------- Group 1: Baseline (5d) ----------
    print("\n  === GROUP 1: Baseline (5d, no ρ pathway) ===")
    t = time.time()
    r_base = train_level4(model_base, p1_frozen, mode="baseline")
    print(f"  Group 1 done in {time.time()-t:.1f}s")

    # ---------- Group 2: Causal + ρ (6d) ----------
    print("\n  === GROUP 2: Causal + ρ (6d) ===")
    t = time.time()
    r_caus = train_level4(model_base, p1_frozen, mode="causal")
    print(f"  Group 2 done in {time.time()-t:.1f}s")

    # ---------- Group 3: Shuffled ρ (6d) ----------
    print("\n  === GROUP 3: Shuffled ρ (6d, same marginals, no temporal info) ===")
    shuffled_rho = list(r_caus["rho_traj"])
    random.Random(SEED + 999).shuffle(shuffled_rho)
    t = time.time()
    r_shuf = train_level4(model_base, p1_frozen, mode="shuffled",
                          rho_traj_source=shuffled_rho)
    print(f"  Group 3 done in {time.time()-t:.1f}s")

    # ---------- Report ----------
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    # -- Metric A: meta-probe at each k (balanced accuracy) --
    print(f"\n  METRIC A: Meta-probe P2_k balanced accuracy (predict P1 correct at t+k)")
    print(f"  {'Group':<24s} " + " ".join(f"k={k:<3d}" for k in KS) +
          "  P1 stab.  p_pos(k=30)")
    print(f"  {'-'*70}")
    for name, r in [("Baseline (5d)",   r_base),
                    ("Causal + ρ (6d)", r_caus),
                    ("Shuffled ρ (6d)", r_shuf)]:
        row = " ".join(f"{r['metric_A'][k]['bal_acc']:5.1f}%" for k in KS)
        print(f"  {name:<24s} {row}  {r['p1_stability']:6.1f}%    "
              f"{r_base['metric_A'][30]['p_pos']*100:5.1f}%")

    # -- Metric B: prediction MSE during eval window --
    print(f"\n  METRIC B: Mean prediction MSE (obs) during eval window")
    print(f"  {'Group':<24s} {'MSE':>10s} {'Δ vs baseline':>16s}")
    print(f"  {'-'*54}")
    print(f"  {'Baseline (5d)':<24s} {r_base['metric_B']:>10.6f} "
          f"{'—':>16s}")
    print(f"  {'Causal + ρ (6d)':<24s} {r_caus['metric_B']:>10.6f} "
          f"{r_caus['metric_B'] - r_base['metric_B']:>+16.6f}")
    print(f"  {'Shuffled ρ (6d)':<24s} {r_shuf['metric_B']:>10.6f} "
          f"{r_shuf['metric_B'] - r_base['metric_B']:>+16.6f}")

    # -- Scorecard --
    print(f"\n  --- CET §13.8 Interpretation ---")
    lifts = {k: {
        "causal":   r_caus["metric_A"][k]["bal_acc"] - r_base["metric_A"][k]["bal_acc"],
        "shuffled": r_shuf["metric_A"][k]["bal_acc"] - r_base["metric_A"][k]["bal_acc"],
    } for k in KS}
    mse_causal_gain   = r_base["metric_B"] - r_caus["metric_B"]  # positive = ρ helps
    mse_shuffled_gain = r_base["metric_B"] - r_shuf["metric_B"]

    for k in KS:
        print(f"  P2_k{k:<3d}: causal lift {lifts[k]['causal']:+5.1f} pp | "
              f"shuffled lift {lifts[k]['shuffled']:+5.1f} pp")
    print(f"  MSE:     causal reduction {mse_causal_gain:+.6f} | "
          f"shuffled reduction {mse_shuffled_gain:+.6f}")

    # Focus scorecard on k=30 (primary horizon per decision)
    K_PRIMARY = 30
    lift_c = lifts[K_PRIMARY]["causal"]
    lift_s = lifts[K_PRIMARY]["shuffled"]

    tests = [
        ("P1 stability > 40%",
            r_caus["p1_stability"] > 40.0,
            f"{r_caus['p1_stability']:.1f}%"),
        (f"A: Causal lift (k={K_PRIMARY}) > 3 pp",
            lift_c > 3.0, f"{lift_c:+.1f} pp"),
        (f"A: Causal > Shuffled + 2 pp (k={K_PRIMARY})",
            lift_c > lift_s + 2.0,
            f"{lift_c:+.1f} vs {lift_s:+.1f}"),
        ("B: MSE causal < baseline",
            mse_causal_gain > 0.0,
            f"{mse_causal_gain:+.6f}"),
        ("B: MSE causal < shuffled",
            r_caus["metric_B"] < r_shuf["metric_B"],
            f"{r_caus['metric_B']:.6f} vs {r_shuf['metric_B']:.6f}"),
    ]
    print(f"\n  SCORECARD:")
    n_pass = 0
    for name, passed, val in tests:
        st = "PASS" if passed else "FAIL"
        n_pass += int(passed)
        print(f"    [{st}] {name:<45s} {val}")
    print(f"  {n_pass}/{len(tests)} PASS")

    # Decay-curve diagnostic: does causal advantage grow with k?
    decay_c = [lifts[k]["causal"]   for k in KS]
    decay_s = [lifts[k]["shuffled"] for k in KS]
    monotone_causal = all(decay_c[i] >= decay_c[i-1] - 2 for i in range(1, len(decay_c)))
    print(f"\n  Decay pattern (bal-acc lift by k):")
    print(f"    causal   {KS} → {[f'{v:+.1f}' for v in decay_c]}")
    print(f"    shuffled {KS} → {[f'{v:+.1f}' for v in decay_s]}")
    if lift_c > 3.0 and monotone_causal:
        print(f"    → causal ρ retains advantage at longer horizons (meta-info persists)")

    if n_pass >= 4:
        print(f"\n  >> Recursive encoding gap CONFIRMED (both A and B).")
        print(f"     Level 3→4 requires its own architectural pathway (ρ_t).")
    elif n_pass >= 2 and lift_c > 3.0:
        print(f"\n  >> Partial support. Metric A shows lift but check B.")
    elif r_base["metric_A"][K_PRIMARY]["bal_acc"] > 65.0:
        print(f"\n  >> Level 4 appears 'for free' (baseline bal_acc "
              f"= {r_base['metric_A'][K_PRIMARY]['bal_acc']:.1f}%).")
        print(f"     Multi-scale EMA may already carry meta-info.")
    else:
        print(f"\n  >> Inconclusive. Check ρ_t distribution and P1 stability.")
    print("=" * 60)


if __name__ == "__main__":
    main()
