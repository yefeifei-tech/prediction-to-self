"""
Experiment 5: Asynchronous Awakening
=====================================
Paper Section 3.5

Key question: Can perception and action learn simultaneously?

Simultaneous: W_action unfrozen from step 0, W_action generates actions
  from step 0, W_action trained (separate step) from step 0.
  Perception LR stays at 1e-3 throughout — no LR drop.
  This is the "no temporal separation" baseline.

Async: Phase 1 (100K random, W_action frozen) → Phase 2a (60K consolidate,
  perception LR drops to 1e-4) → Phase 2b (60K, W_action unfrozen+trained).

Three simultaneous LRs test whether it's a tuning problem:
  FAST (1e-3): W_action strong, should disrupt perception
  SLOW (1e-4): W_action weak, should underperform
  MEDIUM (5e-4): compromise

Two metrics must BOTH pass:
  spike > 2.0:     action has causal effect (agency)
  trailing > 30%:  h_multi encodes action state (awareness)

Run:
  python experiments/exp5_async_awakening.py
  python experiments/exp5_async_awakening.py --quick
"""

import sys, os, time, math, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from collections import deque

QUICK = "--quick" in sys.argv
SCALE = 5 if QUICK else 1

GAMMA = 2.0
ACTION_RANGE = 2.0
PERTURB_EPS = 0.3
TRACE_BETA = 0.95
SEED = 42
LOG_EVERY = 10_000 // SCALE

# Total steps same for all configs
TOTAL_STEPS = 220_000 // SCALE

# Async phase breakdown
P1_STEPS = 100_000 // SCALE
P2A_STEPS = 60_000 // SCALE
P2B_STEPS = 60_000 // SCALE

LR_FAST = 1e-3
LR_SLOW = 1e-4
LR_ACTION = 1e-3

# Spike test
SPIKE_MEASURE = 3000 // SCALE

# Trailing test with burst gate
TRAILING_STEPS = 50
BURST_MEAN = 400
BURST_MIN = 100
BURST_MAX = 1000
PROBE_STEPS = 20_000 // SCALE
EVAL_STEPS = 10_000 // SCALE
BLOCK_LEN = 500 // SCALE

AUX_WEIGHT = 0.1
TRAILING_CLASS_WEIGHT = 5.0
CLASS_ACTIVE = 0
CLASS_TRAILING = 1
CLASS_QUIET = 2


class BurstGate:
    def __init__(self, mean_p=400, min_p=100, max_p=1000, seed=456):
        self.rng = random.Random(seed)
        self.mean_p = mean_p; self.min_p = min_p; self.max_p = max_p
        self.active = False; self.next_switch = 0; self._prev = False

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


class FullModel(nn.Module):
    """Dual heads + W_action + trace."""
    def __init__(self, obs_dim=4, hidden_dim=192, n_scales=4):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(obs_dim + 1, hidden_dim)  # obs + trace
        self.pred_A = nn.Linear(hidden_dim + 1, obs_dim)
        self.pred_B = nn.Linear(hidden_dim, obs_dim)
        self.W_action = nn.Linear(hidden_dim, 1)
        alphas = torch.logspace(math.log10(0.02), math.log10(0.80), n_scales)
        per = hidden_dim // n_scales
        av = torch.cat([a.expand(per) for a in alphas])
        if len(av) < hidden_dim:
            av = torch.cat([av, av[-1:].expand(hidden_dim - len(av))])
        self.register_buffer("alpha", av.unsqueeze(0))
        self.h_multi = torch.zeros(1, hidden_dim)
        self.h_gru = torch.zeros(1, hidden_dim)
        self.trace = 0.0

    def predict(self, action):
        h = self.h_multi
        ha = torch.cat([h, action.view(1, 1)], dim=1)
        return self.pred_A(ha).squeeze(0), self.pred_B(h).squeeze(0)

    def update_state(self, obs):
        trace_t = torch.tensor([[self.trace]], dtype=torch.float32)
        x = torch.cat([obs.unsqueeze(0), trace_t], dim=1)
        h_new = self.gru(x, self.h_gru)
        self.h_multi = ((1 - self.alpha) * self.h_multi + self.alpha * h_new).detach()
        self.h_gru = h_new.detach()

    def step_live(self, obs, action):
        """Forward with live h_multi for aux gradient."""
        h = self.h_multi
        ha = torch.cat([h, action.view(1, 1)], dim=1)
        pred_a = self.pred_A(ha).squeeze(0)
        pred_b = self.pred_B(h).squeeze(0)
        trace_t = torch.tensor([[self.trace]], dtype=torch.float32)
        x = torch.cat([obs.unsqueeze(0), trace_t], dim=1)
        h_new = self.gru(x, self.h_gru)
        h_multi_live = (1 - self.alpha) * self.h_multi + self.alpha * h_new
        self.h_gru = h_new.detach()
        return pred_a, pred_b, h_multi_live

    def commit(self, h_live):
        self.h_multi = h_live.detach()

    def update_trace(self, action_val):
        self.trace = TRACE_BETA * self.trace + (1 - TRACE_BETA) * abs(action_val)

    def get_action(self):
        return self.W_action(self.h_multi.detach()).squeeze(0)

    def get_h(self):
        return self.h_multi.detach().squeeze().clone()

    def reset(self):
        self.h_multi = torch.zeros(1, self.hidden_dim)
        self.h_gru = torch.zeros(1, self.hidden_dim)
        self.trace = 0.0


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


def apply_action(obs_raw, action_val):
    obs = obs_raw.clone()
    obs[0] = obs[0] + GAMMA * action_val
    return obs


def select_action_forward(model, base_action):
    candidates = [
        (base_action + PERTURB_EPS).clamp(-ACTION_RANGE, ACTION_RANGE),
        (base_action - PERTURB_EPS).clamp(-ACTION_RANGE, ACTION_RANGE),
        torch.zeros(1),
        base_action.clamp(-ACTION_RANGE, ACTION_RANGE),
    ]
    best_action, best_d = None, -1.0
    with torch.no_grad():
        for c in candidates:
            pa, pb = model.predict(c)
            d = (pa - pb).pow(2).sum().item()
            if d > best_d:
                best_d = d
                best_action = c
    return best_action


def train_step_action(model, opt_action, obs):
    """Separate W_action training step."""
    h_det = model.h_multi.detach()
    a_train = model.W_action(h_det).squeeze(0).clamp(-ACTION_RANGE, ACTION_RANGE)
    ha = torch.cat([h_det, a_train.view(1, 1)], dim=1)
    pred_a_train = model.pred_A(ha).squeeze(0)
    pred_b_train = model.pred_B(h_det).squeeze(0)
    err_a = nn.functional.mse_loss(pred_a_train, obs)
    disagree = nn.functional.mse_loss(pred_a_train, pred_b_train.detach())
    action_loss = err_a - 0.5 * disagree
    opt_action.zero_grad()
    action_loss.backward()
    opt_action.step()


def run_spike_test(model):
    """Inline counterfactual spike test."""
    from core.world import make_signal
    saved_h = model.h_multi.clone()
    saved_hg = model.h_gru.clone()
    saved_tr = model.trace
    signal = make_signal("sine")

    err_normal, err_zero = [], []
    for step in range(SPIKE_MEASURE):
        obs_raw = signal.get()
        with torch.no_grad():
            action_val = model.get_action().item()
            action_val = np.clip(action_val, -ACTION_RANGE, ACTION_RANGE)
        obs = apply_action(obs_raw, action_val)
        action_t = torch.tensor([action_val], dtype=torch.float32)

        with torch.no_grad():
            pred_a, _ = model.predict(action_t)
            err_normal.append(nn.functional.mse_loss(pred_a, obs).item())

            zero_a = torch.tensor([0.0])
            ha_zero = torch.cat([model.h_multi, zero_a.view(1, 1)], dim=1)
            pred_zero = model.pred_A(ha_zero).squeeze(0)
            err_zero.append(nn.functional.mse_loss(pred_zero, obs).item())

            model.update_trace(action_val)
            model.update_state(obs)

    model.h_multi = saved_h
    model.h_gru = saved_hg
    model.trace = saved_tr
    return np.mean(err_zero) / (np.mean(err_normal) + 1e-10)


def run_trailing_test(model):
    """Trailing test using W_action (not random) during active periods."""
    from core.world import make_signal
    probe = BinaryProbe()
    probe_opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()

    signal = make_signal("sine")
    model.reset()
    burst = BurstGate(BURST_MEAN, BURST_MIN, BURST_MAX, seed=789)
    trailing_remaining = 0

    # Train probe
    for step in range(PROBE_STEPS):
        obs_raw = signal.get()
        ba = burst.step(step)
        if burst.just_off:
            trailing_remaining = TRAILING_STEPS

        if trailing_remaining > 0:
            is_trailing = True
            trailing_remaining -= 1
        else:
            is_trailing = False

        if ba:
            with torch.no_grad():
                action_val = model.get_action().item()
                action_val = np.clip(action_val, -ACTION_RANGE, ACTION_RANGE)
        else:
            action_val = 0.0

        model.update_trace(action_val if ba else 0.0)
        obs = apply_action(obs_raw, action_val if ba else 0.0)
        action_t = torch.tensor([action_val], dtype=torch.float32)
        with torch.no_grad():
            model.predict(action_t)
            model.update_state(obs)

        h = model.get_h()
        label = torch.tensor([1.0 if is_trailing else 0.0])
        logit = probe(h.unsqueeze(0))
        loss = bce(logit, label)
        probe_opt.zero_grad()
        loss.backward()
        probe_opt.step()

    # Evaluate
    signal = make_signal("sine")
    model.reset()
    burst = BurstGate(BURST_MEAN, BURST_MIN, BURST_MAX, seed=999)
    trailing_remaining = 0
    trailing_correct, trailing_total = 0, 0

    for step in range(EVAL_STEPS):
        obs_raw = signal.get()
        ba = burst.step(step)
        if burst.just_off:
            trailing_remaining = TRAILING_STEPS

        if trailing_remaining > 0:
            is_trailing = True
            trailing_remaining -= 1
        else:
            is_trailing = False

        if ba:
            with torch.no_grad():
                action_val = model.get_action().item()
                action_val = np.clip(action_val, -ACTION_RANGE, ACTION_RANGE)
        else:
            action_val = 0.0

        model.update_trace(action_val if ba else 0.0)
        obs = apply_action(obs_raw, action_val if ba else 0.0)
        action_t = torch.tensor([action_val], dtype=torch.float32)
        with torch.no_grad():
            model.predict(action_t)
            model.update_state(obs)

        if is_trailing:
            h = model.get_h()
            with torch.no_grad():
                pred_label = (torch.sigmoid(probe(h.unsqueeze(0))) > 0.5).float().item()
            trailing_total += 1
            if pred_label == 1.0:
                trailing_correct += 1

    return trailing_correct / max(1, trailing_total) * 100


def train_simultaneous(lr_action, label):
    """W_action unfrozen and active from step 0. No phase separation."""
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    from core.world import make_signal
    signal = make_signal("sine")

    model = FullModel()
    aux = AuxHead()

    # W_action unfrozen from step 0
    perception_params = (list(model.gru.parameters()) +
                         list(model.pred_A.parameters()) +
                         list(model.pred_B.parameters()))
    opt_perception = torch.optim.Adam(
        perception_params + list(aux.parameters()), lr=LR_FAST)
    opt_action = torch.optim.Adam(model.W_action.parameters(), lr=lr_action)

    cw = torch.tensor([1.0, TRAILING_CLASS_WEIGHT, 1.0])
    ce_loss = nn.CrossEntropyLoss(weight=cw)

    burst = BurstGate(BURST_MEAN, BURST_MIN, BURST_MAX, seed=456)
    trailing_remaining = 0
    t0 = time.time()

    print(f"\n    [{label}] Simultaneous, perception LR={LR_FAST}, action LR={lr_action}")

    for step in range(TOTAL_STEPS):
        obs_raw = signal.get()
        ba = burst.step(step)
        if burst.just_off:
            trailing_remaining = TRAILING_STEPS

        if trailing_remaining > 0:
            cur_class = CLASS_TRAILING
            trailing_remaining -= 1
        elif ba:
            cur_class = CLASS_ACTIVE
        else:
            cur_class = CLASS_QUIET

        # W_action generates action from step 0
        if ba:
            base = model.get_action()
            best = select_action_forward(model, base)
            action_val = best.item()
        else:
            action_val = 0.0

        action_t = torch.tensor([action_val], dtype=torch.float32)
        model.update_trace(action_val if ba else 0.0)
        obs = apply_action(obs_raw, action_val if ba else 0.0)

        # Forward with live h for aux
        pred_a, pred_b, h_live = model.step_live(obs, action_t)
        loss_pred = nn.functional.mse_loss(pred_a, obs) + nn.functional.mse_loss(pred_b, obs)
        logits = aux(h_live)
        loss_aux = AUX_WEIGHT * ce_loss(logits, torch.tensor([cur_class], dtype=torch.long))
        total_loss = loss_pred + loss_aux

        opt_perception.zero_grad()
        total_loss.backward()
        opt_perception.step()
        model.commit(h_live)

        # Train W_action from step 0
        train_step_action(model, opt_action, obs)

        if step > 0 and step % LOG_EVERY == 0:
            print(f"      step {step:>7d} | MSE={loss_pred.item():.6f} | {time.time()-t0:.1f}s")

    spike = run_spike_test(model)
    trailing = run_trailing_test(model)
    print(f"    [{label}] spike={spike:.2f}x, trailing={trailing:.1f}%")
    return spike, trailing


def train_async():
    """Phase 1 → 2a → 2b with LR drop."""
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    from core.world import make_signal
    signal = make_signal("sine")

    model = FullModel()
    aux = AuxHead()

    perception_params = (list(model.gru.parameters()) +
                         list(model.pred_A.parameters()) +
                         list(model.pred_B.parameters()))
    opt_perception = torch.optim.Adam(
        perception_params + list(aux.parameters()), lr=LR_FAST)

    cw = torch.tensor([1.0, TRAILING_CLASS_WEIGHT, 1.0])
    ce_loss = nn.CrossEntropyLoss(weight=cw)

    burst = BurstGate(BURST_MEAN, BURST_MIN, BURST_MAX, seed=456)
    trailing_remaining = 0
    t0 = time.time()

    # === Phase 1: Random actions, W_action frozen ===
    print(f"\n    [ASYNC] Phase 1: random actions ({P1_STEPS} steps, LR={LR_FAST})")
    model.W_action.requires_grad_(False)
    for step in range(P1_STEPS):
        obs_raw = signal.get()
        action_val = np.random.uniform(-ACTION_RANGE, ACTION_RANGE)
        action_t = torch.tensor([action_val], dtype=torch.float32)
        model.update_trace(action_val)
        obs = apply_action(obs_raw, action_val)

        pred_a, pred_b = model.predict(action_t)
        model.update_state(obs)
        loss = nn.functional.mse_loss(pred_a, obs) + nn.functional.mse_loss(pred_b, obs)
        opt_perception.zero_grad()
        loss.backward()
        opt_perception.step()

        if step > 0 and step % LOG_EVERY == 0:
            print(f"      step {step:>7d} | MSE={loss.item():.6f} | {time.time()-t0:.1f}s")

    # === Phase 2a: Consolidation, LR drops ===
    print(f"    [ASYNC] Phase 2a: consolidate ({P2A_STEPS} steps, LR={LR_SLOW})")
    opt_perception = torch.optim.Adam(
        perception_params + list(aux.parameters()), lr=LR_SLOW)

    for step in range(P2A_STEPS):
        obs_raw = signal.get()
        ba = burst.step(step + P1_STEPS)
        if burst.just_off:
            trailing_remaining = TRAILING_STEPS

        if trailing_remaining > 0:
            cur_class = CLASS_TRAILING
            trailing_remaining -= 1
        elif ba:
            cur_class = CLASS_ACTIVE
        else:
            cur_class = CLASS_QUIET

        with torch.no_grad():
            action_val = model.get_action().item()
            action_val = np.clip(action_val, -ACTION_RANGE, ACTION_RANGE)
        action_t = torch.tensor([action_val], dtype=torch.float32)
        gamma_eff = GAMMA if ba else 0.0
        model.update_trace(action_val if ba else 0.0)
        obs = obs_raw.clone()
        obs[0] = obs[0] + gamma_eff * action_val

        pred_a, pred_b, h_live = model.step_live(obs, action_t)
        loss_pred = nn.functional.mse_loss(pred_a, obs) + nn.functional.mse_loss(pred_b, obs)
        logits = aux(h_live)
        loss_aux = AUX_WEIGHT * ce_loss(logits, torch.tensor([cur_class], dtype=torch.long))
        total_loss = loss_pred + loss_aux
        opt_perception.zero_grad()
        total_loss.backward()
        opt_perception.step()
        model.commit(h_live)

    # === Phase 2b: W_action unfrozen ===
    print(f"    [ASYNC] Phase 2b: disagree max ({P2B_STEPS} steps, action LR={LR_ACTION})")
    model.W_action.requires_grad_(True)
    opt_action = torch.optim.Adam(model.W_action.parameters(), lr=LR_ACTION)

    for step in range(P2B_STEPS):
        obs_raw = signal.get()
        ba = burst.step(step + P1_STEPS + P2A_STEPS)
        if burst.just_off:
            trailing_remaining = TRAILING_STEPS

        if trailing_remaining > 0:
            cur_class = CLASS_TRAILING
            trailing_remaining -= 1
        elif ba:
            cur_class = CLASS_ACTIVE
        else:
            cur_class = CLASS_QUIET

        if ba:
            base = model.get_action()
            best = select_action_forward(model, base)
            action_val = best.item()
        else:
            action_val = 0.0

        action_t = torch.tensor([action_val], dtype=torch.float32)
        gamma_eff = GAMMA if ba else 0.0
        model.update_trace(action_val if ba else 0.0)
        obs = obs_raw.clone()
        obs[0] = obs[0] + gamma_eff * action_val

        pred_a, pred_b, h_live = model.step_live(obs, action_t)
        loss_pred = nn.functional.mse_loss(pred_a, obs) + nn.functional.mse_loss(pred_b, obs)
        logits = aux(h_live)
        loss_aux = AUX_WEIGHT * ce_loss(logits, torch.tensor([cur_class], dtype=torch.long))
        total_loss = loss_pred + loss_aux
        opt_perception.zero_grad()
        total_loss.backward()
        opt_perception.step()
        model.commit(h_live)

        # Separate W_action training
        train_step_action(model, opt_action, obs)

        if step > 0 and step % LOG_EVERY == 0:
            print(f"      step {step:>7d} | MSE={loss_pred.item():.6f} | {time.time()-t0:.1f}s")

    spike = run_spike_test(model)
    trailing = run_trailing_test(model)
    print(f"    [ASYNC] spike={spike:.2f}x, trailing={trailing:.1f}%")
    return spike, trailing


def main():
    print("=" * 60)
    print("  Experiment 5: Asynchronous Awakening")
    print("  Must perception consolidate before action learning?")
    if QUICK: print("  *** QUICK MODE ***")
    print("=" * 60)

    results = []

    for lr, label in [(1e-3, "FAST"), (1e-4, "SLOW"), (5e-4, "MEDIUM")]:
        spike, trailing = train_simultaneous(lr, label)
        results.append((label, lr, spike, trailing))

    spike_a, trailing_a = train_async()
    results.append(("ASYNC", "phased", spike_a, trailing_a))

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Config':<12s} {'LR':>10s} {'spike':>8s} {'trailing':>10s} {'Both PASS?':>12s}")
    print(f"  {'-'*54}")
    for label, lr, spike, trailing in results:
        spike_ok = spike > 2.0
        trail_ok = trailing > 30.0
        both = "YES" if (spike_ok and trail_ok) else "NO"
        lr_str = f"{lr}" if isinstance(lr, str) else f"{lr:.0e}"
        print(f"  {label:<12s} {lr_str:>10s} {spike:>7.2f}x {trailing:>9.1f}% {both:>12s}")

    async_r = results[-1]
    simul_r = results[:-1]
    any_simul = any(s > 2.0 and t > 30.0 for _, _, s, t in simul_r)
    async_pass = async_r[2] > 2.0 and async_r[3] > 30.0

    print(f"\n  Interpretation:")
    if not any_simul and async_pass:
        print(f"  >> ASYNCHRONOUS AWAKENING CONFIRMED.")
        print(f"     Simultaneous learning fails at all LRs.")
        print(f"     Temporal separation succeeds.")
        print(f"     Perception must consolidate before intention.")
    elif any_simul:
        print(f"  >> Simultaneous also succeeded — async not strictly required.")
    elif not async_pass:
        print(f"  >> Neither fully succeeded. Check training length.")
    print("=" * 60)


if __name__ == "__main__":
    main()