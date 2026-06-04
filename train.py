"""
train.py — Three-phase training loop
=====================================
Usage:
    python train.py                    # sine signal, full run
    python train.py --lorenz           # lorenz signal, full run
    python train.py --quick            # sine, quick test
    python train.py --lorenz --quick   # lorenz, quick test
"""

import sys
import time
import torch
import torch.nn as nn
import numpy as np

from core.world import make_signal
from core.model import AgencyModel

# ============================================================
# Config
# ============================================================
QUICK  = "--quick" in sys.argv
LORENZ = "--lorenz" in sys.argv
SCALE  = 10 if QUICK else 1

P1_STEPS  = 100_000 // SCALE
P2A_STEPS = 60_000  // SCALE
P2B_STEPS = 60_000  // SCALE

LR           = 1e-3
ACTION_RANGE = 2.0
PERTURB_EPS  = 0.3
LOG_EVERY    = 10_000 // SCALE


# ============================================================
# One training step
# ============================================================
def train_step(model, optimizer, obs, action):
    pred_a, pred_b = model.predict(action)
    model.update_state(obs)

    loss_a = nn.functional.mse_loss(pred_a, obs)
    loss_b = nn.functional.mse_loss(pred_b, obs)
    loss = loss_a + loss_b

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss_a.item(), loss_b.item()


# ============================================================
# Action selection
# ============================================================
def select_action(model, base_action):
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


# ============================================================
# Apply action to environment
# ============================================================
def apply_action(obs_raw, action):
    obs = obs_raw.clone()
    obs[0] = obs[0] + action.item()
    return obs


# ============================================================
# Logging
# ============================================================
def log_progress(step, losses_a, losses_b, t0, gaps=None):
    if (step + 1) % LOG_EVERY != 0:
        return
    ma = np.mean(losses_a[-LOG_EVERY:])
    mb = np.mean(losses_b[-LOG_EVERY:])
    msg = f"  step {step+1:>7d} | Err_A={ma:.4f}  Err_B={mb:.4f}"
    if gaps:
        mg = np.mean(gaps[-LOG_EVERY:]) * 100
        msg += f"  gap={mg:.1f}%"
    msg += f"  | {time.time()-t0:.1f}s"
    print(msg)


# ============================================================
# Phase 1: Random actions
# ============================================================
def phase1(model, optimizer, signal, t0):
    print("=" * 50)
    print("Phase 1: Random actions")
    print("=" * 50)

    model.W_action.requires_grad_(False)
    losses_a, losses_b = [], []

    for step in range(P1_STEPS):
        obs_raw = signal.get()
        action = torch.empty(1).uniform_(-ACTION_RANGE, ACTION_RANGE)
        obs = apply_action(obs_raw, action)

        la, lb = train_step(model, optimizer, obs, action)
        losses_a.append(la)
        losses_b.append(lb)

        log_progress(step, losses_a, losses_b, t0)

    print(f"  Done. Err_A={np.mean(losses_a[-1000:]):.4f}"
          f"  Err_B={np.mean(losses_b[-1000:]):.4f}\n")


# ============================================================
# Phase 2a: Perception consolidates
# ============================================================
def phase2a(model, optimizer, signal, t0):
    print("=" * 50)
    print("Phase 2a: Perception consolidates (W_action frozen)")
    print("=" * 50)

    losses_a, losses_b = [], []

    for step in range(P2A_STEPS):
        obs_raw = signal.get()
        with torch.no_grad():
            action = model.get_action().clamp(-ACTION_RANGE, ACTION_RANGE)
        obs = apply_action(obs_raw, action)

        la, lb = train_step(model, optimizer, obs, action)
        losses_a.append(la)
        losses_b.append(lb)

        log_progress(step, losses_a, losses_b, t0)

    print(f"  Done. Err_A={np.mean(losses_a[-1000:]):.4f}"
          f"  Err_B={np.mean(losses_b[-1000:]):.4f}\n")


# ============================================================
# Phase 2b: Disagreement maximization
# ============================================================
def phase2b(model, optimizer, signal, t0):
    print("=" * 50)
    print("Phase 2b: Disagreement maximization")
    print("=" * 50)

    model.W_action.requires_grad_(True)
    losses_a, losses_b, gaps, actions_log = [], [], [], []

    for step in range(P2B_STEPS):
        obs_raw = signal.get()
        base_action = model.get_action()
        best_action = select_action(model, base_action)
        obs = apply_action(obs_raw, best_action)

        la, lb = train_step(model, optimizer, obs, best_action)
        losses_a.append(la)
        losses_b.append(lb)
        actions_log.append(best_action.item())
        if lb > 1e-6:
            gaps.append((lb - la) / lb)

        log_progress(step, losses_a, losses_b, t0, gaps)

    final_a = np.mean(losses_a[-1000:])
    final_b = np.mean(losses_b[-1000:])
    final_gap = (final_b - final_a) / final_b * 100 if final_b > 1e-6 else 0

    print(f"\n  Done. Err_A={final_a:.4f}  Err_B={final_b:.4f}")
    print(f"  Prediction gap: {final_gap:.1f}%")

    return final_a, final_b, final_gap, actions_log


# ============================================================
# Scorecard
# ============================================================
def scorecard(final_a, final_b, final_gap, actions_log):
    print(f"\n{'=' * 50}")
    print("TRAINING SCORECARD")
    print(f"{'=' * 50}")

    if len(actions_log) > 100:
        a_arr = np.array(actions_log)
        autocorr = np.corrcoef(a_arr[:-1], a_arr[1:])[0, 1]
    else:
        autocorr = 0.0

    tests = [
        ("Pred gap > 80%",        final_gap > 80,        f"{final_gap:.1f}%"),
        ("Err_A < 0.01",          final_a < 0.01,        f"{final_a:.4f}"),
        ("Err_B > 5x Err_A",      final_b > 5 * final_a, f"{final_b/final_a:.1f}x"),
        ("Action autocorr > 0.5", autocorr > 0.5,        f"{autocorr:.3f}"),
    ]

    n_pass = 0
    for name, passed, value in tests:
        status = "PASS" if passed else "FAIL"
        n_pass += int(passed)
        print(f"  [{status}] {name:30s}  {value}")
    print(f"\n  {n_pass}/{len(tests)} PASS")


# ============================================================
# Main
# ============================================================
def train():
    torch.manual_seed(42)
    np.random.seed(42)

    signal_type = "lorenz" if LORENZ else "sine"
    signal = make_signal(signal_type)

    print(f"Signal: {signal_type}")
    print(f"Steps:  P1={P1_STEPS}  P2a={P2A_STEPS}  P2b={P2B_STEPS}\n")

    model = AgencyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    t0 = time.time()

    phase1(model, optimizer, signal, t0)
    phase2a(model, optimizer, signal, t0)
    final_a, final_b, final_gap, actions_log = phase2b(model, optimizer, signal, t0)

    print(f"\n  Total time: {time.time()-t0:.1f}s")

    scorecard(final_a, final_b, final_gap, actions_log)

    torch.save(model.state_dict(), "model.pt")
    print(f"\n  Model saved to model.pt")

    return model, optimizer


if __name__ == "__main__":
    if QUICK:
        print("*** QUICK MODE ***\n")
    train()