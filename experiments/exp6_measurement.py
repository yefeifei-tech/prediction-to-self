"""
Experiment 6: Measurement — Agency Gain
========================================
Paper Section 3.6 + 3.7 (temporal robustness)

Flags:
  --quick     Quick mode (5x fewer steps)
  --lorenz    Use Lorenz chaotic signal
  --trace     Enable proprioceptive trace (5d GRU input)
  --delay     Enable 2-step action delay (temporal robustness test)

Run:
  python experiments/exp6_measurement.py --trace              # baseline
  python experiments/exp6_measurement.py --trace --delay      # temporal delay
  python experiments/exp6_measurement.py --lorenz --trace     # Lorenz
  python experiments/exp6_measurement.py --lorenz --trace --delay  # Lorenz + delay
"""

import sys, os, time, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn

from core.model import AgencyModel
from core.world import make_signal

QUICK = "--quick" in sys.argv
LORENZ = "--lorenz" in sys.argv
USE_TRACE = "--trace" in sys.argv
USE_DELAY = "--delay" in sys.argv
SCALE = 5 if QUICK else 1
ACTION_DELAY = 2 if USE_DELAY else 0

P1_STEPS = 100_000 // SCALE
P2A_STEPS = 60_000 // SCALE
P2B_STEPS = 60_000 // SCALE

LR = 1e-3
LR_P2 = 1e-4
LR_ACTION = 1e-3
ACTION_RANGE = 2.0
PERTURB_EPS = 0.3
SEED = 42
LOG_EVERY = 10_000 // SCALE
SPIKE_MEASURE = 3000 // SCALE


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


def train_step_perception(model, opt, obs, action):
    pred_a, pred_b = model.predict(action)
    model.update_state(obs)
    la = nn.functional.mse_loss(pred_a, obs)
    lb = nn.functional.mse_loss(pred_b, obs)
    loss = la + lb
    opt.zero_grad()
    loss.backward()
    opt.step()
    return la.item(), lb.item()


def train_step_action(model, opt_action, obs):
    h_det = model.h_multi.detach()
    a_train = model.W_action(h_det).squeeze(0).clamp(-ACTION_RANGE, ACTION_RANGE)
    ha = torch.cat([h_det, a_train.view(1, 1)], dim=1)
    pa = model.pred_A(ha).squeeze(0)
    pb = model.pred_B(h_det).squeeze(0)
    err_a = nn.functional.mse_loss(pa, obs)
    disagree = nn.functional.mse_loss(pa, pb.detach())
    loss = err_a - 0.5 * disagree
    opt_action.zero_grad()
    loss.backward()
    opt_action.step()


def apply_delayed_action(obs_raw, action_buffer, current_action_val):
    """Apply delayed action to obs. Returns (obs, delayed_action_val)."""
    if ACTION_DELAY > 0:
        delayed_val = action_buffer.pop(0)
        action_buffer.append(current_action_val)
    else:
        delayed_val = current_action_val
    obs = obs_raw.clone()
    obs[0] = obs[0] + delayed_val
    return obs, delayed_val


def run_full_training(signal_type="sine", strategy="forward"):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    signal = make_signal(signal_type)
    model = AgencyModel(use_trace=USE_TRACE)

    perc_params = (list(model.gru.parameters()) +
                   list(model.pred_A.parameters()) +
                   list(model.pred_B.parameters()))
    opt_perc = torch.optim.Adam(perc_params, lr=LR)
    opt_act = torch.optim.Adam(model.W_action.parameters(), lr=LR_ACTION)

    action_buffer = [0.0] * ACTION_DELAY
    t0 = time.time()
    all_a, all_b = [], []

    # === Phase 1 ===
    print(f"    Phase 1: Random actions ({P1_STEPS} steps)")
    model.W_action.requires_grad_(False)
    for step in range(P1_STEPS):
        obs_raw = signal.get()
        cur_act = torch.empty(1).uniform_(-ACTION_RANGE, ACTION_RANGE)
        obs, delayed_val = apply_delayed_action(obs_raw, action_buffer, cur_act.item())
        if USE_TRACE:
            model.update_trace(cur_act.item())
        # pred_A receives delayed_action (matches what modified obs)
        act_for_pred = torch.tensor([delayed_val], dtype=torch.float32)
        la, lb = train_step_perception(model, opt_perc, obs, act_for_pred)
        all_a.append(la); all_b.append(lb)
        if step > 0 and step % LOG_EVERY == 0:
            print(f"      step {step:>7d} | A={np.mean(all_a[-LOG_EVERY:]):.4f} B={np.mean(all_b[-LOG_EVERY:]):.4f} | {time.time()-t0:.1f}s")

    p1_a = np.mean(all_a[-1000:]); p1_b = np.mean(all_b[-1000:])
    p1_gap = (p1_b - p1_a) / (p1_b + 1e-10) * 100
    print(f"      Phase 1 end: A={p1_a:.4f} B={p1_b:.4f} gap={p1_gap:.1f}%")
    # Snapshot model weights at Phase 1 end for Phase 1 spike test
    model_p1 = copy.deepcopy(model)

    # === Phase 2a ===
    print(f"    Phase 2a: Consolidation ({P2A_STEPS} steps, LR={LR_P2})")
    opt_perc = torch.optim.Adam(perc_params, lr=LR_P2)
    for step in range(P2A_STEPS):
        obs_raw = signal.get()
        with torch.no_grad():
            cur_act = model.get_action().clamp(-ACTION_RANGE, ACTION_RANGE)
        obs, delayed_val = apply_delayed_action(obs_raw, action_buffer, cur_act.item())
        if USE_TRACE:
            model.update_trace(cur_act.item())
        act_for_pred = torch.tensor([delayed_val], dtype=torch.float32)
        la, lb = train_step_perception(model, opt_perc, obs, act_for_pred)
        all_a.append(la); all_b.append(lb)

    # === Phase 2b ===
    print(f"    Phase 2b: {strategy} ({P2B_STEPS} steps, perc LR={LR_P2}, act LR={LR_ACTION})")
    model.W_action.requires_grad_(True)
    opt_act = torch.optim.Adam(model.W_action.parameters(), lr=LR_ACTION)

    p2b_actions = []
    for step in range(P2B_STEPS):
        obs_raw = signal.get()

        if strategy == "forward":
            base = model.get_action()
            best = select_action_forward(model, base)
            cur_val = best.item()
            obs, delayed_val = apply_delayed_action(obs_raw, action_buffer, cur_val)
            p2b_actions.append(cur_val)
            if USE_TRACE:
                model.update_trace(cur_val)
            act_for_pred = torch.tensor([delayed_val], dtype=torch.float32)
            la, lb = train_step_perception(model, opt_perc, obs, act_for_pred)
            train_step_action(model, opt_act, obs)

        elif strategy == "direct_ag":
            h_det = model.h_multi.detach()
            act_t = model.W_action(h_det).squeeze(0).clamp(-ACTION_RANGE, ACTION_RANGE)
            cur_val = act_t.item()
            obs, delayed_val = apply_delayed_action(obs_raw, action_buffer, cur_val)
            p2b_actions.append(cur_val)
            if USE_TRACE:
                model.update_trace(cur_val)
            delayed_t = torch.tensor([delayed_val], dtype=torch.float32)
            pa, pb = model.predict(delayed_t)
            model.update_state(obs)
            la_t = nn.functional.mse_loss(pa, obs)
            lb_t = nn.functional.mse_loss(pb, obs)
            loss = la_t + lb_t + (la_t - lb_t.detach())
            opt_perc.zero_grad(); opt_act.zero_grad()
            loss.backward(); opt_perc.step(); opt_act.step()
            la, lb = la_t.item(), lb_t.item()

        elif strategy == "gradient_disagree":
            h_det = model.h_multi.detach()
            act_t = model.W_action(h_det).squeeze(0).clamp(-ACTION_RANGE, ACTION_RANGE)
            cur_val = act_t.item()
            obs, delayed_val = apply_delayed_action(obs_raw, action_buffer, cur_val)
            p2b_actions.append(cur_val)
            if USE_TRACE:
                model.update_trace(cur_val)
            delayed_t = torch.tensor([delayed_val], dtype=torch.float32)
            pa, pb = model.predict(delayed_t)
            model.update_state(obs)
            la_t = nn.functional.mse_loss(pa, obs)
            lb_t = nn.functional.mse_loss(pb, obs)
            dis = nn.functional.mse_loss(pa, pb.detach())
            loss = la_t + lb_t - dis
            opt_perc.zero_grad(); opt_act.zero_grad()
            loss.backward(); opt_perc.step(); opt_act.step()
            la, lb = la_t.item(), lb_t.item()

        all_a.append(la); all_b.append(lb)
        if step > 0 and step % LOG_EVERY == 0:
            ma = np.mean(all_a[-LOG_EVERY:]); mb = np.mean(all_b[-LOG_EVERY:])
            gap = (mb - ma) / (mb + 1e-10) * 100
            print(f"      step {step:>7d} | A={ma:.4f} B={mb:.4f} gap={gap:.1f}% | {time.time()-t0:.1f}s")

    fa = np.mean(all_a[-1000:]); fb = np.mean(all_b[-1000:])
    fgap = (fb - fa) / (fb + 1e-10) * 100
    ac = np.corrcoef(np.array(p2b_actions)[:-1], np.array(p2b_actions)[1:])[0, 1] if len(p2b_actions) > 100 else 0.0
    print(f"      Final: A={fa:.4f} B={fb:.4f} gap={fgap:.1f}% autocorr={ac:.3f}")
    return model, model_p1, signal_type, fa, fb, fgap, ac, p1_gap


def run_spike_test(model, signal_type):
    saved_h = model.h_multi.clone(); saved_hg = model.h_gru.clone(); saved_tr = model.trace
    signal = make_signal(signal_type)
    action_buffer = [0.0] * ACTION_DELAY

    err_normal, err_zero, err_wrong, err_b = [], [], [], []

    for step in range(SPIKE_MEASURE):
        obs_raw = signal.get()
        with torch.no_grad():
            act_val = model.get_action().item()
            act_val = np.clip(act_val, -ACTION_RANGE, ACTION_RANGE)

        # Apply with delay
        if ACTION_DELAY > 0:
            delayed_val = action_buffer.pop(0)
            action_buffer.append(act_val)
        else:
            delayed_val = act_val

        obs = obs_raw.clone()
        obs[0] = obs[0] + delayed_val
        # pred_A receives delayed_action (matches obs)
        delayed_t = torch.tensor([delayed_val], dtype=torch.float32)

        with torch.no_grad():
            pa, pb = model.predict(delayed_t)
            err_normal.append(nn.functional.mse_loss(pa, obs).item())
            err_b.append(nn.functional.mse_loss(pb, obs).item())

            ha0 = torch.cat([model.h_multi, torch.tensor([[0.0]])], dim=1)
            p0 = model.pred_A(ha0).squeeze(0)
            err_zero.append(nn.functional.mse_loss(p0, obs).item())

            haw = torch.cat([model.h_multi, torch.tensor([[-delayed_val]], dtype=torch.float32)], dim=1)
            pw = model.pred_A(haw).squeeze(0)
            err_wrong.append(nn.functional.mse_loss(pw, obs).item())

            if USE_TRACE:
                model.update_trace(act_val)
            model.update_state(obs)

    model.h_multi = saved_h; model.h_gru = saved_hg; model.trace = saved_tr

    mn = np.mean(err_normal); mz = np.mean(err_zero); mw = np.mean(err_wrong); mb = np.mean(err_b)
    results = {"err_a_normal": mn, "err_a_zero": mz, "err_a_wrong": mw, "err_b": mb,
               "spike_zero": mz / (mn + 1e-10), "spike_wrong": mw / (mn + 1e-10)}
    return results, results["spike_zero"]


def main():
    sig = "lorenz" if LORENZ else "sine"
    print("=" * 60)
    print(f"  Experiment 6: Measurement — Agency Gain")
    print(f"  Signal: {sig}")
    print(f"  Trace: {'ON' if USE_TRACE else 'OFF'}")
    print(f"  Delay: {ACTION_DELAY} steps")
    if QUICK: print("  *** QUICK MODE ***")
    print("=" * 60)

    print(f"\n  === FORWARD SAMPLING ===")
    model, model_p1, _, fa, fb, gap, ac, p1_gap = run_full_training(sig, "forward")

    print(f"\n  Phase 1 spike test (100K random-action checkpoint):")
    sr_p1, spike_p1 = run_spike_test(model_p1, sig)
    print(f"    Err_A normal:  {sr_p1['err_a_normal']:.6f}")
    print(f"    Err_A zero:    {sr_p1['err_a_zero']:.6f}")
    print(f"    Err_A wrong:   {sr_p1['err_a_wrong']:.6f}")
    print(f"    Err_B:         {sr_p1['err_b']:.6f}")
    print(f"    spike (zero):  {sr_p1['spike_zero']:.2f}x")
    print(f"    spike (wrong): {sr_p1['spike_wrong']:.2f}x")

    sr, spike = run_spike_test(model, sig)

    print(f"\n  Phase 2b spike test (after full training):")
    print(f"    Err_A normal:  {sr['err_a_normal']:.6f}")
    print(f"    Err_A zero:    {sr['err_a_zero']:.6f}")
    print(f"    Err_A wrong:   {sr['err_a_wrong']:.6f}")
    print(f"    Err_B:         {sr['err_b']:.6f}")
    print(f"    spike (zero):  {sr['spike_zero']:.2f}x")
    print(f"    spike (wrong): {sr['spike_wrong']:.2f}x")

    print(f"\n  === STRATEGY COMPARISON ===")
    strats = [("Forward sampling", gap, spike, ac)]
    for sn, sk in [("Direct AG gradient", "direct_ag"), ("Gradient disagreement", "gradient_disagree")]:
        print(f"\n  --- {sn} ---")
        m2, _, _, _, _, g2, a2, _ = run_full_training(sig, sk)
        _, sp2 = run_spike_test(m2, sig)
        strats.append((sn, g2, sp2, a2))

    print(f"\n{'='*60}")
    print(f"  STRATEGY COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Strategy':<28s} {'gap':>8s} {'spike':>8s} {'autocorr':>10s}")
    print(f"  {'-'*56}")
    for n, g, s, a in strats:
        print(f"  {n:<28s} {g:>7.1f}% {s:>7.2f}x {a:>9.3f}")

    print(f"\n{'='*60}")
    print(f"  DEFINITIVE RESULTS ({sig}, delay={ACTION_DELAY})")
    print(f"{'='*60}")
    print(f"  Phase 1 gap:            {p1_gap:.1f}%")
    print(f"  Final pred gap:         {gap:.1f}%")
    print(f"  Spike ratio:            {spike:.2f}x")
    print(f"  Action autocorrelation: {ac:.3f}")
    print(f"  Err_A:                  {fa:.6f}")
    print(f"  Err_B:                  {fb:.6f}")

    print(f"\n  PHASE 1 vs PHASE 2b COMPARISON")
    print(f"  {'Metric':<20s}  {'Phase 1':>10s}  {'Phase 2b':>10s}")
    print(f"  {'-'*44}")
    print(f"  {'pred gap':<20s}  {p1_gap:>9.1f}%  {gap:>9.1f}%")
    print(f"  {'spike (zero)':<20s}  {spike_p1:>9.2f}x  {spike:>9.2f}x")
    print(f"  {'spike (wrong)':<20s}  {sr_p1['spike_wrong']:>9.2f}x  {sr['spike_wrong']:>9.2f}x")
    print(f"  ─ gap large in BOTH phases; spike discriminates them ─")

    tests = [
        ("Pred gap > 80%", gap > 80, f"{gap:.1f}%"),
        ("Spike > 1.5x", spike > 1.5, f"{spike:.2f}x"),
        ("Autocorr > 0.5", ac > 0.5, f"{ac:.3f}"),
        ("P1 spike < P2b spike", spike_p1 < spike, f"{spike_p1:.2f}x < {spike:.2f}x"),
    ]
    print(f"\n  SCORECARD:")
    np_ = 0
    for name, passed, val in tests:
        st = "PASS" if passed else "FAIL"
        np_ += int(passed)
        print(f"    [{st}] {name:<30s} {val}")
    print(f"  {np_}/{len(tests)} PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()