"""
Experiment 2: Causal Budding — Implicit Self-World Decomposition
=================================================================
Paper Section 3.2

Question: When the system's action changes one observation channel,
does it learn which channel it affects?

Setup:
  - Same GRU+EMA as Exp 1, but with action loop
  - SINGLE prediction head (no dual heads yet!)
  - action = random noise, obs[0] += gamma * action
  - Causal group: action = f(h) with gamma=2.0
  - Control group: action = AR(1) noise (matched statistics)

Events:
  - Spike test: disconnect action for 2000 steps
  - Per-channel analysis: which channels spike?

Key metrics:
  - ch0 spike ratio (should be high)
  - ch1-3 spike ratio (should be ~1.0)
  - Causal vs Control long-disconnect recovery

Run:
  python -m experiments.exp2_causal
  python -m experiments.exp2_causal --quick
"""

import sys, os, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn

QUICK = "--quick" in sys.argv
SCALE = 5 if QUICK else 1

N_STEPS = 60_000 // SCALE
LR = 1e-3
GAMMA = 2.0
ACTION_RANGE = 2.0
SEED = 42
LOG_EVERY = 5_000 // SCALE
SPIKE_STEPS = 2000 // SCALE
SPIKE_WARMUP = 50 // SCALE


class SingleHeadModel(nn.Module):
    """GRU + EMA + single prediction head + action projection.
    No dual heads — we're testing implicit self/world decomposition."""
    def __init__(self, obs_dim=4, hidden_dim=192, n_scales=4):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(obs_dim, hidden_dim)
        self.pred = nn.Linear(hidden_dim, obs_dim)
        self.W_action = nn.Linear(hidden_dim, 1)

        alphas = torch.logspace(math.log10(0.02), math.log10(0.80), n_scales)
        per = hidden_dim // n_scales
        av = torch.cat([a.expand(per) for a in alphas])
        if len(av) < hidden_dim:
            av = torch.cat([av, av[-1:].expand(hidden_dim - len(av))])
        self.register_buffer("alpha", av.unsqueeze(0))
        self.h_multi = torch.zeros(1, hidden_dim)
        self.h_gru = torch.zeros(1, hidden_dim)

    def predict(self):
        return self.pred(self.h_multi).squeeze(0)

    def update_state(self, obs):
        x = obs.unsqueeze(0)
        h_new = self.gru(x, self.h_gru)
        self.h_multi = ((1 - self.alpha) * self.h_multi + self.alpha * h_new).detach()
        self.h_gru = h_new.detach()

    def get_action(self):
        return self.W_action(self.h_multi.detach()).squeeze(0)

    def reset_state(self):
        self.h_multi = torch.zeros(1, self.hidden_dim)
        self.h_gru = torch.zeros(1, self.hidden_dim)


def train_model(signal_class, use_causal=True, seed=42):
    """Train with either causal action or AR(1) control."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    from core.world import make_signal
    signal = make_signal("sine")
    model = SingleHeadModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    label = "CAUSAL" if use_causal else "CONTROL"

    # For control group: generate AR(1) noise with matched stats
    ar_state = 0.0
    ar_phi = 0.95  # autocorrelation coefficient

    losses = []
    per_ch_losses = {i: [] for i in range(4)}
    t0 = time.time()

    for step in range(N_STEPS):
        obs_raw = signal.get()

        if use_causal:
            with torch.no_grad():
                action_val = model.get_action().item()
                action_val = np.clip(action_val, -ACTION_RANGE, ACTION_RANGE)
        else:
            # AR(1) noise: matched autocorrelation, similar variance
            ar_state = ar_phi * ar_state + np.random.normal(0, 1.0) * (1 - ar_phi**2)**0.5
            action_val = ar_state * ACTION_RANGE * 0.5

        obs = obs_raw.clone()
        obs[0] = obs[0] + GAMMA * action_val

        pred = model.predict()
        loss = nn.functional.mse_loss(pred, obs)
        losses.append(loss.item())

        # Per-channel error
        with torch.no_grad():
            for ch in range(4):
                per_ch_losses[ch].append((pred[ch] - obs[ch]).pow(2).item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.update_state(obs)

        if step > 0 and step % LOG_EVERY == 0:
            recent = np.mean(losses[-LOG_EVERY:])
            print(f"    [{label}] step {step:>6d} | MSE={recent:.6f} | {time.time()-t0:.1f}s")

    return model, signal, losses, per_ch_losses


def spike_test(model, signal_class, use_causal=True, seed=42):
    """Disconnect action, measure per-channel spike."""
    torch.manual_seed(seed + 999)
    np.random.seed(seed + 999)

    from core.world import make_signal
    signal = make_signal("sine")
    model.reset_state()

    label = "CAUSAL" if use_causal else "CONTROL"
    ar_state = 0.0
    ar_phi = 0.95

    # Episode 0: baseline (action connected)
    # Episode 1: action disconnected
    results = {}
    for ep_name, connect_action in [("connected", True), ("disconnected", False)]:
        model.reset_state()
        signal = make_signal("sine")
        per_ch = {i: [] for i in range(4)}

        for step in range(SPIKE_STEPS):
            obs_raw = signal.get()

            if use_causal:
                with torch.no_grad():
                    action_val = model.get_action().item()
                    action_val = np.clip(action_val, -ACTION_RANGE, ACTION_RANGE)
            else:
                ar_state = ar_phi * ar_state + np.random.normal(0, 1.0) * (1 - ar_phi**2)**0.5
                action_val = ar_state * ACTION_RANGE * 0.5

            obs = obs_raw.clone()
            if connect_action:
                obs[0] = obs[0] + GAMMA * action_val

            with torch.no_grad():
                pred = model.predict()
                for ch in range(4):
                    if step >= SPIKE_WARMUP:
                        per_ch[ch].append((pred[ch] - obs[ch]).pow(2).item())
                model.update_state(obs)

        results[ep_name] = {ch: np.mean(per_ch[ch]) for ch in range(4)}

    # Spike ratios
    print(f"\n    [{label}] Spike test:")
    print(f"    {'Channel':<10s} {'Connected':>12s} {'Disconnected':>12s} {'Spike ratio':>12s}")
    print(f"    {'-'*48}")
    for ch in range(4):
        conn = results["connected"][ch]
        disc = results["disconnected"][ch]
        ratio = disc / (conn + 1e-10)
        marker = " <<<" if ratio > 2.0 else ""
        print(f"    ch{ch:<7d} {conn:>12.6f} {disc:>12.6f} {ratio:>11.2f}x{marker}")

    return results


def long_disconnect_test(model, use_causal=True, seed=42, n_steps=2000):
    """Disconnect action for many steps, measure recovery over time."""
    torch.manual_seed(seed + 888)
    np.random.seed(seed + 888)

    from core.world import make_signal
    signal = make_signal("sine")
    model.reset_state()

    n_steps = n_steps // (5 if QUICK else 1)
    ar_state = 0.0
    ar_phi = 0.95

    # Warm up with action connected
    for step in range(500 // (5 if QUICK else 1)):
        obs_raw = signal.get()
        if use_causal:
            with torch.no_grad():
                action_val = model.get_action().item()
        else:
            ar_state = ar_phi * ar_state + np.random.normal(0, 1.0) * (1 - ar_phi**2)**0.5
            action_val = ar_state * ACTION_RANGE * 0.5
        obs = obs_raw.clone()
        obs[0] = obs[0] + GAMMA * action_val
        with torch.no_grad():
            model.predict()
            model.update_state(obs)

    # Record baseline error
    baseline_errors = []
    for step in range(200 // (5 if QUICK else 1)):
        obs_raw = signal.get()
        if use_causal:
            with torch.no_grad():
                action_val = model.get_action().item()
        else:
            ar_state = ar_phi * ar_state + np.random.normal(0, 1.0) * (1 - ar_phi**2)**0.5
            action_val = ar_state * ACTION_RANGE * 0.5
        obs = obs_raw.clone()
        obs[0] = obs[0] + GAMMA * action_val
        with torch.no_grad():
            pred = model.predict()
            baseline_errors.append((pred[0] - obs[0]).pow(2).item())
            model.update_state(obs)
    baseline = np.mean(baseline_errors)

    # Now disconnect action, track ch0 error over time
    ch0_errors = []
    for step in range(n_steps):
        obs_raw = signal.get()
        obs = obs_raw.clone()  # NO action effect
        with torch.no_grad():
            pred = model.predict()
            ch0_errors.append((pred[0] - obs[0]).pow(2).item())
            model.update_state(obs)

    # Recovery: how much of the error reduces from peak to final?
    peak = np.max(ch0_errors[:50]) if len(ch0_errors) > 50 else np.max(ch0_errors)
    final = np.mean(ch0_errors[-100:]) if len(ch0_errors) > 100 else np.mean(ch0_errors)
    recovery = max(0, 1.0 - final / (peak + 1e-10))

    return recovery, baseline, ch0_errors


def main():
    from core.world import make_signal

    print("=" * 55)
    print("  Experiment 2: Causal Budding")
    print("  Does the system learn which channel its action affects?")
    if QUICK: print("  *** QUICK MODE ***")
    print("=" * 55)

    # Train Causal group
    print(f"\n  --- CAUSAL GROUP ---")
    model_c, sig_c, losses_c, pch_c = train_model("sine", use_causal=True, seed=SEED)
    spike_c = spike_test(model_c, "sine", use_causal=True, seed=SEED)

    # Train Control group
    print(f"\n  --- CONTROL GROUP ---")
    model_t, sig_t, losses_t, pch_t = train_model("sine", use_causal=False, seed=SEED)
    spike_t = spike_test(model_t, "sine", use_causal=False, seed=SEED)

    # Long disconnect test
    print(f"\n  --- LONG DISCONNECT TEST ---")
    rec_c, base_c, _ = long_disconnect_test(model_c, use_causal=True, seed=SEED)
    rec_t, base_t, _ = long_disconnect_test(model_t, use_causal=False, seed=SEED)

    print(f"\n    {'':20s} {'Causal':>10s} {'Control':>10s}")
    print(f"    {'-'*42}")
    print(f"    {'Recovery':20s} {rec_c:>9.1%} {rec_t:>9.1%}")

    # Summary
    print(f"\n{'='*55}")
    print(f"  SUMMARY")
    print(f"{'='*55}")
    ch0_spike_c = spike_c["disconnected"][0] / (spike_c["connected"][0] + 1e-10)
    ch0_spike_t = spike_t["disconnected"][0] / (spike_t["connected"][0] + 1e-10)
    print(f"  Causal ch0 spike:  {ch0_spike_c:.1f}x")
    print(f"  Control ch0 spike: {ch0_spike_t:.1f}x")
    print(f"  Causal recovery:   {rec_c:.1%}")
    print(f"  Control recovery:  {rec_t:.1%}")

    if rec_c > rec_t + 0.1:
        print(f"\n  >> Causal group recovers better than Control.")
        print(f"     Self-world decomposition is CAUSAL, not statistical.")
    elif ch0_spike_c > 2.0:
        print(f"\n  >> Channel-specific spike confirmed (ch0={ch0_spike_c:.1f}x).")
        print(f"     But Causal vs Control not cleanly separated.")
    else:
        print(f"\n  >> No clear self-world decomposition.")
    print("=" * 55)


if __name__ == "__main__":
    main()