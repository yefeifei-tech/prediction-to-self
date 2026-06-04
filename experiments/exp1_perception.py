"""
Experiment 1: Perception — Stable Attractors
=============================================
Paper Section 3.1

Question: Does a continuously-predicting GRU form stable internal structure?

Setup:
  - 4-channel sinusoidal signal (no action, no dual heads)
  - GRU(192d) + multi-scale EMA, single prediction head
  - Train for 50K steps, then run 7 scorecard tests

Events:
  - Step 30K: perturb h_multi with noise → does it recover?
  - Step 40K: change signal frequencies → does h_multi respond?

Scorecard (7 tests):
  1. Low dimensionality: eff. dim < 30% of hidden
  2. Power-law inertia: multi-scale temporal structure
  3. Perturbation recovery: > 50%
  4. Residual white noise: prediction fully extracts structure
  5. Novelty response: h_multi displacement > 1.0
  6. Spectral separation: 4 EMA groups have different peak freqs
  7. Error stationarity: stable prediction error

Run from project root:
  python -m experiments.exp1_perception
  python -m experiments.exp1_perception --quick
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

QUICK = "--quick" in sys.argv
SCALE = 5 if QUICK else 1

N_STEPS = 50_000 // SCALE
PERTURB_STEP = 30_000 // SCALE
PERTURB_MAG = 2.0
NOVELTY_STEP = 40_000 // SCALE
LR = 1e-3
SEED = 42
LOG_EVERY = 5_000 // SCALE
SNAPSHOT_EVERY = 10


class PerceptionModel(nn.Module):
    """Simplified model: GRU + EMA + single pred head. No action, no dual heads."""
    def __init__(self, obs_dim=4, hidden_dim=192, n_scales=4):
        super().__init__()
        import math
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.n_scales = n_scales

        self.gru = nn.GRUCell(obs_dim, hidden_dim)
        self.pred = nn.Linear(hidden_dim, obs_dim)

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

    def get_h(self):
        return self.h_multi.detach().squeeze().numpy().copy()


def run_scorecard(h_snapshots, losses, n_scales=4):
    """Run 7 scorecard tests."""
    H = np.array(h_snapshots)
    results = []

    # 1. Dimensionality
    centered = H - H.mean(axis=0)
    _, S, _ = np.linalg.svd(centered, full_matrices=False)
    var_exp = (S**2) / (S**2).sum()
    cumulative = np.cumsum(var_exp)
    eff_dim = np.searchsorted(cumulative, 0.95) + 1
    total = H.shape[1]
    passed = eff_dim / total < 0.30
    results.append(("Dim < 30%", passed, f"{eff_dim}/{total}"))

    # 2. Power-law
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    pc1 = centered @ Vt[0]
    pc1 = pc1 / (pc1.std() + 1e-8)
    max_lag = min(500, len(pc1) // 4)
    lags = np.arange(1, max_lag + 1)
    autocorrs = np.array([np.corrcoef(pc1[:-l], pc1[l:])[0, 1] for l in lags])
    valid = autocorrs > 0.01
    if valid.sum() > 10:
        log_l = np.log(lags[valid])
        log_ac = np.log(autocorrs[valid])
        _, _, r_pl, _, _ = stats.linregress(log_l, log_ac)
        _, _, r_exp, _, _ = stats.linregress(lags[valid], log_ac)
        passed = r_pl**2 > r_exp**2
        dtype = "power-law" if passed else "exponential"
    else:
        passed = False
        dtype = "insufficient data"
    results.append(("Power-law", passed, dtype))

    # 3. Recovery (computed from perturbation data, passed in separately)
    # Will be filled by caller
    results.append(None)  # placeholder

    # 4. Residual white noise
    E = np.array(losses)
    stable = E[len(E)//4 : PERTURB_STEP]
    if len(stable) > 500:
        res = stable - stable.mean()
        res = res / (res.std() + 1e-8)
        n = len(res)
        band = 1.96 / np.sqrt(n)
        max_lag_r = min(150, n // 3)
        acs = [np.corrcoef(res[:-l], res[l:])[0, 1] for l in range(1, max_lag_r + 1)]
        outside = sum(1 for a in acs if abs(a) > band)
        frac = outside / len(acs)
        passed = frac < 0.10
        results.append(("Residual white", passed, f"{outside}/{len(acs)} ({frac:.1%})"))
    else:
        results.append(("Residual white", False, "insufficient data"))

    # 5. Novelty (placeholder, filled by caller)
    results.append(None)

    # 6. Spectral separation
    hidden = H.shape[1]
    per = hidden // n_scales
    peaks = []
    for i in range(n_scales):
        group = H[:, i*per:(i+1)*per]
        gm = group.mean(axis=1)
        gm = gm - gm.mean()
        if gm.std() < 1e-10:
            peaks.append(0)
            continue
        fft_v = np.abs(np.fft.rfft(gm))**2
        freqs = np.fft.rfftfreq(len(gm), d=1.0)
        fft_v[0] = 0
        peak_idx = np.argmax(fft_v[1:]) + 1
        peaks.append(freqs[peak_idx])
    # Check monotonic increase (slow→fast)
    passed = all(peaks[i] <= peaks[i+1] for i in range(len(peaks)-1)) and all(p > 0 for p in peaks)
    results.append(("Spectral sep.", passed, f"peaks={[f'{p:.4f}' for p in peaks]}"))

    # 7. Stationarity
    half = len(E) // 2
    E2 = E[half:PERTURB_STEP] if PERTURB_STEP > half else E[half:]
    if len(E2) > 2000:
        window = 1000
        variances = []
        for i in range(0, len(E2) - window, window // 2):
            variances.append(np.var(E2[i:i+window]))
        variances = np.array(variances)
        cv = np.std(variances) / (np.mean(variances) + 1e-10)
        passed = cv < 2.0
        results.append(("Stationarity", passed, f"CV={cv:.3f}"))
    else:
        results.append(("Stationarity", False, "insufficient data"))

    return results


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    from core.world import make_signal
    signal = make_signal("sine")
    model = PerceptionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("=" * 55)
    print("  Experiment 1: Perception — Stable Attractors")
    print(f"  {N_STEPS} steps, hidden=192, 4 EMA scales")
    if QUICK: print("  *** QUICK MODE ***")
    print("=" * 55)

    losses = []
    h_snapshots = []
    h_pre_perturb = []
    h_post_perturb = []
    h_pre_novelty = []
    h_post_novelty = []
    novelty_done = False
    t0 = time.time()

    for step in range(N_STEPS):
        obs = signal.get()

        # Predict from old state
        pred = model.predict()
        loss = nn.functional.mse_loss(pred, obs)
        losses.append(loss.item())

        # Backward + update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update state with current obs
        model.update_state(obs)

        # Collect snapshots
        if step % SNAPSHOT_EVERY == 0:
            h = model.get_h()
            h_snapshots.append(h)
            if PERTURB_STEP - 2000 // SCALE < step < PERTURB_STEP:
                h_pre_perturb.append(h)
            if PERTURB_STEP < step < NOVELTY_STEP:
                h_post_perturb.append(h)
            if NOVELTY_STEP - 2000 // SCALE < step < NOVELTY_STEP:
                h_pre_novelty.append(h)
            if step > NOVELTY_STEP:
                h_post_novelty.append(h)

        # Event 1: Perturbation
        if step == PERTURB_STEP:
            print(f"\n  >>> PERTURBATION at step {step} (magnitude={PERTURB_MAG})")
            with torch.no_grad():
                model.h_multi = model.h_multi + torch.randn_like(model.h_multi) * PERTURB_MAG

        # Event 2: Novelty
        if step == NOVELTY_STEP and not novelty_done:
            print(f"  >>> NOVELTY at step {step} (frequency change)")
            # Change frequencies of the sine signal
            for ch in signal.CHANNELS:
                ch["freqs"] = [f * 0.7 for f in ch["freqs"]]
            novelty_done = True

        # Log
        if step > 0 and step % LOG_EVERY == 0:
            recent = np.mean(losses[-LOG_EVERY:])
            print(f"    step {step:>6d} | MSE={recent:.6f} | {time.time()-t0:.1f}s")

    print(f"\n  Training done. {time.time()-t0:.1f}s")
    print(f"  Final MSE: {np.mean(losses[-1000:]):.6f}")

    # Scorecard
    sc = run_scorecard(h_snapshots, losses)

    # Fill in recovery (test 3)
    if len(h_pre_perturb) > 5 and len(h_post_perturb) > 5:
        pre = np.array(h_pre_perturb)
        post = np.array(h_post_perturb)
        mean_pre = pre.mean(axis=0)
        dist_post = np.linalg.norm(post - mean_pre, axis=1)
        max_dist = dist_post[0] if len(dist_post) > 0 else 1.0
        settled = np.mean(dist_post[-50:]) if len(dist_post) > 50 else dist_post[-1]
        recovery = max(0, 1.0 - settled / (max_dist + 1e-10))
        sc[2] = ("Recovery > 50%", recovery > 0.50, f"{recovery:.1%}")
    else:
        sc[2] = ("Recovery > 50%", False, "insufficient data")

    # Fill in novelty (test 5)
    if len(h_pre_novelty) > 5 and len(h_post_novelty) > 5:
        pre_n = np.array(h_pre_novelty)
        post_n = np.array(h_post_novelty)
        mean_pre_n = pre_n.mean(axis=0)
        displacements = np.linalg.norm(post_n - mean_pre_n, axis=1)
        peak = np.max(displacements)
        sc[4] = ("Novelty > 1.0", peak > 1.0, f"{peak:.3f}")
    else:
        sc[4] = ("Novelty > 1.0", False, "insufficient data")

    # Print scorecard
    n_pass = sum(1 for _, p, _ in sc if p)
    print(f"\n    {'='*50}")
    print(f"    SCORECARD: {n_pass}/{len(sc)} PASS")
    print(f"    {'='*50}")
    for name, passed, val in sc:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"    {status} {name:<20s} {val}")
    print(f"    {'='*50}")


if __name__ == "__main__":
    main()