"""
Generate Figure 2: Causal vs Control — Long-Disconnect Recovery
Paper Section 3.2

Uses the exact training and test logic from experiments/exp2_causal.py.
Key result: Causal recovery 74.8% > Control recovery 57.2%

Output: figures/output/fig2_spike.pdf
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Pull in exp2 helper functions with minimal duplication ----
import math, time
import torch
import torch.nn as nn

GAMMA        = 2.0
ACTION_RANGE = 2.0
LR           = 1e-3
SEED         = 42
N_STEPS      = 60_000
DISC_STEPS   = 2000
WARMUP       = 500
BASELINE_WIN = 200
SMOOTH       = 20


class SingleHeadModel(nn.Module):
    def __init__(self, obs_dim=4, hidden_dim=192, n_scales=4):
        super().__init__()
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
        self.h_gru   = torch.zeros(1, hidden_dim)

    def predict(self):
        return self.pred(self.h_multi).squeeze(0)

    def update_state(self, obs):
        x = obs.unsqueeze(0)
        h_new = self.gru(x, self.h_gru)
        self.h_multi = ((1 - self.alpha) * self.h_multi + self.alpha * h_new).detach()
        self.h_gru   = h_new.detach()

    def get_action(self):
        return self.W_action(self.h_multi.detach()).squeeze(0)

    def reset_state(self):
        self.h_multi = torch.zeros(1, self.hidden_dim)
        self.h_gru   = torch.zeros(1, self.hidden_dim)


def train_model(use_causal=True, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    from core.world import make_signal
    signal = make_signal("sine")
    model = SingleHeadModel()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ar_state, ar_phi = 0.0, 0.95
    label = "CAUSAL" if use_causal else "CONTROL"
    t0 = time.time()
    for step in range(N_STEPS):
        obs_raw = signal.get()
        if use_causal:
            with torch.no_grad():
                action_val = float(np.clip(model.get_action().item(),
                                           -ACTION_RANGE, ACTION_RANGE))
        else:
            ar_state = (ar_phi * ar_state
                        + np.random.normal(0, 1.0) * (1 - ar_phi**2)**0.5)
            action_val = ar_state * ACTION_RANGE * 0.5
        obs = obs_raw.clone()
        obs[0] = obs[0] + GAMMA * action_val
        loss = nn.functional.mse_loss(model.predict(), obs)
        opt.zero_grad(); loss.backward(); opt.step()
        model.update_state(obs)
        if step % 10000 == 0 and step > 0:
            print(f"  [{label}] {step}/{N_STEPS}  ({time.time()-t0:.0f}s)")
    print(f"  [{label}] done ({time.time()-t0:.0f}s)")
    return model


def long_disconnect_test(model, use_causal=True, seed=42):
    """
    Exactly matches experiments/exp2_causal.py::long_disconnect_test logic.
    Returns: recovery (float), baseline_mean (float), ch0_errors (array)
    """
    torch.manual_seed(seed + 888); np.random.seed(seed + 888)
    from core.world import make_signal
    signal = make_signal("sine")
    model.reset_state()
    ar_state, ar_phi = 0.0, 0.95

    def _action():
        nonlocal ar_state
        if use_causal:
            return model.get_action().item()
        else:
            ar_state = (ar_phi * ar_state
                        + np.random.normal(0, 1.0) * (1 - ar_phi**2)**0.5)
            return ar_state * ACTION_RANGE * 0.5

    # Warm up with action connected
    for _ in range(WARMUP):
        obs_raw = signal.get()
        action_val = _action()
        obs = obs_raw.clone(); obs[0] = obs[0] + GAMMA * action_val
        with torch.no_grad():
            model.predict(); model.update_state(obs)

    # Record baseline error (action still connected)
    baseline_errors = []
    for _ in range(BASELINE_WIN):
        obs_raw = signal.get()
        action_val = _action()
        obs = obs_raw.clone(); obs[0] = obs[0] + GAMMA * action_val
        with torch.no_grad():
            pred = model.predict()
            baseline_errors.append((pred[0] - obs[0]).pow(2).item())
            model.update_state(obs)
    baseline = np.mean(baseline_errors)

    # Disconnect action
    ch0_errors = []
    for _ in range(DISC_STEPS):
        obs_raw = signal.get()
        obs = obs_raw.clone()            # NO action
        with torch.no_grad():
            pred = model.predict()
            ch0_errors.append((pred[0] - obs[0]).pow(2).item())
            model.update_state(obs)

    peak     = np.max(ch0_errors[:50])
    final    = np.mean(ch0_errors[-100:])
    recovery = max(0.0, 1.0 - final / (peak + 1e-10))
    return recovery, baseline, np.array(ch0_errors)


def rolling_mean(arr, w):
    out = np.convolve(arr, np.ones(w)/w, mode='valid')
    return np.concatenate([np.full(w-1, out[0]), out])


def main():
    print("=== Figure 2: Long-Disconnect Recovery ===")
    print("\nTraining Causal model..."); mc = train_model(True,  SEED)
    print("\nTraining Control model..."); mt = train_model(False, SEED)

    print("\nLong disconnect test...")
    rec_c, bl_c, disc_c = long_disconnect_test(mc, True,  SEED)
    rec_t, bl_t, disc_t = long_disconnect_test(mt, False, SEED)
    print(f"  Causal:  baseline={bl_c:.5f}  recovery={rec_c:.1%}")
    print(f"  Control: baseline={bl_t:.5f}  recovery={rec_t:.1%}")

    STYLE = os.path.join(os.path.dirname(__file__), 'style', 'paper.mplstyle')
    plt.style.use(STYLE)

    RED  = '#C0504D'   # Causal — brick red
    BLUE = '#2C5F8A'   # Control — main blue
    GRAY = '#6B6B6B'

    sm_c = rolling_mean(disc_c, SMOOTH)
    sm_t = rolling_mean(disc_t, SMOOTH)
    steps = np.arange(DISC_STEPS)

    fig, ax = plt.subplots(figsize=(7.5, 4))

    ax.plot(steps, sm_c, color=RED,  lw=1.5,
            label=f'Causal  (recovery {rec_c:.0%})', zorder=3)
    ax.plot(steps, sm_t, color=BLUE, lw=1.5,
            label=f'Control (recovery {rec_t:.0%})', zorder=2)

    # Baseline dotted reference lines
    ax.axhline(bl_c, color=RED,  lw=0.8, ls=':', alpha=0.55)
    ax.axhline(bl_t, color=BLUE, lw=0.8, ls=':', alpha=0.55)
    ax.text(DISC_STEPS * 0.96, bl_c * 1.04,
            'Causal baseline',  fontsize=7.5, color=RED,  ha='right', va='bottom')
    ax.text(DISC_STEPS * 0.96, bl_t * 1.04,
            'Control baseline', fontsize=7.5, color=BLUE, ha='right', va='bottom')

    ax.set_xlabel('Steps after action disconnected')
    ax.set_ylabel('ch0 Prediction Error (MSE)')
    ax.set_title('')
    ax.legend(loc='upper right')
    ax.set_xlim(0, DISC_STEPS)
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    fig_dir = os.path.join(os.path.dirname(__file__), "output")
    plt.savefig(os.path.join(fig_dir, "fig2_spike.pdf"), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(fig_dir, "fig2_preview.png"), dpi=200, bbox_inches='tight')
    print(f"\nSaved fig2")
    plt.close()


if __name__ == "__main__":
    main()
