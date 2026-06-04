"""
Generate Figure 3: Agency Gain Training Curves
Paper Section 3.6

- Full 3-phase training on Lorenz signal (P1=100K + P2a=60K + P2b=60K = 220K steps)
- Record pred_A MSE and pred_B MSE per step
- Log-scale y-axis
- Red = pred_A (self-aware), Blue = pred_B (self-blind)
- Phase boundary lines + Phase 2b orange shadow showing gap
- Output: figures/output/fig3_agency_gain.pdf
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

from core.model import AgencyModel
from core.world import make_signal

P1_STEPS  = 100_000
P2A_STEPS =  60_000
P2B_STEPS =  60_000
TOTAL     = P1_STEPS + P2A_STEPS + P2B_STEPS

LR_PERC   = 1e-3
LR_P2     = 1e-4
LR_ACTION = 1e-3
ACTION_RANGE = 2.0
PERTURB_EPS  = 0.3
SEED = 42
LOG_EVERY = 20_000
SMOOTH = 500   # rolling window for plot


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


def rolling_mean(arr, window):
    if len(arr) < window:
        return arr
    result = np.convolve(arr, np.ones(window)/window, mode='valid')
    pad_len = len(arr) - len(result)
    pad = np.full(pad_len, result[0])
    return np.concatenate([pad, result])


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=== Figure 3: Agency Gain Training Curves ===")
    print(f"  Total steps: {TOTAL:,} (P1={P1_STEPS:,} P2a={P2A_STEPS:,} P2b={P2B_STEPS:,})")
    print("  Signal: Lorenz, Strategy: Forward sampling")

    signal = make_signal("lorenz")
    model = AgencyModel(use_trace=True)

    perc_params = (list(model.gru.parameters()) +
                   list(model.pred_A.parameters()) +
                   list(model.pred_B.parameters()))
    opt_perc   = torch.optim.Adam(perc_params, lr=LR_PERC)
    opt_action = torch.optim.Adam(model.W_action.parameters(), lr=LR_ACTION)

    all_la, all_lb = [], []
    t0 = time.time()

    for step in range(TOTAL):
        phase = ("P1"  if step < P1_STEPS else
                 "P2a" if step < P1_STEPS + P2A_STEPS else "P2b")

        obs_raw = signal.get()

        # Action selection
        with torch.no_grad():
            base_action = model.get_action()

        if phase == "P1":
            # Random action
            action = torch.tensor([np.random.uniform(-ACTION_RANGE, ACTION_RANGE)],
                                   dtype=torch.float32)
        else:
            # Forward sampling
            action = select_action_forward(model, base_action)

        action_val = action.item()

        # Apply action to obs
        obs = obs_raw.clone()
        obs[0] = obs[0] + float(np.clip(action_val, -ACTION_RANGE, ACTION_RANGE))

        # Update trace if using
        if model.use_trace:
            model.update_trace(action_val)

        # Perception step
        pred_a, pred_b = model.predict(action)
        model.update_state(obs)

        la = nn.functional.mse_loss(pred_a, obs)
        lb = nn.functional.mse_loss(pred_b, obs)
        loss = la + lb

        # LR schedule: drop in P2a
        if phase in ("P2a", "P2b") and opt_perc.param_groups[0]['lr'] > LR_P2 + 1e-7:
            for g in opt_perc.param_groups:
                g['lr'] = LR_P2

        opt_perc.zero_grad()
        loss.backward()
        opt_perc.step()

        # Action training in P2b only
        if phase == "P2b":
            h_det = model.h_multi.detach()
            a_train = model.W_action(h_det).squeeze(0).clamp(-ACTION_RANGE, ACTION_RANGE)
            ha = torch.cat([h_det, a_train.view(1, 1)], dim=1)
            pa2 = model.pred_A(ha).squeeze(0)
            pb2 = model.pred_B(h_det).squeeze(0)
            err_a2  = nn.functional.mse_loss(pa2, obs)
            disagree = nn.functional.mse_loss(pa2, pb2.detach())
            loss_act = err_a2 - 0.5 * disagree
            opt_action.zero_grad()
            loss_act.backward()
            opt_action.step()

        all_la.append(la.item())
        all_lb.append(lb.item())

        if step % LOG_EVERY == 0 and step > 0:
            la_avg = np.mean(all_la[-LOG_EVERY:])
            lb_avg = np.mean(all_lb[-LOG_EVERY:])
            print(f"  [{phase}] step {step:>7d} | pred_A={la_avg:.5f}  pred_B={lb_avg:.5f} "
                  f"| {time.time()-t0:.1f}s")

    print(f"\n  Training done ({time.time()-t0:.1f}s)")

    # Smooth for plotting
    la_arr = np.array(all_la)
    lb_arr = np.array(all_lb)
    la_sm = rolling_mean(la_arr, SMOOTH)
    lb_sm = rolling_mean(lb_arr, SMOOTH)

    # Final gap in P2b (relative comparison)
    p2b_start = P1_STEPS + P2A_STEPS
    la_p2b = np.mean(la_arr[p2b_start:])
    lb_p2b = np.mean(lb_arr[p2b_start:])
    print(f"  P2b: pred_A MSE < pred_B MSE  (pred_B/pred_A = {lb_p2b/(la_p2b+1e-10):.1f}×)")

    STYLE = os.path.join(os.path.dirname(__file__), 'style', 'paper.mplstyle')
    plt.style.use(STYLE)

    BLUE = '#2C5F8A'
    RED  = '#C0504D'
    GRAY = '#6B6B6B'

    steps = np.arange(TOTAL)
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Phase 2b background tint
    ax.axvspan(p2b_start, TOTAL, alpha=0.07, color=BLUE, zorder=0)

    # Phase boundary lines
    ax.axvline(x=P1_STEPS,  color=GRAY, lw=0.8, ls='--', alpha=0.7)
    ax.axvline(x=p2b_start, color=GRAY, lw=0.8, ls='--', alpha=0.7)

    # Loss curves — pred_A blue, pred_B red
    ax.semilogy(steps, la_sm, color=BLUE, lw=1.5,
                label='pred_A  (action-aware, self)', zorder=3)
    ax.semilogy(steps, lb_sm, color=RED,  lw=1.5,
                label='pred_B  (action-blind, world)', zorder=2)

    # Phase labels at top (blended transform: x=data, y=axes fraction)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    _lbkw = dict(transform=trans, ha='center', va='top', fontsize=8.5, color=GRAY,
                 bbox=dict(boxstyle='round,pad=0.22', facecolor='white',
                           edgecolor='none', alpha=0.9), zorder=6)
    ax.text(P1_STEPS / 2,              0.97, 'Phase 1\n(random actions)', **_lbkw)
    ax.text(P1_STEPS + P2A_STEPS / 2,  0.97, 'Phase 2a\n(consolidation)', **_lbkw)
    ax.text(p2b_start + P2B_STEPS / 2, 0.97, 'Phase 2b\n(agency)',
            transform=trans, ha='center', va='top', fontsize=8.5, color=BLUE,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.22', facecolor='#EEF4FA',
                      edgecolor=BLUE, alpha=0.95), zorder=6)

    # Agency-gain annotation — placed in left part of P2b to not collide with legend
    la_end = np.mean(la_sm[-500:])
    lb_end = np.mean(lb_sm[-500:])
    mid_y  = np.exp((np.log(max(la_end, 1e-12)) + np.log(max(lb_end, 1e-12))) / 2)
    ax.text(p2b_start + P2B_STEPS * 0.22, mid_y,
            'pred_B > pred_A\n(Agency Gain)',
            fontsize=8.5, color=RED, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=RED, alpha=0.85), zorder=6)

    ax.set_xlabel('Training step')
    ax.set_ylabel('MSE (log scale)')
    ax.set_title('')
    # Legend in lower-left to avoid curve overlap in upper right
    ax.legend(loc='lower left', fontsize=8.5)
    ax.set_xlim(0, TOTAL)
    plt.tight_layout()

    fig_dir  = os.path.join(os.path.dirname(__file__), "output")
    out_pdf  = os.path.join(fig_dir, "fig3_agency_gain.pdf")
    out_png  = os.path.join(fig_dir, "fig3_preview.png")
    plt.savefig(out_pdf, dpi=200, bbox_inches='tight')
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    print(f"  Saved: {out_pdf}")
    plt.close()


if __name__ == "__main__":
    main()
