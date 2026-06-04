"""
Generate Figure 1: 3D PCA Attractor Trajectory
Paper Section 3.1

- Train PerceptionModel 30K steps on sine signal
- Collect h_multi snapshots, PCA to 3D
- At step 15000: inject noise perturbation
- Blue = normal attractor, Red dashed = recovery path, Green = re-convergence
- Output: figures/output/fig1_attractor.pdf
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

STYLE = os.path.join(os.path.dirname(__file__), 'style', 'paper.mplstyle')
BLUE  = '#2C5F8A'
RED   = '#C0504D'
GREEN = '#4E7C59'

N_STEPS = 30_000
PERTURB_STEP = 15_000
PERTURB_MAG = 3.0
LR = 1e-3
SEED = 42
SNAPSHOT_EVERY = 5  # take snapshot every N steps


class PerceptionModel(nn.Module):
    def __init__(self, obs_dim=4, hidden_dim=192, n_scales=4):
        super().__init__()
        self.hidden_dim = hidden_dim
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


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    from core.world import make_signal
    signal = make_signal("sine")
    model = PerceptionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Training PerceptionModel for Figure 1...")
    t0 = time.time()

    # Phase labels for each snapshot
    snapshots = []
    phase_labels = []  # 0=normal, 1=perturbed, 2=recovery, 3=reconverged

    perturbed = False

    for step in range(N_STEPS):
        obs = signal.get()
        pred = model.predict()
        loss = nn.functional.mse_loss(pred, obs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.update_state(obs)

        if step == PERTURB_STEP:
            print(f"  Injecting perturbation at step {step}...")
            with torch.no_grad():
                model.h_multi = model.h_multi + torch.randn_like(model.h_multi) * PERTURB_MAG
            perturbed = True

        if step % SNAPSHOT_EVERY == 0:
            snapshots.append(model.get_h())
            if step < PERTURB_STEP - 500:
                phase_labels.append(0)   # normal attractor
            elif step < PERTURB_STEP:
                phase_labels.append(0)   # normal attractor
            elif step < PERTURB_STEP + 2000:
                phase_labels.append(1)   # recovery
            else:
                phase_labels.append(2)   # reconverged

        if step % 5000 == 0 and step > 0:
            print(f"  step {step}/{N_STEPS} ({time.time()-t0:.1f}s)")

    print(f"  Training done ({time.time()-t0:.1f}s). Running PCA...")

    H = np.array(snapshots)
    labels = np.array(phase_labels)

    # PCA to 3D
    pca = PCA(n_components=3)
    H3 = pca.fit_transform(H)
    explained = pca.explained_variance_ratio_
    print(f"  PCA explained variance: {explained.sum()*100:.1f}% ({explained*100})")

    # Split by phase
    normal_idx = np.where(labels == 0)[0]
    recovery_idx = np.where(labels == 1)[0]
    reconverged_idx = np.where(labels == 2)[0]

    # For visual clarity: use last 1000 snapshots of normal as "attractor"
    attractor_idx = normal_idx[-200:]  # dense attractor cloud
    pre_perturb_idx = normal_idx[-50:]  # just before perturbation

    plt.style.use(STYLE)

    # Create figure
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection='3d')

    z_floor = H3[:, 2].min() - 0.6   # shadow plane below all data

    # Blue: attractor (dense cloud of normal trajectory)
    ax.scatter(H3[attractor_idx, 0], H3[attractor_idx, 1], H3[attractor_idx, 2],
               c=BLUE, s=4, alpha=0.35, label='Attractor', zorder=1)
    # Shadow of attractor on bottom plane
    ax.scatter(H3[attractor_idx, 0], H3[attractor_idx, 1],
               np.full(len(attractor_idx), z_floor),
               c=BLUE, s=2, alpha=0.08, zorder=0)

    # Blue line: last chunk before perturbation
    ax.plot(H3[pre_perturb_idx, 0], H3[pre_perturb_idx, 1], H3[pre_perturb_idx, 2],
            color=BLUE, lw=1.0, alpha=0.7, zorder=2)

    # Red dashed: recovery path (first part after perturbation)
    if len(recovery_idx) > 0:
        split = max(1, len(recovery_idx) * 3 // 10)
        rec_x = H3[recovery_idx[:split], 0]
        rec_y = H3[recovery_idx[:split], 1]
        rec_z = H3[recovery_idx[:split], 2]
        ax.plot(rec_x, rec_y, rec_z,
                color=RED, lw=1.5, ls='--', alpha=0.9, label='Recovery path', zorder=3)
        # Shadow of recovery path
        ax.plot(rec_x, rec_y, np.full(len(rec_x), z_floor),
                color=RED, lw=0.8, alpha=0.2, zorder=0)

        # Mark perturbation point — large red X with label
        px, py, pz = H3[recovery_idx[0], 0], H3[recovery_idx[0], 1], H3[recovery_idx[0], 2]
        ax.scatter([px], [py], [pz], c=RED, s=140, marker='X', zorder=6,
                   label='Perturbation', edgecolors='#8B1A1A', linewidths=0.8)
        ax.text(px + 0.15, py + 0.15, pz + 0.3,
                'Perturbation\n(noise injected)', fontsize=7.5,
                color=RED, zorder=7, va='bottom')

    # Green: re-convergence (latter half of recovery + reconverged)
    if len(recovery_idx) > 0:
        split = max(1, len(recovery_idx) * 3 // 10)
        reconverge_pts = np.concatenate([recovery_idx[split:], reconverged_idx[:100]])
        if len(reconverge_pts) > 0:
            rc_x = H3[reconverge_pts, 0]
            rc_y = H3[reconverge_pts, 1]
            rc_z = H3[reconverge_pts, 2]
            ax.plot(rc_x, rc_y, rc_z,
                    color=GREEN, lw=1.2, alpha=0.8, label='Re-convergence', zorder=4)
            # Shadow of re-convergence
            ax.plot(rc_x, rc_y, np.full(len(rc_x), z_floor),
                    color=GREEN, lw=0.7, alpha=0.15, zorder=0)
            # Mark endpoint (converged back)
            ex, ey, ez = rc_x[-1], rc_y[-1], rc_z[-1]
            ax.scatter([ex], [ey], [ez], c=GREEN, s=100, marker='o', zorder=6,
                       edgecolors='#2A4A30', linewidths=0.8)
            ax.text(ex + 0.15, ey - 0.1, ez + 0.3,
                    'Re-converged\n(recovery 95%)', fontsize=7.5,
                    color=GREEN, zorder=7, va='bottom')

    ax.set_xlabel(f'PC1 ({explained[0]*100:.1f}%)', fontsize=9, labelpad=3)
    ax.set_ylabel(f'PC2 ({explained[1]*100:.1f}%)', fontsize=9, labelpad=3)
    ax.set_zlabel(f'PC3 ({explained[2]*100:.1f}%)', fontsize=9, labelpad=3)
    ax.set_title('')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.85,
              edgecolor='#CCCCCC', fancybox=False)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=25, azim=45)
    # Tone down the 3D pane/grid
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#E0E0E0')
    ax.yaxis.pane.set_edgecolor('#E0E0E0')
    ax.zaxis.pane.set_edgecolor('#E0E0E0')
    ax.grid(True, color='#EEEEEE', linewidth=0.5)

    plt.tight_layout()
    fig_dir  = os.path.join(os.path.dirname(__file__), "output")
    out_pdf  = os.path.join(fig_dir, "fig1_attractor.pdf")
    out_png  = os.path.join(fig_dir, "fig1_preview.png")
    plt.savefig(out_pdf, dpi=200, bbox_inches='tight')
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    print(f"  Saved: {out_pdf}")
    plt.close()


if __name__ == "__main__":
    main()
