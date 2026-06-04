"""
Generate Figure 6: Action Strategy Comparison
Paper Section 3.6 — Forward sampling vs two gradient-based methods.

Plan B: Grad-disagree (pred gap -1894.5%) is pinned at the left edge of the
degenerate zone with an annotation stating its true value; x-axis is kept at
[-14, 97] so the other two points are well-separated and readable.

Output: figures/output/fig6_strategy_compare.pdf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = os.path.join(os.path.dirname(__file__), 'style', 'paper.mplstyle')
plt.style.use(STYLE)

GREEN = '#4E7C59'   # forward-sampled — success / highlight
RED   = '#C0504D'   # gradient methods — degenerate
GRAY  = '#6B6B6B'

# ── Data (Section 3.6, fixed) ─────────────────────────────────────────────────
#  name               true_gap    spike   autocorr  display_gap
DATA = [
    ('Forward-sampled',  80.7,    17.32,  0.788,   80.7),
    ('Direct-AG',        -2.0,     0.98,  1.000,   -2.0),
    ('Grad-disagree',  -1894.5,    0.02,  0.972,  -10.5),   # pinned left
]

XLIM = (-14, 97)

fig, ax = plt.subplots(figsize=(8.0, 4.8))
ax.set_xlim(*XLIM)
ax.set_yscale('log')
ax.set_ylim(0.007, 60)

# ── Degenerate zone ───────────────────────────────────────────────────────────
ax.axvspan(XLIM[0], 0, color='#EEEEEE', alpha=0.80, zorder=0)
ax.axvline(0, color='#C0C0C0', lw=0.9, ls='--', zorder=1)
ax.text(-7.0, 0.0085, 'degenerate\nzone', fontsize=7.5, color='#BBBBBB',
        ha='center', va='bottom', style='italic', linespacing=1.3, zorder=2)
ax.text(50, 0.0085, 'good zone', fontsize=7.5, color=GREEN,
        ha='center', va='bottom', style='italic', fontweight='bold', zorder=2)

# spike = 1× reference
ax.axhline(1.0, color='#DDDDDD', lw=0.7, ls=':', zorder=1)
ax.text(XLIM[1] - 1, 1.10, 'spike = 1×', fontsize=6.5, color='#BBBBBB',
        ha='right', va='bottom', zorder=2)

# ── Forward-sampled  (green filled star) ──────────────────────────────────────
_, true_gap, spike, ac, disp_x = DATA[0]
ax.scatter([disp_x], [spike], marker='*', s=400, color=GREEN,
           edgecolors=GREEN, zorder=5)
ax.annotate(
    f'Forward-sampled\nspike {spike:.2f}×   autocorr {ac:.3f}',
    xy=(disp_x, spike),
    xytext=(disp_x - 7, spike * 0.33),
    fontsize=7.5, color=GREEN, ha='right', va='top', zorder=6,
    arrowprops=dict(arrowstyle='->', color=GREEN, lw=0.9, shrinkB=4),
    bbox=dict(facecolor='white', edgecolor='none', pad=2.5),
)

# ── Direct-AG  (hollow red circle) ────────────────────────────────────────────
_, true_gap, spike, ac, disp_x = DATA[1]
ax.scatter([disp_x], [spike], marker='o', s=75,
           edgecolors=RED, facecolors='none', lw=1.5, zorder=5)
ax.annotate(
    f'Direct-AG\npred gap {true_gap:.1f}%   spike {spike:.2f}×\n'
    f'autocorr {ac:.3f}  — action locked, no diversity',
    xy=(disp_x, spike),
    xytext=(disp_x + 13, spike * 5.5),
    fontsize=7.2, color=RED, ha='left', va='bottom', zorder=6,
    arrowprops=dict(arrowstyle='->', color=RED, lw=0.9, shrinkB=4),
    bbox=dict(facecolor='white', edgecolor='none', pad=2.5),
)

# ── Grad-disagree  (hollow red circle, Plan B — pinned at left edge) ──────────
_, true_gap, spike, ac, disp_x = DATA[2]
ax.scatter([disp_x], [spike], marker='o', s=75,
           edgecolors=RED, facecolors='none', lw=1.5, zorder=5)
ax.annotate(
    f'Grad-disagree\npred gap {true_gap:.0f}%  ◄ plotted at left edge\n'
    f'spike {spike:.2f}×   autocorr {ac:.3f}\n'
    '(actions saturate at ±max — no diversity)',
    xy=(disp_x, spike),
    xytext=(disp_x + 15, spike * 14),
    fontsize=7.2, color=RED, ha='left', va='bottom', zorder=6,
    arrowprops=dict(arrowstyle='->', color=RED, lw=0.9, shrinkB=4),
    bbox=dict(facecolor='white', edgecolor='none', pad=2.5),
)

# ── Axes / title ──────────────────────────────────────────────────────────────
ax.set_xlabel('Pred Gap (%)')
ax.set_ylabel('Spike (×,  log scale)')
ax.set_title('')

plt.tight_layout()

fig_dir = os.path.join(os.path.dirname(__file__), 'output')
out_pdf = os.path.join(fig_dir, 'fig6_strategy_compare.pdf')
out_png = os.path.join(fig_dir, 'fig6_preview.png')
plt.savefig(out_pdf, dpi=200, bbox_inches='tight')
plt.savefig(out_png, dpi=200, bbox_inches='tight')
print(f'Saved: {out_pdf}')
plt.close()
