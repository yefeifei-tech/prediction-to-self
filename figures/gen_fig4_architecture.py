"""
Figure 4: System Architecture
Palette: #2C5F8A blue / #C0504D red / grays only.
Perception side = blue; pred_B (world-blind contrast) = red; trace/world/AG = gray.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── Palette ───────────────────────────────────────────────────────────
B_F  = '#DCE9F5'   # blue fill (light)
B_E  = '#2C5F8A'   # blue edge / main blue
R_F  = '#F5E0E0'   # red fill (light)
R_E  = '#C0504D'   # red edge / brick red
G_F  = '#F0F0F0'   # gray fill
G_E  = '#888888'   # gray edge
TXT  = '#1A1A1A'   # primary text
TXT2 = '#6B6B6B'   # secondary text
CCLR = '#CCCCCC'   # light border color

fig, ax = plt.subplots(figsize=(12, 5.4))
ax.set_xlim(-0.3, 12.2)
ax.set_ylim(-1.0, 5.2)
ax.axis('off')

# ── Clean grid ────────────────────────────────────────────────────────
C1, C2, C3, C4, C5 = 1.0, 3.3, 5.7, 8.0, 10.5   # column x-centers
R4, R3, R2, R1     = 4.05, 2.85, 1.65, 0.25      # row y-centers
BW, BH              = 1.85, 0.64                   # box width / height

HMID = (R3 + R2) / 2   # h_multi vertical center = 2.25

# ── Helper: draw a rounded box ────────────────────────────────────────
def box(cx, cy, text, fc, ec, fs=8.5, bh=None, subtext=None):
    h = bh or BH
    p = FancyBboxPatch((cx - BW/2, cy - h/2), BW, h,
                       boxstyle="round,pad=0.07",
                       facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=3)
    ax.add_patch(p)
    if subtext:
        ax.text(cx, cy + 0.15, text,
                ha='center', va='center', fontsize=fs,
                fontweight='bold', color=TXT, zorder=4)
        ax.text(cx, cy - 0.18, subtext,
                ha='center', va='center', fontsize=7.2, color=TXT2, zorder=4)
    else:
        ax.text(cx, cy, text, ha='center', va='center',
                fontsize=fs, fontweight='bold', color=TXT, zorder=4)

# ── Helper: straight arrow ─────────────────────────────────────────────
def arr(x0, y0, x1, y1, color='#444444', lw=1.2, ls='-'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                linestyle=ls,
                                connectionstyle='arc3,rad=0.0'),
                zorder=5)

# ── Helper: polyline (for routed paths) ───────────────────────────────
def poly(xs, ys, color='#444444', lw=1.2, ls='-'):
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=4,
            solid_capstyle='round', dash_capstyle='round')

# ════════════════════════════════════════════════════════════════════════
#  NODES
# ════════════════════════════════════════════════════════════════════════
box(C1, R3,   'World',             G_F, G_E)
box(C2, R3,   'GRU (192d)',        B_F, B_E)
box(C2, R2,   'Multi-scale EMA',   B_F, B_E, fs=8.0)
box(C3, HMID, r'$h_\mathrm{multi}$', B_F, B_E, fs=10.5)

# Trace box (taller, two lines)
box(C1, R1, 'Action trace  τ', G_F, G_E, bh=0.84,
    subtext=r'$\tau = 0.95\tau + 0.05|a|$')

box(C4, R4,  r'$W_\mathrm{action}$', B_F, B_E, fs=9.5)
box(C4, R3,  r'$\mathrm{pred\_A}(h,a)$', B_F, B_E, fs=8.5)
box(C4, R2,  r'$\mathrm{pred\_B}(h)$',   R_F, R_E, fs=8.5)
box(C5, R3,  'Agency\nGain',          G_F, G_E, bh=0.78)

# ════════════════════════════════════════════════════════════════════════
#  DATA-FLOW ARROWS  (dark gray, solid, uniform style)
# ════════════════════════════════════════════════════════════════════════
# World → GRU
arr(C1 + BW/2, R3, C2 - BW/2, R3)
ax.text((C1+C2)/2, R3 + 0.32, 'obs (4d)',
        ha='center', va='bottom', fontsize=7.5, color=TXT2, zorder=6)

# GRU ↓ EMA
arr(C2, R3 - BH/2, C2, R2 + BH/2)

# EMA → h_multi  (diagonal: EMA is lower-left, h_multi is right-center)
arr(C2 + BW/2, R2, C3 - BW/2, HMID - 0.05)

# h_multi → W_action / pred_A / pred_B  (orthogonal elbow routing)
# Exits spread on h_multi right edge; all turn at shared vertical bus BUS_X
BUS_X = C3 + BW/2 + 0.20   # = 6.825 (shared vertical bus, inside PM outline)
YT    = HMID + 0.12         # = 2.37  (W_action exit — upper)
YM    = HMID                # = 2.25  (pred_A exit  — center)
YB    = HMID - 0.12         # = 2.13  (pred_B exit  — lower)

# h_multi → W_action  (horizontal stub → vertical up → horizontal right)
poly([C3 + BW/2, BUS_X, BUS_X], [YT, YT, R4])
arr(BUS_X, R4, C4 - BW/2, R4)

# h_multi → pred_A  (horizontal stub → vertical up → horizontal right)
poly([C3 + BW/2, BUS_X, BUS_X], [YM, YM, R3])
arr(BUS_X, R3, C4 - BW/2, R3)

# h_multi → pred_B  (horizontal stub → vertical down → horizontal right)
poly([C3 + BW/2, BUS_X, BUS_X], [YB, YB, R2])
arr(BUS_X, R2, C4 - BW/2, R2)

# W_action ↓ pred_A  (action value $a$)
arr(C4, R4 - BH/2, C4, R3 + BH/2)
ax.text(C4 + 0.14, (R4 + R3)/2, '$a$',
        fontsize=9.5, color=B_E, va='center', ha='left', zorder=6)

# pred_A → AG
arr(C4 + BW/2, R3, C5 - BW/2, R3)

# pred_B → AG  (diagonal up to AG left-lower)
arr(C4 + BW/2, R2, C5 - BW/2, R3 - 0.12)

# ════════════════════════════════════════════════════════════════════════
#  TRACE CONNECTIONS (gray, dashed)
# ════════════════════════════════════════════════════════════════════════
# Trace → GRU  (proprioceptive 5th input — enters GRU bottom from below)
# Route: right from trace → vertical up (left of EMA) → horizontal → up into GRU bottom
TR_VX = 2.20                  # vertical x; left of EMA left edge (2.375) — no EMA crossing
TR_HY = 2.38                  # horizontal y; above EMA top (1.97), below GRU bottom (2.53)
TR_BX = C2 - BW/2 + 0.28     # GRU-bottom entry x = 2.655 (inside GRU x-range 2.375–4.225)
poly([C1 + BW/2, TR_VX, TR_VX, TR_BX, TR_BX],
     [R1 + 0.30,  R1 + 0.30, TR_HY, TR_HY, R3 - BH/2 - 0.03],
     color=G_E, lw=1.1, ls='--')
arr(TR_BX, R3 - BH/2 - 0.04, TR_BX, R3 - BH/2, color=G_E, lw=1.1, ls='--')
ax.text(TR_VX - 0.10, 1.10, 'τ (1d)',
        fontsize=7.5, color=TXT2, style='italic', ha='right', va='center')

# W_action → Trace  (feedback: routes UNDER diagram to avoid crossing)
UNDER_Y = -0.68
poly([C4 + BW/2 + 0.1, 11.8, 11.8, C1, C1],
     [R4,               R4,   UNDER_Y, UNDER_Y, R1 - 0.43],
     color=G_E, lw=1.0, ls='--')
arr(C1, R1 - 0.45, C1, R1 - 0.42 - 0.02, color=G_E, lw=1.0, ls='--')
ax.text(6.4, UNDER_Y - 0.18, r'action $a$ → update trace',
        ha='center', fontsize=7.5, color=TXT2, style='italic')

# ════════════════════════════════════════════════════════════════════════
#  CAUSAL LOOP  (brick red, dashed, over the top)
# ════════════════════════════════════════════════════════════════════════
CTOP = 4.76
poly([C4, C4, C1, C1],
     [R4 + BH/2, CTOP, CTOP, R3 + BH/2],
     color=R_E, lw=1.5, ls='--')
arr(C1, R3 + BH/2 + 0.02, C1, R3 + BH/2, color=R_E, lw=1.5, ls='--')
ax.text((C1 + C4)/2, CTOP + 0.15,
        'Causal loop',
        ha='center', va='bottom', fontsize=8.5,
        color=R_E, fontweight='bold')

# ════════════════════════════════════════════════════════════════════════
#  PERCEPTION MODULE BOX  (light border)
# ════════════════════════════════════════════════════════════════════════
PM_X0 = C2 - BW/2 - 0.28
PM_X1 = C3 + BW/2 + 0.28
PM_Y0 = R2 - BH/2 - 0.22
PM_Y1 = R3 + BH/2 + 0.42
ax.plot([PM_X0, PM_X1, PM_X1, PM_X0, PM_X0],
        [PM_Y0, PM_Y0, PM_Y1, PM_Y1, PM_Y0],
        color=CCLR, lw=0.9, zorder=2)
ax.text((PM_X0 + PM_X1)/2, PM_Y1 + 0.1,
        'Perception module',
        ha='center', va='bottom', fontsize=7.5, color=TXT2)

# ════════════════════════════════════════════════════════════════════════
#  DETACH BOUNDARY  (vertical dashed, with label)
# ════════════════════════════════════════════════════════════════════════
# Midpoint between PM right edge (6.905) and pred-heads left edge (7.075) = 6.99
# This keeps the line in the clear gap between Perception module outline and pred boxes
DX = (PM_X1 + (C4 - BW/2)) / 2   # = (6.905 + 7.075)/2 = 6.99
ax.plot([DX, DX], [R2 - BH/2 - 0.08, R4 + BH/2 + 0.12],
        color='#CCCCCC', lw=0.9, ls='--', zorder=2)
ax.text(DX, R4 + BH/2 + 0.28,
        'detach boundary\n(gradient stops here)',
        ha='center', va='bottom', fontsize=6.8, color=TXT2, linespacing=1.35,
        bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                  edgecolor='none', alpha=0.95),
        zorder=6)

# ════════════════════════════════════════════════════════════════════════
#  SAVE
# ════════════════════════════════════════════════════════════════════════
ax.set_title('', pad=5)
fig_dir = os.path.join(os.path.dirname(__file__), 'output')
plt.savefig(os.path.join(fig_dir, 'fig4_architecture.pdf'), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(fig_dir, 'fig4_preview.png'),      dpi=200, bbox_inches='tight')
print('Saved fig4')
plt.close()
