"""
Generate Figure 5: Developmental Chain Diagram (matplotlib fallback)
Produces figures/output/fig5_developmental_chain.pdf
When LaTeX/TikZ is available, compile fig5_developmental_chain.tex instead.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Palette (blue system / red = bottleneck / green = breakthrough) ──
_BF = '#DCE9F5'   # blue fill (all stages)
_BE = '#2C5F8A'   # blue edge (default)
_RE = '#C0504D'   # brick-red edge  (Encoding Gap — bottleneck)
_GE = '#4E7C59'   # green edge      (Proprioceptive Breakthrough)

STAGES = [
    {
        "title": "Exp 1 — Perception: Stable Attractors",
        "detail": "6/7 PASS  |  recovery 95.0%  |  eff. dim 5/192",
        "fc": _BF, "ec": _BE,
        "q": "Does structured state imply causal influence?",
    },
    {
        "title": "Exp 2 — Causal Budding",
        "detail": "Causal recovery 74.8%  vs.  Control 57.2%",
        "fc": _BF, "ec": _BE,
        "q": "Can the system remember what it did?",
    },
    {
        "title": "Exp 3 — Encoding Gap",
        "detail": "Trailing recall 12.3%  (no proprioception)",
        "fc": _BF, "ec": _RE,   # red border = bottleneck
        "q": "Add proprioceptive trace — does recall improve?",
    },
    {
        "title": "Exp 4 — Proprioceptive Breakthrough",
        "detail": "Trailing recall 56.5%  (4.6x improvement)",
        "fc": _BF, "ec": _GE,   # green border = breakthrough
        "q": "Can perception and action learn simultaneously?",
    },
    {
        "title": "Exp 5 — Asynchronous Awakening",
        "detail": "Async: spike 5.58x, trailing 66.3%  |  Simul FAST: 4.76x, 60.5%",
        "fc": _BF, "ec": _BE,
        "q": "Can we quantify agency objectively?",
    },
    {
        "title": "Exp 6 — Measurement: Agency Gain",
        "detail": "Agency gain positive on both Sine and Lorenz;\nforward sampling > gradient methods",
        "fc": _BF, "ec": _BE,
        "q": None,
        "tall": True,
    },
]

STYLE = os.path.join(os.path.dirname(__file__), 'style', 'paper.mplstyle')
plt.style.use(STYLE)

BOX_W    = 7.5
BOX_H    = 0.74    # normal box height
BOX_H_T  = 0.96   # tall box height (for 2-line detail)
GAP      = 0.58    # increased: must fit one line of 7.2pt italic text + margins
START_Y  = 9.0
COND_GAP = 0.68   # extra gap before condition bar

BOX_W_TOTAL = BOX_W + 1.0
cx = BOX_W_TOTAL / 2

# ── Figure sizing: compute total height exactly ─────────────────
total_content_h = 0.0
for i, s in enumerate(STAGES):
    h = BOX_H_T if s.get("tall") else BOX_H
    total_content_h += h
    if i < len(STAGES) - 1:
        total_content_h += GAP
COND_H = 0.52
total_content_h += COND_GAP + COND_H

# y-coordinates of each box center (top-down)
box_cy = []
y_cursor = START_Y
for i, s in enumerate(STAGES):
    h = BOX_H_T if s.get("tall") else BOX_H
    box_cy.append(y_cursor)
    if i < len(STAGES) - 1:
        y_cursor -= (h + GAP)
    else:
        y_cursor -= (h + COND_GAP)

cond_y = y_cursor + COND_H / 2   # center of condition bar

y_top    = START_Y + BOX_H / 2 + 0.4
y_bottom = cond_y - COND_H / 2 - 0.25

fig_h = (y_top - y_bottom) * 0.78  # scale data units → inches

# Right side is extended to accommodate negative-result branch boxes
BRANCH_X  = cx + BOX_W / 2 + 1.8   # center of branch boxes
BRANCH_W  = 2.7
BRANCH_H  = 0.90
XLIM_MAX  = BRANCH_X + BRANCH_W / 2 + 0.35

fig, ax = plt.subplots(figsize=(13, max(fig_h, 6.5)))
ax.set_xlim(0, XLIM_MAX)
ax.set_ylim(y_bottom, y_top)
ax.axis('off')

# ── Draw stage boxes ────────────────────────────────────────────
for i, s in enumerate(STAGES):
    h  = BOX_H_T if s.get("tall") else BOX_H
    cy = box_cy[i]

    rect = mpatches.FancyBboxPatch(
        (cx - BOX_W / 2, cy - h / 2), BOX_W, h,
        boxstyle="round,pad=0.06",
        facecolor=s["fc"], edgecolor=s["ec"],
        linewidth=1.8, zorder=3)
    ax.add_patch(rect)

    # Title (upper)
    title_y_offset = 0.16 if s.get("tall") else 0.13
    ax.text(cx, cy + title_y_offset, s["title"],
            ha='center', va='center', fontsize=9.5,
            fontweight='bold', color='#1a1a1a', zorder=4)

    # Detail (lower) — handles \n in text
    detail_y_offset = -0.18 if not s.get("tall") else -0.16
    ax.text(cx, cy + detail_y_offset, s["detail"],
            ha='center', va='center', fontsize=8.0,
            color='#333333', zorder=4, linespacing=1.4)

    # Arrow + question label between boxes
    if i < len(STAGES) - 1:
        next_h = BOX_H_T if STAGES[i+1].get("tall") else BOX_H
        arrow_top = cy - h / 2
        arrow_bot = box_cy[i+1] + next_h / 2
        mid_y = (arrow_top + arrow_bot) / 2
        ax.annotate('', xy=(cx, arrow_bot + 0.02),
                    xytext=(cx, arrow_top - 0.02),
                    arrowprops=dict(arrowstyle='->', color='#cccccc', lw=1.6),
                    zorder=5)
        q = s.get("q")
        if q:
            ax.text(cx, mid_y, f'  {q}  ',
                    ha='center', va='center', fontsize=7.5,
                    style='italic', color='#6B6B6B', zorder=6,
                    bbox=dict(facecolor='white', edgecolor='none', pad=2))

# ── Condition bar ────────────────────────────────────────────────
bar_rect = mpatches.FancyBboxPatch(
    (cx - BOX_W / 2, cond_y - COND_H / 2), BOX_W, COND_H,
    boxstyle="round,pad=0.06",
    facecolor='#EEF4FA', edgecolor='#2C5F8A',
    linewidth=1.2, zorder=3)
ax.add_patch(bar_rect)
ax.text(cx, cond_y,
        "Sufficient conditions:\n"
        "persistent state  →  causal loop  →  proprioception  →  async awakening",
        ha='center', va='center', fontsize=7.6, color='#2C5F8A',
        fontweight='bold', zorder=4, linespacing=1.5)

# ── Negative-result branch boxes ─────────────────────────────────
# Stages with falsified hypotheses branch off to the right as "dead ends"
DEAD_ENDS = {
    2: {                          # Exp 3 — Encoding Gap
        "title": "X  6 Dead Ends",
        "lines": ("- Stronger action: +1.9pp only\n"
                  "- Complex probe: collapses (1.4%)\n"
                  "- EMA readout: ceiling 34%"),
    },
    4: {                          # Exp 5 — Async Awakening
        "title": "X  Simultaneous: Fragile",
        "lines": ("- Medium LR: spike 3.98x, trailing 21.5%\n"
                  "- LR-sensitive; strict isolation needed"),
    },
}
for si, de in DEAD_ENDS.items():
    h_main = BOX_H_T if STAGES[si].get("tall") else BOX_H
    cy     = box_cy[si]
    # Dashed red arrow from main-box right edge → branch box left edge
    ax.annotate('',
                xy=(BRANCH_X - BRANCH_W / 2, cy),
                xytext=(cx + BOX_W / 2, cy),
                arrowprops=dict(arrowstyle='->', color='#cc2222',
                               lw=1.4, linestyle='dashed'), zorder=5)
    # Branch box
    br = mpatches.FancyBboxPatch(
        (BRANCH_X - BRANCH_W / 2, cy - BRANCH_H / 2), BRANCH_W, BRANCH_H,
        boxstyle="round,pad=0.06",
        facecolor='#fff0f0', edgecolor='#cc2222',
        linewidth=1.3, linestyle='--', zorder=3)
    ax.add_patch(br)
    # Title (red, bold)
    ax.text(BRANCH_X, cy + 0.18, de["title"],
            ha='center', va='center', fontsize=7.8,
            fontweight='bold', color='#cc2222', zorder=4)
    # Detail lines
    ax.text(BRANCH_X, cy - 0.14, de["lines"],
            ha='center', va='center', fontsize=6.6,
            color='#884444', zorder=4, linespacing=1.35)

ax.set_title('', pad=5)
fig_dir = os.path.join(os.path.dirname(__file__), 'output')
out_pdf = os.path.join(fig_dir, 'fig5_developmental_chain.pdf')
out_png = os.path.join(fig_dir, 'fig5_preview.png')
plt.savefig(out_pdf, dpi=200, bbox_inches='tight')
plt.savefig(out_png, dpi=200, bbox_inches='tight')
print(f'Saved: {out_pdf}')
plt.close()
