"""
Experiment 3: The Encoding Gap
==========================================
Paper Section 3.3

GRU input: obs(4d) only — NO trace.
Aux head (3-class: active/trailing/quiet) gradient flows to GRU.
Detached binary probe measures trailing recall.

Expected: trailing recall LOW (~12%) despite aux head.
Without proprioceptive trace the GRU has no channel to retain "I acted".
Exp 4 adds the trace and breaks this gap.

Run:
  python experiments/exp3_encoding_gap.py
  python experiments/exp3_encoding_gap.py --quick
"""

import sys, os, time, math, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn

QUICK = "--quick" in sys.argv
SCALE = 5 if QUICK else 1

TRAIN_STEPS  = 100_000 // SCALE
PHASE2_STEPS =  80_000 // SCALE

LR_P1    = 1e-3
LR_P2    = 1e-4
AUX_LR   = 5e-4
PROBE_LR = 5e-4

GAMMA        = 2.0
ACTION_RANGE = 2.0
AUX_WEIGHT   = 0.1
TRAILING_CLASS_WEIGHT = 5.0
TRAILING_STEPS = 50 // SCALE
SEED      = 42
LOG_EVERY = 5_000 // SCALE

BURST_MEAN = 400 // SCALE
BURST_MIN  = 100 // SCALE
BURST_MAX  = 1000 // SCALE

CLASS_ACTIVE   = 0
CLASS_TRAILING = 1
CLASS_QUIET    = 2


class BurstGate:
    def __init__(self, mean_p=400, min_p=100, max_p=1000, seed=456):
        self.rng = random.Random(seed)
        self.mean_p = mean_p; self.min_p = min_p; self.max_p = max_p
        self.active = False; self.next_switch = 0; self._prev = False

    def step(self, t):
        self._prev = self.active
        if t >= self.next_switch:
            self.active = not self.active
            iv = int(self.rng.expovariate(1.0 / self.mean_p))
            self.next_switch = t + max(self.min_p, min(iv, self.max_p))
        return self.active

    @property
    def just_off(self):
        return self._prev and not self.active


class Model(nn.Module):
    """GRU(4d obs only) — encoding-gap configuration, no trace."""
    def __init__(self, obs_dim=4, hidden_dim=192, n_scales=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru  = nn.GRUCell(obs_dim, hidden_dim)
        self.pred = nn.Linear(hidden_dim, obs_dim)
        alphas = torch.logspace(math.log10(0.02), math.log10(0.80), n_scales)
        per = hidden_dim // n_scales
        av  = torch.cat([a.expand(per) for a in alphas])
        if len(av) < hidden_dim:
            av = torch.cat([av, av[-1:].expand(hidden_dim - len(av))])
        self.register_buffer("alpha", av.unsqueeze(0))
        self.h_multi = torch.zeros(1, hidden_dim)
        self.h_gru   = torch.zeros(1, hidden_dim)

    def step_frozen(self, obs):
        pred  = self.pred(self.h_multi)
        h_new = self.gru(obs.unsqueeze(0), self.h_gru)
        self.h_multi = ((1 - self.alpha) * self.h_multi + self.alpha * h_new).detach()
        self.h_gru   = h_new.detach()
        return pred.squeeze(0)

    def step_live(self, obs):
        pred   = self.pred(self.h_multi)
        h_new  = self.gru(obs.unsqueeze(0), self.h_gru)
        h_live = (1 - self.alpha) * self.h_multi + self.alpha * h_new
        self.h_gru = h_new.detach()
        return pred.squeeze(0), h_live

    def commit(self, h_live):
        self.h_multi = h_live.detach()


class AuxHead(nn.Module):
    def __init__(self, hidden_dim=192):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 3)
    def forward(self, h):
        return self.fc(h)


class BinaryProbe(nn.Module):
    def __init__(self, hidden_dim=192):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, h):
        return self.fc(h).squeeze(-1)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    from core.world import make_signal
    signal = make_signal("sine")

    print("=" * 55)
    print("  Experiment 3: The Encoding Gap")
    print("  GRU input: obs(4d) only — NO trace")
    print("  Aux head: 3-class (active/trailing/quiet)")
    if QUICK:
        print("  *** QUICK MODE ***")
    print("=" * 55)

    model = Model()
    aux   = AuxHead()
    probe = BinaryProbe()
    burst = BurstGate(BURST_MEAN, BURST_MIN, BURST_MAX)

    opt_p1 = torch.optim.Adam(model.parameters(), lr=LR_P1)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_p1, T_max=TRAIN_STEPS, eta_min=1e-5)

    cw       = torch.tensor([1.0, TRAILING_CLASS_WEIGHT, 1.0])
    ce_loss  = nn.CrossEntropyLoss(weight=cw)
    bce_loss = nn.BCEWithLogitsLoss()

    t0 = time.time()
    trailing_remaining = 0
    total_steps    = TRAIN_STEPS + PHASE2_STEPS
    trailing_correct = 0
    trailing_total   = 0
    opt_p2 = opt_probe = None

    print(f"\n  Phase 1: Perception training ({TRAIN_STEPS} steps)")

    for step in range(total_steps):
        is_p1 = step < TRAIN_STEPS
        obs_raw = signal.get()

        if is_p1:
            ba = True
        else:
            ba = burst.step(step)
            if burst.just_off:
                trailing_remaining = TRAILING_STEPS

        # Random actions throughout — no W_action policy in exp3
        action_val = np.random.uniform(-ACTION_RANGE, ACTION_RANGE) if (ba or is_p1) else 0.0

        if is_p1:
            cur_class   = CLASS_ACTIVE
            is_trailing = False
        elif trailing_remaining > 0:
            cur_class   = CLASS_TRAILING
            is_trailing = True
            trailing_remaining -= 1
        elif ba:
            cur_class   = CLASS_ACTIVE
            is_trailing = False
        else:
            cur_class   = CLASS_QUIET
            is_trailing = False

        obs = obs_raw.clone()
        obs[0] = obs[0] + GAMMA * action_val   # action_val is 0 when quiet

        # ── Phase 1 ──────────────────────────────────────────────────────
        if is_p1:
            pred = model.step_frozen(obs)
            loss = nn.functional.mse_loss(pred, obs)
            opt_p1.zero_grad(); loss.backward(); opt_p1.step(); sched.step()
            if step > 0 and step % LOG_EVERY == 0:
                print(f"    step {step:>6d} | MSE={loss.item():.6f} | {time.time()-t0:.1f}s")

        # ── Phase 2 ──────────────────────────────────────────────────────
        else:
            if step == TRAIN_STEPS:
                print(f"\n  Phase 2: Aux head + probe ({PHASE2_STEPS} steps)")
                opt_p2 = torch.optim.Adam([
                    {"params": model.gru.parameters(),  "lr": LR_P2},
                    {"params": model.pred.parameters(), "lr": LR_P2},
                    {"params": aux.parameters(),        "lr": AUX_LR},
                ])
                opt_probe = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)

            pred, h_live = model.step_live(obs)
            loss_pred = nn.functional.mse_loss(pred, obs)
            logits    = aux(h_live)
            loss_aux  = AUX_WEIGHT * ce_loss(
                logits, torch.tensor([cur_class], dtype=torch.long))
            opt_p2.zero_grad(); (loss_pred + loss_aux).backward(); opt_p2.step()
            model.commit(h_live)

            h_det = model.h_multi.detach()
            logit = probe(h_det)
            opt_probe.zero_grad()
            bce_loss(logit, torch.tensor([1.0 if is_trailing else 0.0])).backward()
            opt_probe.step()

            if is_trailing:
                with torch.no_grad():
                    pred_trail = (torch.sigmoid(logit) > 0.5).item()
                trailing_total   += 1
                trailing_correct += int(pred_trail)

            if step > 0 and step % LOG_EVERY == 0:
                tr = trailing_correct / max(1, trailing_total) * 100
                print(f"    step {step:>6d} | MSE={loss_pred.item():.6f}"
                      f" | trailing={tr:.1f}% | {time.time()-t0:.1f}s")

    trailing_recall = trailing_correct / max(1, trailing_total) * 100
    print(f"\n{'='*55}")
    print(f"  RESULTS")
    print(f"{'='*55}")
    print(f"  Trailing recall: {trailing_recall:.1f}%  ({trailing_correct}/{trailing_total})")
    print(f"  Expected:        ~12.3%")

    if trailing_recall < 20:
        print(f"\n  >> Encoding gap confirmed.")
        print(f"     GRU(4d) cannot retain 'I acted' without proprioceptive trace.")
    elif trailing_recall > 50:
        print(f"\n  >> Encoding gap NOT found — unexpected.")
    else:
        print(f"\n  >> Marginal ({trailing_recall:.1f}% between 20-50%).")
    print("=" * 55)


if __name__ == "__main__":
    main()
