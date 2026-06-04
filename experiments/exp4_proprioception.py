"""
Experiment 4: Proprioceptive Breakthrough
==========================================
Paper Section 3.4

IDENTICAL to Exp 3 except: GRU input is 5d (obs + trace) instead of 4d.
Same aux head, same burst gate, same probe, same everything.
Only difference: one extra input dimension.

Expected: trailing recall jumps from ~9-15% (Exp 3) to ~60%+ (Exp 4).

Run:
  python experiments/exp4_proprioception.py
  python experiments/exp4_proprioception.py --quick
"""

import sys, os, time, math, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn

QUICK = "--quick" in sys.argv
SCALE = 5 if QUICK else 1

TRAIN_STEPS = 100_000 // SCALE
PHASE2_STEPS = 80_000 // SCALE
LR_P1 = 1e-3
LR_P2 = 1e-4
AUX_LR = 5e-4
PROBE_LR = 5e-4
GAMMA = 2.0
ACTION_RANGE = 2.0
TRACE_BETA = 0.95
AUX_WEIGHT = 0.1
TRAILING_CLASS_WEIGHT = 5.0
TRAILING_STEPS = 50 // SCALE
SEED = 42
LOG_EVERY = 5_000 // SCALE

BURST_MEAN = 400 // SCALE
BURST_MIN = 100 // SCALE
BURST_MAX = 1000 // SCALE

CLASS_ACTIVE = 0
CLASS_TRAILING = 1
CLASS_QUIET = 2


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


class ProprioceptiveModel(nn.Module):
    """GRU(5d) = obs(4) + trace(1). Only difference from Exp 3's Model."""
    def __init__(self, obs_dim=4, hidden_dim=192, n_scales=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(obs_dim + 1, hidden_dim)  # +1 for trace
        self.pred = nn.Linear(hidden_dim, obs_dim)
        alphas = torch.logspace(math.log10(0.02), math.log10(0.80), n_scales)
        per = hidden_dim // n_scales
        av = torch.cat([a.expand(per) for a in alphas])
        if len(av) < hidden_dim:
            av = torch.cat([av, av[-1:].expand(hidden_dim - len(av))])
        self.register_buffer("alpha", av.unsqueeze(0))
        self.h_multi = torch.zeros(1, hidden_dim)
        self.h_gru = torch.zeros(1, hidden_dim)
        self.trace = 0.0

    def update_trace(self, action_val):
        self.trace = TRACE_BETA * self.trace + (1 - TRACE_BETA) * abs(action_val)

    def _gru_input(self, obs):
        trace_t = torch.tensor([[self.trace]], dtype=torch.float32)
        return torch.cat([obs.unsqueeze(0), trace_t], dim=1)

    def step_frozen(self, obs):
        pred = self.pred(self.h_multi)
        x = self._gru_input(obs)
        h_new = self.gru(x, self.h_gru)
        self.h_multi = ((1 - self.alpha) * self.h_multi + self.alpha * h_new).detach()
        self.h_gru = h_new.detach()
        return pred.squeeze(0)

    def step_live(self, obs):
        pred = self.pred(self.h_multi)
        x = self._gru_input(obs)
        h_new = self.gru(x, self.h_gru)
        h_multi_live = (1 - self.alpha) * self.h_multi + self.alpha * h_new
        self.h_gru = h_new.detach()
        return pred.squeeze(0), h_multi_live

    def commit(self, h_live):
        self.h_multi = h_live.detach()

    def get_h(self):
        return self.h_multi.detach().squeeze().clone()


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
    print("  Experiment 4: Proprioceptive Breakthrough")
    print("  GRU input: obs(4d) + trace(1d) = 5d")
    print("  Same aux head as Exp 3, only trace added")
    if QUICK: print("  *** QUICK MODE ***")
    print("=" * 55)

    model = ProprioceptiveModel()
    aux = AuxHead()
    probe = BinaryProbe()
    burst = BurstGate(BURST_MEAN, BURST_MIN, BURST_MAX)

    opt_p1 = torch.optim.Adam(model.parameters(), lr=LR_P1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p1, T_max=TRAIN_STEPS, eta_min=1e-5)

    cw = torch.tensor([1.0, TRAILING_CLASS_WEIGHT, 1.0])
    ce_loss = nn.CrossEntropyLoss(weight=cw)
    bce_loss = nn.BCEWithLogitsLoss()

    t0 = time.time()
    trailing_remaining = 0
    total_steps = TRAIN_STEPS + PHASE2_STEPS
    trailing_correct = 0
    trailing_total = 0

    print(f"\n  Phase 1: Perception training ({TRAIN_STEPS} steps)")

    for step in range(total_steps):
        is_p1 = step < TRAIN_STEPS
        obs_raw = signal.get()

        if not is_p1:
            ba = burst.step(step)
            if burst.just_off:
                trailing_remaining = TRAILING_STEPS
        else:
            ba = True

        if ba and not is_p1:
            action_val = np.random.uniform(-ACTION_RANGE, ACTION_RANGE)
        elif is_p1:
            action_val = np.random.uniform(-ACTION_RANGE, ACTION_RANGE)
        else:
            action_val = 0.0

        if not is_p1:
            if trailing_remaining > 0:
                cur_class = CLASS_TRAILING
                is_trailing = True
                trailing_remaining -= 1
            elif ba:
                cur_class = CLASS_ACTIVE
                is_trailing = False
            else:
                cur_class = CLASS_QUIET
                is_trailing = False
        else:
            is_trailing = False

        # Update trace BEFORE state update
        model.update_trace(action_val if (ba or is_p1) else 0.0)

        obs = obs_raw.clone()
        gamma_eff = GAMMA if (ba or is_p1) else 0.0
        obs[0] = obs[0] + gamma_eff * action_val

        if is_p1:
            pred = model.step_frozen(obs)
            loss = nn.functional.mse_loss(pred, obs)
            opt_p1.zero_grad()
            loss.backward()
            opt_p1.step()
            sched.step()

            if step > 0 and step % LOG_EVERY == 0:
                print(f"    step {step:>6d} | MSE={loss.item():.6f} | trace={model.trace:.3f} | {time.time()-t0:.1f}s")
        else:
            if step == TRAIN_STEPS:
                print(f"\n  Phase 2: Aux head + probe ({PHASE2_STEPS} steps)")
                opt_p2 = torch.optim.Adam([
                    {"params": model.gru.parameters(), "lr": LR_P2},
                    {"params": model.pred.parameters(), "lr": LR_P2},
                    {"params": aux.parameters(), "lr": AUX_LR},
                ])
                opt_probe = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)

            pred, h_live = model.step_live(obs)
            loss_pred = nn.functional.mse_loss(pred, obs)

            logits = aux(h_live)
            target = torch.tensor([cur_class], dtype=torch.long)
            loss_aux = AUX_WEIGHT * ce_loss(logits, target)

            total_loss = loss_pred + loss_aux
            opt_p2.zero_grad()
            total_loss.backward()
            opt_p2.step()
            model.commit(h_live)

            h_det = model.get_h()
            logit = probe(h_det.unsqueeze(0))
            probe_target = torch.tensor([1.0 if is_trailing else 0.0])
            probe_loss = bce_loss(logit, probe_target)
            opt_probe.zero_grad()
            probe_loss.backward()
            opt_probe.step()

            if is_trailing:
                with torch.no_grad():
                    pred_trailing = (torch.sigmoid(logit) > 0.5).float().item()
                trailing_total += 1
                if pred_trailing == 1.0:
                    trailing_correct += 1

            if step > 0 and step % LOG_EVERY == 0:
                tr = trailing_correct / max(1, trailing_total) * 100
                print(f"    step {step:>6d} | MSE={loss_pred.item():.6f} | trailing={tr:.1f}% | trace={model.trace:.3f} | {time.time()-t0:.1f}s")

    trailing_recall = trailing_correct / max(1, trailing_total) * 100
    print(f"\n{'='*55}")
    print(f"  RESULTS")
    print(f"{'='*55}")
    print(f"  Trailing recall: {trailing_recall:.1f}% ({trailing_correct}/{trailing_total})")

    print(f"\n  Comparison:")
    print(f"    Exp 3 (no trace, 4d):   trailing recall LOW [encoding gap]")
    print(f"    Exp 4 (with trace, 5d): trailing recall {trailing_recall:.1f}%")

    if trailing_recall > 50:
        print(f"\n  >> PROPRIOCEPTIVE BREAKTHROUGH!")
        print(f"     One extra input dimension breaks the encoding gap.")
    elif trailing_recall > 20:
        print(f"\n  >> Partial improvement. Trace helps significantly.")
    else:
        print(f"\n  >> Trace did not break the gap.")
    print("=" * 55)


if __name__ == "__main__":
    main()
