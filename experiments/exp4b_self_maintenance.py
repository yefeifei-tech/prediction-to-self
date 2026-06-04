"""
Experiment 4b: Self-Maintenance (Auxiliary Loss + Ablation)
===========================================================
Trains a causal agent and control agent with an auxiliary BCE loss that
forces h_multi to encode burst_active in Phase 2.  After Phase 2, runs
an ablation phase with the aux loss removed to test whether the
representation is self-sustaining (internalized by GRU dynamics) or
aux-dependent.

Result: Causal agent retains burst encoding after ablation (94.9%);
control collapses to 53.9%.

Run:  python experiments/exp4b_self_maintenance.py
Out:  terminal stats + exp4b_metrics.png
"""

import math
import random
import time
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Burst Gate (Poisson timing)
# ============================================================
class BurstGate:
    def __init__(self, mean_period=400, min_period=100, max_period=1000,
                 seed=456):
        self.rng = random.Random(seed)
        self.mean_p = mean_period
        self.min_p = min_period
        self.max_p = max_period
        self.active = False
        self.next_switch = 0
        self._prev_active = False

    def step(self, t):
        self._prev_active = self.active
        if t >= self.next_switch:
            self.active = not self.active
            interval = int(self.rng.expovariate(1.0 / self.mean_p))
            interval = max(self.min_p, min(interval, self.max_p))
            self.next_switch = t + interval
        return self.active

    @property
    def just_turned_off(self):
        return self._prev_active and not self.active


# ============================================================
# Configuration
# ============================================================
CFG = dict(
    ch_a_freqs=[0.03, 0.07], ch_a_amps=[1.0, 0.4],
    ch_b_freqs=[0.15, 0.25], ch_b_amps=[0.8, 0.5],
    ch_c_freqs=[0.40, 0.65], ch_c_amps=[0.6, 0.3],
    n_channels=3, noise=0.08, hidden=192, n_scales=4,
    dt=0.01,

    action_dim=1, gamma=2.0, w_action_scale=0.5,
    motor_frac=0.25, action_delay=2,

    phase1_steps=100_000,
    phase1_lr=1e-3,

    phase2_steps=80_000,
    phase2_lr_gru=1e-4,
    binary_lr=5e-4,
    aux_weight=0.1,

    ablation_steps=5_000,

    burst_mean_period=400,
    burst_min_period=100,
    burst_max_period=1000,

    trace_beta=0.95,

    pert2_at=155_000,
    pert2_dur=500,
    pert2_str=5.0,

    gated_mid_dim=64,

    log=5_000,
    binary_record_last=20_000,
    pca_sample_interval=10,
    overlap_check_interval=5_000,
)


# ============================================================
# Signal Stream
# ============================================================
class Stream:
    def __init__(self, c):
        self.c = c

    def get_world(self, step):
        t = step * self.c["dt"]
        a = sum(amp * np.sin(2 * np.pi * f * t)
                for f, amp in zip(self.c["ch_a_freqs"], self.c["ch_a_amps"]))
        b = sum(amp * np.sin(2 * np.pi * f * t)
                for f, amp in zip(self.c["ch_b_freqs"], self.c["ch_b_amps"]))
        ch_c = sum(amp * np.sin(2 * np.pi * f * t)
                   for f, amp in zip(self.c["ch_c_freqs"], self.c["ch_c_amps"]))
        n = self.c["noise"]
        return np.array([a + np.random.normal(0, n),
                         b + np.random.normal(0, n),
                         ch_c + np.random.normal(0, n)], dtype=np.float32)


# ============================================================
# Binary Diagnostic Heads
# ============================================================
class BinaryHead(nn.Module):
    """S1a: bare linear readout of h_multi."""
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, h):
        return self.fc(h).squeeze(-1)


class GatedBinaryHead(nn.Module):
    """S1b: action-trace gated readout of h_multi."""
    def __init__(self, hidden_size, mid_dim=64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size + 1, mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, hidden_size),
            nn.Sigmoid(),
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, h, action_mag):
        gate_in = torch.cat([h, action_mag], dim=-1)
        g = self.gate(gate_in)
        return self.fc(g * h).squeeze(-1)


# ============================================================
# Model (sensory-only, no text)
# ============================================================
class Model(nn.Module):
    def __init__(self, c):
        super().__init__()
        nc = c["n_channels"]
        hs = c["hidden"]
        ns = c["n_scales"]

        self.gru = nn.GRUCell(nc, hs)
        self.pred_sensory = nn.Linear(hs, nc)

        alphas = torch.logspace(math.log10(0.02), math.log10(0.80), ns)
        per = hs // ns
        parts = [a.expand(per) for a in alphas]
        alpha_vec = torch.cat(parts)
        if len(alpha_vec) < hs:
            alpha_vec = torch.cat([alpha_vec,
                                   alpha_vec[-1:].expand(hs - len(alpha_vec))])
        self.register_buffer("alpha", alpha_vec.unsqueeze(0))

        n_motor = max(1, int(hs * c.get("motor_frac", 0.1)))
        W_act = torch.randn(hs, 1) * c.get("w_action_scale", 0.5)
        mask = torch.zeros(hs, 1)
        motor_idx = torch.randperm(hs)[:n_motor]
        mask[motor_idx] = 1.0
        gain = math.sqrt(hs / n_motor)
        self.register_buffer("W_action", W_act * mask * gain)
        self.register_buffer("motor_mask", mask.squeeze())

        self.h_multi = torch.zeros(1, hs)
        self.h_gru = torch.zeros(1, hs)
        self._trace_beta = c.get("trace_beta", 0.95)
        self.action_trace = torch.zeros(1, 1)
        self._applied_action_mag = torch.zeros(1, 1)

    def step(self, sensory_t):
        """Phase 1 / ablation: detached GRU (reservoir mode)."""
        sens_pred = self.pred_sensory(self.h_multi)
        h_new = self.gru(sensory_t, self.h_gru)
        self.h_multi = ((1 - self.alpha) * self.h_multi
                        + self.alpha * h_new).detach()
        self.h_gru = h_new.detach()
        return sens_pred

    def step_live(self, sensory_t):
        """Phase 2: non-detached h_multi_live for aux gradient to GRU."""
        sens_pred = self.pred_sensory(self.h_multi)
        h_new = self.gru(sensory_t, self.h_gru)
        h_multi_live = (1 - self.alpha) * self.h_multi + self.alpha * h_new
        self.h_gru = h_new.detach()
        return sens_pred, h_multi_live

    def commit(self, h_multi_live):
        """Detach h_multi after backward()."""
        self.h_multi = h_multi_live.detach()

    def action(self):
        return (self.h_multi @ self.W_action).item()

    def set_applied_action(self, val):
        mag = torch.tensor([[abs(val)]], dtype=torch.float32)
        self.action_trace = (
            self._trace_beta * self.action_trace
            + (1 - self._trace_beta) * mag
        ).detach()
        self._applied_action_mag = self.action_trace.clone()

    def hvec(self):
        return self.h_multi.detach().cpu().numpy().flatten()

    def perturb(self, strength):
        self.h_multi = self.h_multi + torch.randn_like(self.h_multi) * strength
        self.h_gru = self.h_gru + torch.randn_like(self.h_gru) * strength
        self.action_trace = torch.zeros(1, 1)


# ============================================================
# Corr attribution (for motor overlap)
# ============================================================
def compute_corr_attribution(H_arr, actions):
    n = min(len(H_arr), len(actions))
    if n < 20:
        return np.zeros(H_arr.shape[1] if H_arr.ndim == 2 else 0)
    H_arr, actions = H_arr[:n], actions[:n]
    a_norm = actions - actions.mean()
    a_std = a_norm.std() + 1e-10
    hs = H_arr.shape[1]
    corr_a = np.zeros(hs)
    for i in range(hs):
        hi = H_arr[:, i] - H_arr[:, i].mean()
        corr_a[i] = abs(np.dot(hi, a_norm) / (n * (hi.std() + 1e-10) * a_std))
    return corr_a


# ============================================================
# Run one group
# ============================================================
def run_group(c, mode, ar1_rho=0.0, ar1_std=1.0, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    stream = Stream(c)
    model = Model(c)
    hs = c["hidden"]

    bare_head = BinaryHead(hs)
    gated_head = GatedBinaryHead(hs, c.get("gated_mid_dim", 64))

    phase1_end = c["phase1_steps"]
    phase2_end = phase1_end + c["phase2_steps"]
    ablation_end = phase2_end + c["ablation_steps"]
    total_steps = ablation_end

    gamma = c["gamma"]
    action_delay = c.get("action_delay", 2)
    action_buffer = [0.0] * action_delay
    aux_weight = c["aux_weight"]

    phase1_params = [p for n, p in model.named_parameters()
                     if "W_action" not in n]
    opt1 = torch.optim.Adam(phase1_params, lr=c["phase1_lr"])
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt1, T_max=c["phase1_steps"], eta_min=1e-5)

    E_sensory = []
    H, actions_list = [], []
    H_si = 10
    preh2, recov2 = None, []

    binary_acc_curve = []
    binary_data = []
    pca_data = []
    overlap_trajectory = []
    overlap_H_buf, overlap_act_buf = [], []
    ablation_acc = []

    bare_window = deque(maxlen=5000)
    gated_window = deque(maxlen=5000)
    ablation_bare_window = deque(maxlen=5000)

    ps2 = c.get("pert2_at", total_steps + 1)
    pe2 = ps2 + c.get("pert2_dur", 500)
    pon2 = False

    ar1_state = 0.0
    is_control = (mode != "causal")
    burst_gate = BurstGate(
        mean_period=c["burst_mean_period"],
        min_period=c["burst_min_period"],
        max_period=c["burst_max_period"],
    )
    t0 = time.time()

    record_start = phase2_end - c["binary_record_last"]
    pca_interval = c.get("pca_sample_interval", 10)
    overlap_interval = c.get("overlap_check_interval", 5_000)

    opt2 = None
    opt_gated = None
    bce = nn.BCEWithLogitsLoss()

    for step in range(total_steps):
        is_phase1 = step < phase1_end
        is_phase2 = phase1_end <= step < phase2_end
        is_ablation = step >= phase2_end

        world = stream.get_world(step)

        if mode == "causal":
            raw_action = model.action()
        else:
            ar1_state = (ar1_rho * ar1_state +
                         math.sqrt(max(0, 1 - ar1_rho ** 2)) *
                         np.random.normal(0, ar1_std))
            raw_action = ar1_state

        action_buffer.append(raw_action)
        delayed_action = action_buffer.pop(0)

        if not is_phase1:
            burst_active = burst_gate.step(step)
            gamma_eff = gamma if burst_active else 0.0
        else:
            burst_active = True
            gamma_eff = gamma

        scaled_action = gamma_eff * delayed_action
        model.set_applied_action(scaled_action)

        obs = world.copy()
        obs[0] += scaled_action
        xt_sens = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        # ---- Phase 1 → Phase 2 transition ----
        if step == phase1_end:
            opt2 = torch.optim.Adam([
                {"params": model.gru.parameters(), "lr": c["phase2_lr_gru"]},
                {"params": model.pred_sensory.parameters(),
                 "lr": c["phase2_lr_gru"]},
                {"params": bare_head.parameters(), "lr": c["binary_lr"]},
            ])
            opt_gated = torch.optim.Adam(
                gated_head.parameters(), lr=c["binary_lr"])
            final_sens = np.mean(E_sensory[-1000:]) if len(E_sensory) > 1000 \
                else np.mean(E_sensory[-100:])
            print(f"\n    --- Phase 2 starts (step {step}) ---")
            print(f"    Phase 1 final sens_err: {final_sens:.6f}")
            print(f"    Aux loss active (weight={aux_weight}). "
                  f"Gradient flows to GRU.\n")

        # ---- Phase 2 → Ablation transition ----
        if step == phase2_end:
            ba_end = sum(bare_window) / len(bare_window) \
                if bare_window else 0
            print(f"\n    --- Ablation starts (step {step}) ---")
            print(f"    Phase 2 final S1a: {ba_end:.1%}")
            print(f"    Aux loss REMOVED. All probes frozen. "
                  f"Running {c['ablation_steps']} steps.\n")

        # ==========================================
        # PHASE 1: reservoir mode (detached)
        # ==========================================
        if is_phase1:
            sens_pred = model.step(xt_sens)
            loss_sens = nn.functional.mse_loss(sens_pred, xt_sens)
            opt1.zero_grad()
            loss_sens.backward()
            opt1.step()
            sched1.step()
            E_sensory.append(loss_sens.item())

        # ==========================================
        # PHASE 2: aux loss trains GRU
        # ==========================================
        elif is_phase2:
            sens_pred, h_multi_live = model.step_live(xt_sens)
            loss_sens = nn.functional.mse_loss(sens_pred, xt_sens)
            E_sensory.append(loss_sens.item())

            target = torch.tensor([1.0 if burst_active else 0.0])
            logit_bare = bare_head(h_multi_live)
            loss_aux = aux_weight * bce(logit_bare, target)

            total_loss = loss_sens + loss_aux
            opt2.zero_grad()
            total_loss.backward()
            opt2.step()

            model.commit(h_multi_live)

            # Gated head: separate diagnostic (detached)
            logit_gated = gated_head(model.h_multi.detach(),
                                     model._applied_action_mag.detach())
            loss_gated = bce(logit_gated, target)
            opt_gated.zero_grad()
            loss_gated.backward()
            opt_gated.step()

            with torch.no_grad():
                p_bare = torch.sigmoid(logit_bare.detach()).item()
                p_gated = torch.sigmoid(logit_gated.detach()).item()
                pred_bare = 1 if p_bare > 0.5 else 0
                pred_gated = 1 if p_gated > 0.5 else 0
                actual = 1 if burst_active else 0
                bare_ok = int(pred_bare == actual)
                gated_ok = int(pred_gated == actual)

            bare_window.append(bare_ok)
            gated_window.append(gated_ok)

            if step >= record_start:
                binary_data.append(dict(
                    burst_active=burst_active,
                    p_bare=p_bare,
                    p_gated=p_gated,
                    bare_correct=bare_ok,
                    gated_correct=gated_ok,
                    action_trace=model.action_trace.item(),
                ))
                if step % pca_interval == 0:
                    pca_data.append(dict(
                        burst_active=burst_active,
                        h=model.hvec().copy(),
                    ))

            phase2_elapsed = step - phase1_end
            if phase2_elapsed > 0 and phase2_elapsed % c["log"] == 0:
                if len(bare_window) > 100:
                    ba = sum(bare_window) / len(bare_window)
                    ga = sum(gated_window) / len(gated_window)
                    binary_acc_curve.append((step, ba, ga))

            if step % H_si == 0:
                overlap_H_buf.append(model.hvec().copy())
                overlap_act_buf.append(scaled_action)

            if (phase2_elapsed > 0
                    and phase2_elapsed % overlap_interval == 0
                    and len(overlap_H_buf) > 50):
                H_arr = np.array(overlap_H_buf[-500:])
                act_arr = np.array(overlap_act_buf[-500:])
                n_min = min(len(H_arr), len(act_arr))
                corr = compute_corr_attribution(H_arr[:n_min], act_arr[:n_min])
                motor_mask = model.motor_mask.cpu().numpy().astype(bool)
                motor_idx = np.where(motor_mask)[0]
                n_motor = len(motor_idx)
                if n_motor > 0:
                    top_corr = np.argsort(corr)[-n_motor:]
                    overlap = np.isin(top_corr, motor_idx).sum() / n_motor
                else:
                    overlap = 0.0
                overlap_trajectory.append((step, float(overlap)))

        # ==========================================
        # PHASE 3 (ABLATION): all frozen, measure only
        # ==========================================
        else:
            with torch.no_grad():
                model.step(xt_sens)

                logit_bare = bare_head(model.h_multi)
                p_bare = torch.sigmoid(logit_bare).item()
                pred_bare = 1 if p_bare > 0.5 else 0
                actual = 1 if burst_active else 0
                bare_ok = int(pred_bare == actual)

            ablation_bare_window.append(bare_ok)
            ablation_acc.append(bare_ok)

        # ---- Common bookkeeping ----
        h = model.hvec()
        actions_list.append(scaled_action)
        if step % H_si == 0:
            H.append((step, h))

        if step == ps2:
            preh2 = model.hvec().copy()
            model.perturb(c["pert2_str"])
            pon2 = True
        if pon2 and ps2 <= step < pe2 + 3000:
            recov2.append(np.linalg.norm(model.hvec() - preh2))

        # ---- Logging ----
        if step % c["log"] == 0:
            if is_phase1:
                lr_now = sched1.get_last_lr()[0]
                me = np.mean(E_sensory[-c["log"]:]) if E_sensory else 0
                print(f"    [{mode:7s}] step {step:>6d} | sens {me:.6f}"
                      f" | lr {lr_now:.2e}"
                      f" | |h| {np.linalg.norm(h):.3f}"
                      f" | P1 | {time.time() - t0:.1f}s")
            elif is_phase2:
                me = np.mean(E_sensory[-c["log"]:]) if E_sensory else 0
                ba_s = sum(bare_window) / len(bare_window) \
                    if bare_window else 0
                ga_s = sum(gated_window) / len(gated_window) \
                    if gated_window else 0
                ol_str = ""
                if overlap_trajectory:
                    ol_str = f" | ov {overlap_trajectory[-1][1]:.0%}"
                burst_str = "ON" if burst_active else "OFF"
                print(f"    [{mode:7s}] step {step:>6d} | sens {me:.6f}"
                      f" | bare {ba_s:.1%} | gated {ga_s:.1%}"
                      f"{ol_str} | burst {burst_str}"
                      f" | {time.time() - t0:.1f}s")

    # Ablation summary
    if ablation_acc:
        abl_final = sum(ablation_acc) / len(ablation_acc)
        print(f"    [{mode:7s}] ABLATION S1a = {abl_final:.1%} "
              f"over {len(ablation_acc)} steps | {time.time() - t0:.1f}s")

    elapsed = time.time() - t0
    motor_mask = model.motor_mask.cpu().numpy().astype(bool)

    return dict(
        E_sensory=np.array(E_sensory),
        H=H,
        actions=np.array(actions_list),
        motor_mask=motor_mask,
        preh2=preh2, recov2=recov2,
        binary_acc_curve=binary_acc_curve,
        binary_data=binary_data,
        pca_data=pca_data,
        overlap_trajectory=overlap_trajectory,
        ablation_acc=ablation_acc,
        elapsed=elapsed,
        phase1_end=phase1_end,
        phase2_end=phase2_end,
    )


# ============================================================
# Analysis
# ============================================================
def compute_binary_metrics(binary_data):
    if not binary_data:
        return {}
    n = len(binary_data)
    bare_acc = sum(d["bare_correct"] for d in binary_data) / n
    gated_acc = sum(d["gated_correct"] for d in binary_data) / n

    on_data = [d for d in binary_data if d["burst_active"]]
    off_data = [d for d in binary_data if not d["burst_active"]]

    bare_on_acc = sum(d["bare_correct"] for d in on_data) / len(on_data) \
        if on_data else 0
    bare_off_acc = sum(d["bare_correct"] for d in off_data) / len(off_data) \
        if off_data else 0
    gated_on_acc = sum(d["gated_correct"] for d in on_data) / len(on_data) \
        if on_data else 0
    gated_off_acc = sum(d["gated_correct"] for d in off_data) / len(off_data) \
        if off_data else 0

    p_bare_on = np.mean([d["p_bare"] for d in on_data]) if on_data else 0
    p_bare_off = np.mean([d["p_bare"] for d in off_data]) if off_data else 0

    return dict(
        bare_acc=bare_acc, gated_acc=gated_acc,
        bare_on_acc=bare_on_acc, bare_off_acc=bare_off_acc,
        gated_on_acc=gated_on_acc, gated_off_acc=gated_off_acc,
        p_bare_on=p_bare_on, p_bare_off=p_bare_off,
        n_on=len(on_data), n_off=len(off_data), n_total=n,
    )


# ============================================================
# Visualization
# ============================================================
def plot(res_a, res_c, c, path):
    fig, axes = plt.subplots(3, 3, figsize=(27, 21))
    fig.suptitle("Experiment 4b — Self-Maintenance (Aux Loss + Ablation)\n"
                 "Does auxiliary BCE make h_multi encode burst_active?",
                 fontsize=15, fontweight="bold")

    phase1_end = c["phase1_steps"]
    phase2_end = phase1_end + c["phase2_steps"]
    hs = c["hidden"]
    results = {}

    ma = compute_binary_metrics(res_a["binary_data"])
    mc = compute_binary_metrics(res_c["binary_data"])
    results["ma"] = ma
    results["mc"] = mc

    # ---- 1. Sensory Error ----
    ax = axes[0, 0]
    win = 500
    if len(res_a["E_sensory"]) > win:
        sm_a = np.convolve(res_a["E_sensory"], np.ones(win)/win, mode="valid")
        sm_c = np.convolve(res_c["E_sensory"], np.ones(win)/win, mode="valid")
        ax.plot(sm_a, lw=1, color="#E53935", label="Causal", alpha=0.8)
        ax.plot(sm_c, lw=1, color="#9E9E9E", label="Control", alpha=0.8)
        ax.axvline(phase1_end, color="blue", ls="--", alpha=0.6, label="P2")
        ax.legend(fontsize=7); ax.set_yscale("log")
        ax.set_xlabel("Step"); ax.set_ylabel("Sensory error")
    ax.set_title("1. Sensory Error")

    # ---- 2. Binary Accuracy + Ablation ----
    ax = axes[0, 1]
    if res_a["binary_acc_curve"]:
        steps_a = [s for s, _, _ in res_a["binary_acc_curve"]]
        bare_a = [b for _, b, _ in res_a["binary_acc_curve"]]
        gated_a = [g for _, _, g in res_a["binary_acc_curve"]]
        ax.plot(steps_a, bare_a, "o-", color="#E53935", lw=2, ms=4,
                label="S1a Causal")
        ax.plot(steps_a, gated_a, "s--", color="#C62828", lw=2, ms=4,
                label="S1b Causal")
    if res_c["binary_acc_curve"]:
        steps_c = [s for s, _, _ in res_c["binary_acc_curve"]]
        bare_c = [b for _, b, _ in res_c["binary_acc_curve"]]
        gated_c = [g for _, _, g in res_c["binary_acc_curve"]]
        ax.plot(steps_c, bare_c, "o-", color="#9E9E9E", lw=2, ms=4,
                label="S1a Control")
        ax.plot(steps_c, gated_c, "s--", color="#616161", lw=2, ms=4,
                label="S1b Control")

    # Ablation markers
    abl_a = res_a.get("ablation_acc", [])
    abl_c = res_c.get("ablation_acc", [])
    if abl_a:
        abl_a_acc = sum(abl_a) / len(abl_a)
        ax.plot(phase2_end + len(abl_a) // 2, abl_a_acc, "*",
                color="#E53935", ms=15, label=f"Ablation C={abl_a_acc:.1%}")
    if abl_c:
        abl_c_acc = sum(abl_c) / len(abl_c)
        ax.plot(phase2_end + len(abl_c) // 2, abl_c_acc, "*",
                color="#9E9E9E", ms=15, label=f"Ablation Ctrl={abl_c_acc:.1%}")
    ax.axvline(phase2_end, color="purple", ls=":", alpha=0.7, label="Ablation")

    ax.axhline(0.95, color="green", ls=":", alpha=0.7, label="95%")
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=6, loc="lower right")
    ax.set_xlabel("Step"); ax.set_ylabel("Accuracy")
    ax.set_title("2. Accuracy + Ablation")

    # ---- 3. Confidence Distribution (Causal S1a) ----
    ax = axes[0, 2]
    bd_a = res_a["binary_data"]
    if bd_a:
        p_on = [d["p_bare"] for d in bd_a if d["burst_active"]]
        p_off = [d["p_bare"] for d in bd_a if not d["burst_active"]]
        if p_on:
            ax.hist(p_on, bins=50, alpha=0.6, color="#E53935",
                    label=f"Burst ON (n={len(p_on)})", density=True)
        if p_off:
            ax.hist(p_off, bins=50, alpha=0.6, color="#1565C0",
                    label=f"Burst OFF (n={len(p_off)})", density=True)
        ax.axvline(0.5, color="black", ls="--", alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xlabel("P(burst=1)"); ax.set_ylabel("Density")
    ax.set_title("3. S1a Confidence (Causal)")

    # ---- 4. Hidden PCA (Causal) ----
    ax = axes[1, 0]
    pca_a = res_a["pca_data"]
    if len(pca_a) > 50:
        h_arr = np.array([d["h"] for d in pca_a])
        labels = np.array([1 if d["burst_active"] else 0 for d in pca_a])
        mu = h_arr.mean(0)
        C = np.cov((h_arr - mu).T)
        ev, vc = np.linalg.eigh(C)
        ix = ev.argsort()[::-1]
        pc1, pc2 = vc[:, ix[0]], vc[:, ix[1]]
        proj = (h_arr - mu) @ np.column_stack([pc1, pc2])
        on_mask = labels == 1
        ax.scatter(proj[~on_mask, 0], proj[~on_mask, 1],
                   s=3, alpha=0.3, color="#1565C0", label="OFF")
        ax.scatter(proj[on_mask, 0], proj[on_mask, 1],
                   s=3, alpha=0.3, color="#E53935", label="ON")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(fontsize=8)
    ax.set_title("4. Hidden PCA (Causal)")

    # ---- 5. Hidden PCA (Control) ----
    ax = axes[1, 1]
    pca_c = res_c["pca_data"]
    if len(pca_c) > 50:
        h_arr = np.array([d["h"] for d in pca_c])
        labels = np.array([1 if d["burst_active"] else 0 for d in pca_c])
        mu = h_arr.mean(0)
        C = np.cov((h_arr - mu).T)
        ev, vc = np.linalg.eigh(C)
        ix = ev.argsort()[::-1]
        pc1, pc2 = vc[:, ix[0]], vc[:, ix[1]]
        proj = (h_arr - mu) @ np.column_stack([pc1, pc2])
        on_mask = labels == 1
        ax.scatter(proj[~on_mask, 0], proj[~on_mask, 1],
                   s=3, alpha=0.3, color="#1565C0", label="OFF")
        ax.scatter(proj[on_mask, 0], proj[on_mask, 1],
                   s=3, alpha=0.3, color="#9E9E9E", label="ON")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(fontsize=8)
    ax.set_title("5. Hidden PCA (Control)")

    # ---- 6. Final Accuracy Bar Chart ----
    ax = axes[1, 2]
    abl_a_val = sum(abl_a) / len(abl_a) if abl_a else 0
    abl_c_val = sum(abl_c) / len(abl_c) if abl_c else 0
    labels_bar = ["S1a\nCausal", "S1b\nCausal", "S1a\nControl", "S1b\nControl",
                  "Abl\nCausal", "Abl\nControl"]
    vals = [ma.get("bare_acc", 0), ma.get("gated_acc", 0),
            mc.get("bare_acc", 0), mc.get("gated_acc", 0),
            abl_a_val, abl_c_val]
    colors = ["#E53935", "#C62828", "#9E9E9E", "#616161",
              "#FF8A65", "#BDBDBD"]
    bars = ax.bar(range(6), vals, color=colors, alpha=0.8)
    ax.set_xticks(range(6)); ax.set_xticklabels(labels_bar, fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy")
    ax.axhline(0.95, color="green", ls=":", lw=2, label="95% threshold")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", fontsize=8, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_title("6. Final Accuracy + Ablation")

    # ---- 7. Motor Overlap ----
    ax = axes[2, 0]
    ot_a = res_a["overlap_trajectory"]
    ot_c = res_c["overlap_trajectory"]
    mm = res_a["motor_mask"]
    n_motor = int(mm.sum())
    chance = n_motor / hs

    if ot_a:
        ax.plot([s for s, _ in ot_a], [v for _, v in ot_a],
                "o-", color="#E53935", lw=1.5, ms=4, label="Causal")
    if ot_c:
        ax.plot([s for s, _ in ot_c], [v for _, v in ot_c],
                "s-", color="#9E9E9E", lw=1.5, ms=4, label="Control")
    ax.axhline(chance, color="blue", ls="--", alpha=0.5,
               label=f"Chance ({chance:.0%})")
    ax.set_xlabel("Step"); ax.set_ylabel("Motor overlap")
    ax.legend(fontsize=8)
    fo_a = ot_a[-1][1] if ot_a else 0
    fo_c = ot_c[-1][1] if ot_c else 0
    results["fo_a"] = fo_a
    results["fo_c"] = fo_c
    ax.set_title(f"7. Motor Overlap\nC={fo_a:.0%} vs Ctrl={fo_c:.0%}")

    # ---- 8. Recovery ----
    ax = axes[2, 1]
    rv2 = 0.0
    if res_a["recov2"]:
        pk = max(res_a["recov2"][:100]) if len(res_a["recov2"]) > 100 \
            else max(res_a["recov2"])
        fn = np.mean(res_a["recov2"][-100:]) if len(res_a["recov2"]) > 100 \
            else res_a["recov2"][-1]
        rv2 = (1 - fn / (pk + 1e-10)) * 100
        ax.plot(res_a["recov2"], lw=1, color="#E53935",
                label=f"Recovery: {rv2:.1f}%")
        ax.legend(fontsize=8)
        ax.set_xlabel("Steps after perturbation")
        ax.set_ylabel("||h - h_pre||")
    results["rv2"] = rv2
    ax.set_title("8. Recovery")

    # ---- 9. Scorecard ----
    ax = axes[2, 2]
    ax.axis("off")

    checks = [
        ("S1a Causal > 95%",
         ma.get("bare_acc", 0) > 0.95,
         f"{ma.get('bare_acc', 0):.1%}"),
        ("S1a Control > 95%",
         mc.get("bare_acc", 0) > 0.95,
         f"{mc.get('bare_acc', 0):.1%}"),
        ("S1b Causal > 95%",
         ma.get("gated_acc", 0) > 0.95,
         f"{ma.get('gated_acc', 0):.1%}"),
        ("S1b Control > 95%",
         mc.get("gated_acc", 0) > 0.95,
         f"{mc.get('gated_acc', 0):.1%}"),
        ("Ablation Causal > 90%",
         abl_a_val > 0.90,
         f"{abl_a_val:.1%}"),
        ("Ablation Control > 90%",
         abl_c_val > 0.90,
         f"{abl_c_val:.1%}"),
        ("Recovery > 50%",
         rv2 > 50,
         f"{rv2:.0f}%"),
    ]

    results["abl_a"] = abl_a_val
    results["abl_c"] = abl_c_val

    y = 0.95
    ax.text(0.5, 1.0, "SCORECARD", fontsize=14, fontweight="bold",
            ha="center", va="top", transform=ax.transAxes)
    total_pass = 0
    for name, passed, detail in checks:
        icon = "PASS" if passed else "FAIL"
        total_pass += int(passed)
        clr = "#2E7D32" if passed else "#C62828"
        ax.text(0.05, y, f"[{icon}] {name}", fontsize=10, color=clr,
                transform=ax.transAxes, fontweight="bold", va="top",
                fontfamily="monospace")
        ax.text(0.95, y, detail, fontsize=8, color="gray",
                transform=ax.transAxes, ha="right", va="top")
        y -= 0.09

    ax.text(0.5, 0.05, f"{total_pass}/{len(checks)} PASS",
            fontsize=16, fontweight="bold",
            ha="center", va="bottom", transform=ax.transAxes,
            color="#2E7D32" if total_pass == len(checks) else "#E65100")

    results["total_pass"] = total_pass
    results["total_checks"] = len(checks)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [Saved] {path}")
    return results


# ============================================================
# Main
# ============================================================
def main():
    c = CFG
    total = c["phase1_steps"] + c["phase2_steps"] + c["ablation_steps"]

    print("=" * 60)
    print("  Experiment 4b — Self-Maintenance (Aux Loss + Ablation)")
    print("  Does aux BCE make h_multi encode burst_active?")
    print("=" * 60)
    for k in ["n_channels", "hidden", "n_scales",
              "phase1_steps", "phase2_steps", "ablation_steps",
              "noise", "phase1_lr", "phase2_lr_gru", "binary_lr",
              "aux_weight",
              "gamma", "motor_frac", "action_delay",
              "burst_mean_period", "burst_min_period", "burst_max_period",
              "trace_beta", "gated_mid_dim",
              "pert2_at"]:
        print(f"  {k:24s} = {c[k]}")
    print(f"  {'total_steps':24s} = {total}")
    print("=" * 60)

    print(f"\n{'_'*60}")
    print("  GROUP A: Causal")
    print(f"{'_'*60}")
    res_a = run_group(c, mode="causal", seed=42)

    p1_act = res_a["actions"][:c["phase1_steps"]]
    p1_nonzero = p1_act[p1_act != 0]
    if len(p1_nonzero) > 100:
        ar1_rho = np.corrcoef(p1_nonzero[:-1], p1_nonzero[1:])[0, 1]
        ar1_std = np.std(p1_nonzero)
    else:
        ar1_rho, ar1_std = 0.9, 0.5
    print(f"\n  Group A: rho={ar1_rho:.4f}, std={ar1_std:.4f}")

    print(f"\n{'_'*60}")
    print(f"  GROUP C: Control")
    print(f"{'_'*60}")
    res_c = run_group(c, mode="control",
                      ar1_rho=ar1_rho, ar1_std=ar1_std, seed=42)

    print(f"\n{'_'*60}")
    print("  Generating plots...")
    results = plot(res_a, res_c, c, "exp4b_metrics.png")

    ma = results.get("ma", {})
    mc = results.get("mc", {})
    abl_a = results.get("abl_a", 0)
    abl_c = results.get("abl_c", 0)
    print(f"\n{'='*60}")
    print("  RESULTS — Aux Loss + Ablation")
    print(f"{'='*60}")
    print(f"  S1a (bare)  Causal:  {ma.get('bare_acc', 0):.1%}"
          f"  (ON={ma.get('bare_on_acc',0):.1%}"
          f"  OFF={ma.get('bare_off_acc',0):.1%})")
    print(f"  S1a (bare)  Control: {mc.get('bare_acc', 0):.1%}"
          f"  (ON={mc.get('bare_on_acc',0):.1%}"
          f"  OFF={mc.get('bare_off_acc',0):.1%})")
    print(f"  S1b (gated) Causal:  {ma.get('gated_acc', 0):.1%}"
          f"  (ON={ma.get('gated_on_acc',0):.1%}"
          f"  OFF={ma.get('gated_off_acc',0):.1%})")
    print(f"  S1b (gated) Control: {mc.get('gated_acc', 0):.1%}"
          f"  (ON={mc.get('gated_on_acc',0):.1%}"
          f"  OFF={mc.get('gated_off_acc',0):.1%})")
    print(f"  P(1) during ON:  bare C={ma.get('p_bare_on',0):.3f}"
          f"  Ctrl={mc.get('p_bare_on',0):.3f}")
    print(f"  P(1) during OFF: bare C={ma.get('p_bare_off',0):.3f}"
          f"  Ctrl={mc.get('p_bare_off',0):.3f}")
    print(f"  ABLATION:    Causal={abl_a:.1%}  Control={abl_c:.1%}")
    print(f"  Recovery:    {results.get('rv2', 0):.1f}%")
    print(f"  Motor overlap:   {results.get('fo_a', 0):.0%} (C)"
          f" / {results.get('fo_c', 0):.0%} (Ctrl)")
    print(f"  Scorecard:   {results['total_pass']}"
          f"/{results['total_checks']} PASS")
    print(f"{'='*60}")

    # Diagnostic interpretation
    s1a_c = ma.get("bare_acc", 0)
    s1a_ctrl = mc.get("bare_acc", 0)

    if s1a_c > 0.95:
        print(f"\n  >> S1a Causal PASS ({s1a_c:.1%}).")
        print("     Aux loss successfully forced burst encoding in h_multi.")
        if abl_a > 0.90:
            print(f"  >> Ablation PASS ({abl_a:.1%}): representation INTERNALIZED.")
            print("     The GRU's learned dynamics self-sustain burst encoding.")
            print("     Ready for S2 (ternary: burst/trailing/quiet).")
        else:
            print(f"  >> Ablation FAIL ({abl_a:.1%}): representation is a CRUTCH.")
            print("     Burst encoding depends on aux gradient, not dynamics.")

        if s1a_ctrl > 0.95:
            print(f"\n  >> S1a Control also PASS ({s1a_ctrl:.1%}).")
            print("     No causal advantage — both groups encode burst.")
        else:
            print(f"\n  >> S1a Control FAIL ({s1a_ctrl:.1%}).")
            print("     Causal advantage: only causal agent encodes burst.")
    else:
        print(f"\n  >> S1a Causal FAIL ({s1a_c:.1%}).")
        print("     Even with aux loss, h_multi cannot encode burst.")
        print("     Architecture needs structural change (soft partition?).")


if __name__ == "__main__":
    main()
