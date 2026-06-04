"""
model.py — GRU + Multi-Scale EMA + Dual Prediction Heads + W_action
====================================================================
Architecture:
    obs(4d) [+ trace(1d)] → GRU(192d) → h_multi(192d) via EMA
                                  ↓               ↓
                             pred_A(h, a)→4d   pred_B(h)→4d
                                  ↓
                             W_action(h)→1d

use_trace=True:  GRU input is 5d (obs + action_trace), as described in the paper (Section 3.4)
use_trace=False: GRU input is 4d (obs only), matches paper v2.0 minimal config
"""

import math
import torch
import torch.nn as nn


class AgencyModel(nn.Module):

    def __init__(self, obs_dim=4, hidden_dim=192, n_scales=4,
                 use_trace=False, trace_beta=0.95):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.use_trace = use_trace
        self.trace_beta = trace_beta

        # GRU input: obs + optional trace
        gru_input_dim = obs_dim + (1 if use_trace else 0)
        self.gru = nn.GRUCell(gru_input_dim, hidden_dim)

        # Multi-Scale EMA
        alphas = torch.logspace(math.log10(0.02), math.log10(0.80), n_scales)
        per_group = hidden_dim // n_scales
        alpha_vec = torch.cat([a.expand(per_group) for a in alphas])
        if len(alpha_vec) < hidden_dim:
            alpha_vec = torch.cat([alpha_vec,
                        alpha_vec[-1:].expand(hidden_dim - len(alpha_vec))])
        self.register_buffer("alpha", alpha_vec.unsqueeze(0))

        # Dual Prediction Heads
        self.pred_A = nn.Linear(hidden_dim + 1, obs_dim)  # h + action → obs
        self.pred_B = nn.Linear(hidden_dim, obs_dim)       # h → obs

        # Action Policy
        self.W_action = nn.Linear(hidden_dim, 1)

        # Persistent State
        self.register_buffer("h_multi", torch.zeros(1, hidden_dim))
        self.register_buffer("h_gru", torch.zeros(1, hidden_dim))

        # Action trace (proprioception)
        self.trace = 0.0

    def predict(self, action: torch.Tensor):
        """Predict from CURRENT h_multi (before updating state)."""
        h = self.h_multi
        ha = torch.cat([h, action.view(1, 1)], dim=1)
        pred_a = self.pred_A(ha).squeeze(0)
        pred_b = self.pred_B(h).squeeze(0)
        return pred_a, pred_b

    def update_state(self, obs: torch.Tensor):
        """Update GRU + EMA with new observation (and trace if enabled)."""
        if self.use_trace:
            trace_t = torch.tensor([[self.trace]], dtype=torch.float32)
            x = torch.cat([obs.unsqueeze(0), trace_t], dim=1)  # (1, 5)
        else:
            x = obs.unsqueeze(0)  # (1, 4)

        h_new = self.gru(x, self.h_gru)
        self.h_multi = ((1 - self.alpha) * self.h_multi
                        + self.alpha * h_new).detach()
        self.h_gru = h_new.detach()

    def update_trace(self, action_val):
        """Update proprioceptive trace: tau(t) = beta*tau(t-1) + (1-beta)*|a(t)|"""
        self.trace = self.trace_beta * self.trace + (1 - self.trace_beta) * abs(action_val)

    def get_action(self):
        """Generate action from detached h_multi."""
        h_detached = self.h_multi.detach()
        return self.W_action(h_detached).squeeze(0)


# Quick test
if __name__ == "__main__":
    print("Model test (no trace)")
    m1 = AgencyModel(use_trace=False)
    print(f"  GRU input: {m1.gru.input_size}d")
    print(f"  Parameters: {sum(p.numel() for p in m1.parameters()):,}")

    print("\nModel test (with trace)")
    m2 = AgencyModel(use_trace=True)
    print(f"  GRU input: {m2.gru.input_size}d")
    print(f"  Parameters: {sum(p.numel() for p in m2.parameters()):,}")

    # One step
    obs = torch.randn(4)
    action = torch.tensor([0.5])

    for name, m in [("no_trace", m1), ("with_trace", m2)]:
        if m.use_trace:
            m.update_trace(0.5)
        pred_a, pred_b = m.predict(action)
        m.update_state(obs)
        print(f"\n  {name}: pred_A={pred_a.detach().numpy().round(3)}, "
              f"h_norm={m.h_multi.norm().item():.4f}")
    print("\n  ✓ Both modes work")