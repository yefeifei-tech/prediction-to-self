"""
lorenz.py — Lorenz attractor signal generator
==============================================
Generates a 4-channel observation from the Lorenz system:
  - ch0: x (normalized)
  - ch1: y (normalized)  
  - ch2: z (normalized)
  - ch3: x*y nonlinear interaction (normalized)

The Lorenz system is chaotic: deterministic but extremely sensitive
to initial conditions. This is a much harder prediction target than
sinusoids — if agency gain still works here, it's a stronger result.

Usage:
    python lorenz.py              # demo: plot the attractor + signal
    python lorenz.py --test       # quick stats check
"""

import sys
import numpy as np
import torch


class LorenzSignal:
    """Lorenz attractor with continuous stepping.
    
    Classic parameters: sigma=10, rho=28, beta=8/3
    These produce the famous butterfly attractor.
    """

    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0,
                 dt=0.005, noise_std=0.02, seed=42):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.noise_std = noise_std

        # Initial condition (slightly off the attractor, it will settle)
        rng = np.random.RandomState(seed)
        self.x = 1.0 + rng.randn() * 0.1
        self.y = 1.0 + rng.randn() * 0.1
        self.z = 1.0 + rng.randn() * 0.1

        # Run 1000 steps to get onto the attractor
        for _ in range(1000):
            self._step()

        # Collect stats for normalization (run 10K steps, record, rewind)
        xs, ys, zs = [], [], []
        sx, sy, sz = self.x, self.y, self.z  # save state
        for _ in range(10000):
            self._step()
            xs.append(self.x)
            ys.append(self.y)
            zs.append(self.z)
        self.x_mean, self.x_std = np.mean(xs), np.std(xs)
        self.y_mean, self.y_std = np.mean(ys), np.std(ys)
        self.z_mean, self.z_std = np.mean(zs), np.std(zs)
        self.x, self.y, self.z = sx, sy, sz  # restore state

    def _step(self):
        """One Runge-Kutta 4 integration step."""
        x, y, z = self.x, self.y, self.z
        s, r, b, dt = self.sigma, self.rho, self.beta, self.dt

        def deriv(x, y, z):
            dx = s * (y - x)
            dy = x * (r - z) - y
            dz = x * y - b * z
            return dx, dy, dz

        k1x, k1y, k1z = deriv(x, y, z)
        k2x, k2y, k2z = deriv(x + dt/2*k1x, y + dt/2*k1y, z + dt/2*k1z)
        k3x, k3y, k3z = deriv(x + dt/2*k2x, y + dt/2*k2y, z + dt/2*k2z)
        k4x, k4y, k4z = deriv(x + dt*k3x, y + dt*k3y, z + dt*k3z)

        self.x += dt/6 * (k1x + 2*k2x + 2*k3x + k4x)
        self.y += dt/6 * (k1y + 2*k2y + 2*k3y + k4y)
        self.z += dt/6 * (k1z + 2*k2z + 2*k3z + k4z)

    def _normalize(self, val, mean, std):
        return (val - mean) / (std + 1e-8)

    def get(self) -> torch.Tensor:
        """Advance one step, return 4-channel normalized observation."""
        self._step()

        xn = self._normalize(self.x, self.x_mean, self.x_std)
        yn = self._normalize(self.y, self.y_mean, self.y_std)
        zn = self._normalize(self.z, self.z_mean, self.z_std)
        interaction = xn * yn  # nonlinear channel

        obs = np.array([xn, yn, zn, interaction], dtype=np.float32)
        obs += np.random.randn(4).astype(np.float32) * self.noise_std

        return torch.from_numpy(obs)


# ============================================================
# Demo / Test
# ============================================================
if __name__ == "__main__":

    signal = LorenzSignal()

    if "--test" in sys.argv:
        # Quick stats check
        print("Lorenz signal test")
        print("=" * 40)
        obs_list = [signal.get() for _ in range(5000)]
        obs_all = torch.stack(obs_list)
        print(f"  Shape: {obs_all.shape}")
        print(f"  Mean:  {obs_all.mean(0).numpy().round(3)}")
        print(f"  Std:   {obs_all.std(0).numpy().round(3)}")
        print(f"  Min:   {obs_all.min(0).values.numpy().round(3)}")
        print(f"  Max:   {obs_all.max(0).values.numpy().round(3)}")

        # Check it's actually chaotic (positive Lyapunov = low autocorrelation at long lags)
        ch0 = obs_all[:, 0].numpy()
        lag1 = np.corrcoef(ch0[:-1], ch0[1:])[0, 1]
        lag50 = np.corrcoef(ch0[:-50], ch0[50:])[0, 1]
        lag200 = np.corrcoef(ch0[:-200], ch0[200:])[0, 1]
        print(f"\n  Autocorrelation ch0:")
        print(f"    lag-1:   {lag1:.3f}  (high = smooth)")
        print(f"    lag-50:  {lag50:.3f}  (medium)")
        print(f"    lag-200: {lag200:.3f}  (low = chaotic)")
        print("\n  ✓ Lorenz signal works")

    else:
        # Visual demo
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            N = 10000
            data = torch.stack([signal.get() for _ in range(N)]).numpy()

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle("Lorenz Chaotic Signal — 4 Channels", fontsize=14)

            # Time series
            ax = axes[0, 0]
            t = np.arange(N)
            for i, name in enumerate(["x", "y", "z", "x*y"]):
                ax.plot(t[:2000], data[:2000, i], lw=0.5, label=name, alpha=0.8)
            ax.set_xlabel("step")
            ax.set_ylabel("normalized value")
            ax.set_title("Time series (first 2000 steps)")
            ax.legend()

            # XY phase portrait
            ax = axes[0, 1]
            ax.scatter(data[:, 0], data[:, 1], s=0.3, c=t, cmap="viridis", alpha=0.5)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("Phase portrait (x vs y)")

            # XZ phase portrait (the butterfly)
            ax = axes[1, 0]
            ax.scatter(data[:, 0], data[:, 2], s=0.3, c=t, cmap="viridis", alpha=0.5)
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            ax.set_title("Phase portrait (x vs z) — the butterfly")

            # Autocorrelation
            ax = axes[1, 1]
            ch0 = data[:, 0]
            ch0_norm = (ch0 - ch0.mean()) / (ch0.std() + 1e-8)
            lags = range(0, 500, 5)
            acorr = []
            for lag in lags:
                if lag == 0:
                    acorr.append(1.0)
                else:
                    n = len(ch0_norm) - lag
                    acorr.append(np.dot(ch0_norm[:n], ch0_norm[lag:lag+n]) / n)
            ax.plot(list(lags), acorr, lw=1.5)
            ax.axhline(0, color="gray", ls="--", alpha=0.5)
            ax.set_xlabel("lag")
            ax.set_ylabel("autocorrelation")
            ax.set_title("Autocorrelation of x channel")

            plt.tight_layout()
            plt.savefig("lorenz_demo.png", dpi=150)
            print("  Saved lorenz_demo.png")
            plt.close()

        except ImportError:
            print("  matplotlib not available, run with --test instead")