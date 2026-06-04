"""
world.py — Signal generator (switchable)
=========================================
Two signal types:
  - "sine":   4-channel sinusoids (easy, periodic)
  - "lorenz": Lorenz chaotic attractor (hard, non-periodic)

Usage:
    from world import make_signal
    signal = make_signal("sine")    # or "lorenz"
    obs = signal.get()              # shape: (4,)
"""

import numpy as np
import torch


# ============================================================
# Sinusoidal signal (original)
# ============================================================
class SineSignal:
    """4-channel sum-of-sinusoids + noise."""

    CHANNELS = [
        {"freqs": [0.05, 0.13], "amps": [1.0, 0.5]},
        {"freqs": [0.08, 0.21], "amps": [0.8, 0.4]},
        {"freqs": [0.03, 0.17], "amps": [0.9, 0.3]},
        {"freqs": [0.11, 0.29], "amps": [0.7, 0.6]},
    ]
    NOISE = 0.05
    DT = 0.01

    def __init__(self):
        self.t = 0

    def get(self) -> torch.Tensor:
        time = self.t * self.DT
        values = []
        for ch in self.CHANNELS:
            v = sum(a * np.sin(2 * np.pi * f * time)
                    for f, a in zip(ch["freqs"], ch["amps"]))
            v += np.random.normal(0, self.NOISE)
            values.append(v)
        self.t += 1
        return torch.tensor(values, dtype=torch.float32)


# ============================================================
# Lorenz chaotic signal
# ============================================================
class LorenzSignal:
    """Lorenz attractor: ch0=x, ch1=y, ch2=z, ch3=x*y (all normalized)."""

    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0,
                 dt=0.005, noise_std=0.02, seed=42):
        self.sigma, self.rho, self.beta = sigma, rho, beta
        self.dt = dt
        self.noise_std = noise_std

        rng = np.random.RandomState(seed)
        self.x = 1.0 + rng.randn() * 0.1
        self.y = 1.0 + rng.randn() * 0.1
        self.z = 1.0 + rng.randn() * 0.1

        for _ in range(1000):
            self._step()

        xs, ys, zs = [], [], []
        sx, sy, sz = self.x, self.y, self.z
        for _ in range(10000):
            self._step()
            xs.append(self.x); ys.append(self.y); zs.append(self.z)
        self.x_mean, self.x_std = np.mean(xs), np.std(xs)
        self.y_mean, self.y_std = np.mean(ys), np.std(ys)
        self.z_mean, self.z_std = np.mean(zs), np.std(zs)
        self.x, self.y, self.z = sx, sy, sz

    def _step(self):
        x, y, z = self.x, self.y, self.z
        s, r, b, dt = self.sigma, self.rho, self.beta, self.dt

        def d(x, y, z):
            return s*(y-x), x*(r-z)-y, x*y-b*z

        k1 = d(x, y, z)
        k2 = d(x+dt/2*k1[0], y+dt/2*k1[1], z+dt/2*k1[2])
        k3 = d(x+dt/2*k2[0], y+dt/2*k2[1], z+dt/2*k2[2])
        k4 = d(x+dt*k3[0], y+dt*k3[1], z+dt*k3[2])

        self.x += dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
        self.y += dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
        self.z += dt/6*(k1[2]+2*k2[2]+2*k3[2]+k4[2])

    def get(self) -> torch.Tensor:
        self._step()
        xn = (self.x - self.x_mean) / (self.x_std + 1e-8)
        yn = (self.y - self.y_mean) / (self.y_std + 1e-8)
        zn = (self.z - self.z_mean) / (self.z_std + 1e-8)
        obs = np.array([xn, yn, zn, xn*yn], dtype=np.float32)
        obs += np.random.randn(4).astype(np.float32) * self.noise_std
        return torch.from_numpy(obs)


# ============================================================
# Factory
# ============================================================
def make_signal(kind="sine"):
    if kind == "sine":
        return SineSignal()
    elif kind == "lorenz":
        return LorenzSignal()
    else:
        raise ValueError(f"Unknown signal type: {kind}")


if __name__ == "__main__":
    for kind in ["sine", "lorenz"]:
        sig = make_signal(kind)
        obs = torch.stack([sig.get() for _ in range(1000)])
        print(f"{kind:8s} | mean={obs.mean(0).numpy().round(3)} | std={obs.std(0).numpy().round(3)}")