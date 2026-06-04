# From Prediction to Self: Developmental Conditions for Agency in Minimal Neural Systems

Official code for the paper *From Prediction to Self: Developmental Conditions for Agency in Minimal Neural Systems*.

**Paper:** [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) *(link coming soon)*

## Overview

We trace how a minimal 192-dimensional GRU comes to distinguish its own causal influence from world-caused changes, through 40 controlled experiments arranged as a developmental sequence. The work makes three core contributions:

1. **A developmental sequence** from prediction to self-world decomposition, with four sufficient conditions that must be satisfied in strict order: persistent state, causal action loop, proprioceptive feedback, and asynchronous awakening.
2. **The encoding gap** — a system can implicitly compensate for its own actions in prediction while failing to encode "I am acting" as a readable state variable (causal utilization ≠ self-representation).
3. **Self-maintenance** — self-representation is sustained without external supervision only when it is causally useful: after the training signal is removed, the causal agent retains its encoding (94.9%) while a statistically-matched control collapses to chance (53.9%).

We introduce **agency gain** (A = Err_world − Err_self) as a metric to track this developmental process.

## Repository structure

```
prediction-to-self/
├── train.py                      # Main training entry point
├── requirements.txt
├── core/                         # Model and environment definitions
│   ├── model.py                  # AgencyModel (GRU + multi-scale EMA + dual heads)
│   ├── world.py                  # Signal generators
│   └── lorenz.py                 # Lorenz chaotic attractor
├── experiments/                  # One script per developmental stage
│   ├── exp1_perception.py        # Sec 3.1 — Stable attractors
│   ├── exp2_causal.py            # Sec 3.2 — Causal budding
│   ├── exp3_encoding_gap.py      # Sec 3.3 — The encoding gap
│   ├── exp4_proprioception.py    # Sec 3.4 — Proprioceptive breakthrough
│   ├── exp4b_self_maintenance.py # Sec 3.2 — Self-maintenance ablation
│   ├── exp5_async_awakening.py   # Sec 3.5 — Asynchronous awakening
│   └── exp6_measurement.py       # Sec 3.6 — Agency gain measurement
└── figures/                      # Figure generation scripts
    ├── gen_fig1_attractor.py … gen_fig6_strategy.py
    └── output/                   # Generated figures land here
```

## Installation

```bash
git clone https://github.com/<your-username>/prediction-to-self.git
cd prediction-to-self
pip install -r requirements.txt
```

## Running experiments

Each experiment corresponds to a section of the paper and can be run independently:

```bash
python -m experiments.exp1_perception
python -m experiments.exp6_measurement --trace
python -m experiments.exp6_measurement --lorenz --trace
```

Most scripts support `--quick` for a fast reduced-step run.

## Generating figures

```bash
python figures/gen_fig1_attractor.py
# Figures are written to figures/output/
```

## Citation

```bibtex
@article{ye2026prediction,
  title   = {From Prediction to Self: Developmental Conditions for Agency in Minimal Neural Systems},
  author  = {Ye, Evan},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
