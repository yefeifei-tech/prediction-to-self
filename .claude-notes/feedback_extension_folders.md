---
name: Extensions go in separate folders, not existing experiments/
description: When extending Paper 1's experiments with new work (CET-driven or otherwise), create a sibling folder and inline any model variants — do not touch experiments/ or core/
type: feedback
originSessionId: 88fc7d6b-54be-4eda-af17-d7042a1ed263
---
For any experiment beyond exp1–exp6 in this repo, create a new sibling folder (e.g., `experiments_ext/`, `experiments_meta/`). Do not modify files under `experiments/` or `core/`.

If a new experiment needs a modified model, define the variant inline in the extension experiment file rather than editing `core/model.py`.

**Why**: Paper 1's experiment structure and model must remain reproducible as-is. Extensions carry the risk of breaking the reproducibility of published results, and are also epistemically distinct — they belong to Paper 2+ or exploratory work, not to Paper 1.

**How to apply**: When the user proposes a new experimental direction, default to proposing a new folder path. Only touch `experiments/` or `core/` if the user explicitly asks for an in-place modification to Paper 1's setup.
