# Artifact manifest

| Artifact | Provenance | Included | Notes |
| --- | --- | --- | --- |
| `results/thesis_run_metrics.csv` | Validated thesis W&B export | Yes | Derived per-run metrics and class diagnostics |
| `results/thesis_run_mapping.md` | Validated thesis run mapping | Yes | Maps manuscript labels to W&B runs/checkpoints |
| `results/loss_dynamics_summary.csv` | Validated thesis analysis | Yes | Derived loss and DWA summary statistics |
| `configs/experiment_protocol.md` | Thesis Chapters 3--4 | Yes | Recorded settings; not executable training code |
| `environment/environment.yml` | Thesis environment record | Yes | Reproducibility environment specification |
| `scripts/` | Thesis analysis utilities | Yes | Analysis/plotting only; no training implementation |
| nuScenes data and annotations | nuScenes provider | No | Access and redistribution governed by provider terms |
| Training implementation and checkpoints | Project workspace | No | Redistribution permission not established |

All included tables are derived artifacts. Consult the manuscript and thesis for interpretation, limitations, and selected checkpoints.

## SHA-256 checksums

| File | SHA-256 |
| --- | --- |
| `configs/experiment_protocol.md` | `d23a98a2cb6b8a045f6249395205136cb3a1b08afc3196cc267c75cf9a45ec1c` |
| `environment/environment.yml` | `080867e007b860c820d9cae19519ee42a3d71637a3ad9b88e69a1f2e05a96795` |
| `results/loss_dynamics_summary.csv` | `79b7c1bb571941619c10ec561d357a80a9302b885dd157ca281b66d201dbf767` |
| `results/thesis_run_mapping.md` | `82e0fb109e7db5b627f70b715f59b30b901c84af14d9540abdc201005a85ee0e` |
| `results/thesis_run_metrics.csv` | `f157c9d097e117f80ab3b4fd08ed7f46fc1d0cec893c9b0fd8482e4e4e8` |
