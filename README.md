# Multi-Stage Knowledge Distillation for Multimodal BEVFusion

This repository is the public reproducibility bundle for the manuscript, *Multi-Stage Knowledge Distillation for Resource-Constrained Multimodal BEVFusion 3D Object Detection*, prepared for the *Journal of Real-Time Image Processing*.

## Scope and limitations

The bundle documents the validated thesis experiments: recorded environment settings, experiment/run mappings, derived results, and analysis scripts. It is **not** a release of the full BEVFusion training implementation, model checkpoints, or nuScenes data. The reported ablations are continued-training, single-run observations on the nuScenes validation split; they do not establish causal effects or statistical significance. Runtime values are isolated-model measurements, not end-to-end vehicle-pipeline guarantees.

The teacher is an internal reproduction baseline. Do not compare its absolute scores directly with published BEVFusion implementations or differently configured external systems.

## Repository map

| Path | Contents |
| --- | --- |
| `environment/` | Environment and dependency specification |
| `configs/` | Recorded training and inference configuration summary |
| `results/` | Derived metric tables and the run-to-thesis mapping |
| `scripts/` | Approved analysis and plotting scripts |
| `ARTIFACT_MANIFEST.md` | Provenance, inclusion status, and checksums |

## Environment and result reproduction

Create an environment from `environment/environment.yml`, then install the listed MMDetection3D-compatible packages. The scripts require Python, pandas, NumPy, and Matplotlib. Scripts that refresh W&B histories also require an authorized W&B account; the derived CSV files already included here are sufficient to reproduce the documented tables and plots without dataset access.

```bash
conda env create -f environment/environment.yml
conda activate bevfusion-multistage-kd
python scripts/generate_thesis_wandb_figures.py
```

The plotting script reads `results/thesis_run_metrics.csv` and writes derived figures locally. It does not train or evaluate a detector.

## nuScenes access

nuScenes is not included. Obtain access directly from the dataset provider and comply with its licence and terms before using any configuration with the dataset. This repository must not be used to redistribute raw data, annotations, sensor recordings, or derivative assets whose redistribution is restricted.

## Citation

Until publication, cite this repository with the manuscript title and the repository URL. Update this section with the DOI and final bibliographic record after acceptance.

## Contact

Bambang Riyanto Trilaksono (corresponding author): bambang.riyanto@itb.ac.id
