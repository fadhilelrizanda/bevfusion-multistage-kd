#!/usr/bin/env python3
"""Fetch W&B loss histories and generate thesis loss-dynamics artifacts."""

import csv
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import wandb
except ImportError as exc:
    raise SystemExit("wandb Python SDK is required") from exc


ROOT = Path(__file__).resolve().parents[1]
WANDB_DIR = ROOT / "results"
FIG_DIR = ROOT / "analysis"
REPORT_DIR = ROOT / "analysis"

ENTITY = "fadhilelrizanda-student"
PROJECT = "thesis-KD"

RUNS = [
    ("Student S", "S1", "ynh2rrwe", "bevfusion_student_xxs_kd_feature"),
    ("Student S", "S2", "bpbdh38x", "bevfusion_student_xxs_kd_feature_bev"),
    ("Student S", "S3", "lv3ftiq8", "bevfusion_student_xxs_kd_feature_bev_head"),
    ("Student S", "S4", "5zkf0n7i", "bevfusion_student_xxs_kd_feature_bev_head_dwa_V2"),
    ("Student XS", "S1", "2f6bs75l", "student_v2_kd_xxs_feature_lidar_cam"),
    ("Student XS", "S2", "8utrbfn6", "student_v2_kd_xxs_feature_lidar_cam_bev"),
    ("Student XS", "S3", "xprs9b5n", "student_v2_kd_xxs_feature_lidar_cam_bev_head_rel"),
    ("Student XS", "S4", "6u81evo4", "student_v2_kd_xxs_feature_lidar_cam_bev_head_rel_dwa"),
]

LOSS_KEYS = [
    "loss",
    "grad_norm",
    "loss_heatmap",
    "layer_-1_loss_cls",
    "layer_-1_loss_bbox",
    "loss_kd_feature_camera",
    "loss_kd_feature_lidar",
    "loss_kd_feature_bev",
    "loss_kd_bev",
    "loss_kd_rel",
    "loss_kd_logit",
    "weight_lidar",
    "weight_camera",
    "weight_bev",
    "weight_relation",
    "weight_logit",
    "weighted_loss_kd_feature_camera",
    "weighted_loss_kd_feature_lidar",
    "weighted_loss_kd_bev",
    "weighted_loss_kd_rel",
    "weighted_loss_kd_logit",
]

KD_COMPONENTS = [
    ("Camera", "loss_kd_feature_camera"),
    ("LiDAR", "loss_kd_feature_lidar"),
    ("BEV", "loss_kd_feature_bev"),
    ("Relation", "loss_kd_rel"),
    ("Logit", "loss_kd_logit"),
]

WEIGHT_COMPONENTS = [
    ("LiDAR", "weight_lidar"),
    ("Camera", "weight_camera"),
    ("BEV", "weight_bev"),
    ("Relation", "weight_relation"),
    ("Logit", "weight_logit"),
]

METRICS = {
    ("Student S", "S1"): (66.36, 69.47),
    ("Student S", "S2"): (66.50, 69.50),
    ("Student S", "S3"): (67.16, 69.80),
    ("Student S", "S4"): (67.02, 69.59),
    ("Student XS", "S1"): (49.02, 54.06),
    ("Student XS", "S2"): (48.70, 54.16),
    ("Student XS", "S3"): (49.34, 53.41),
    ("Student XS", "S4"): (49.71, 53.83),
}


def load_env() -> None:
    env_path = WANDB_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def finite(value):
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def stats(values):
    clean = [float(v) for v in values if finite(v) is not None]
    if not clean:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None, "last": None}
    mean = sum(clean) / len(clean)
    var = sum((v - mean) ** 2 for v in clean) / len(clean)
    return {
        "n": len(clean),
        "mean": mean,
        "std": math.sqrt(var),
        "min": min(clean),
        "max": max(clean),
        "last": clean[-1],
    }


def rolling_mean(values, window=75):
    arr = np.asarray(values, dtype=float)
    if len(arr) <= 1:
        return arr
    radius = max(1, window // 2)
    smoothed = np.empty_like(arr)
    for i in range(len(arr)):
        start = max(0, i - radius)
        end = min(len(arr), i + radius + 1)
        smoothed[i] = arr[start:end].mean()
    return smoothed


def fetch_histories():
    load_env()
    api_key = os.environ.get("WANDB_KEY") or os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise SystemExit("WANDB_KEY or WANDB_API_KEY is required")
    wandb.login(key=api_key, relogin=True)
    api = wandb.Api(timeout=120)

    histories = {}
    for model, config, run_id, expected_name in RUNS:
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        rows = list(run.scan_history(keys=LOSS_KEYS, page_size=1000))
        normalized = []
        for row in rows:
            out = {"step": row.get("_step")}
            for key in LOSS_KEYS:
                value = finite(row.get(key))
                if key == "loss_kd_bev" and value is not None:
                    out["loss_kd_feature_bev"] = value
                elif value is not None:
                    out[key] = value
            normalized.append(out)
        histories[(model, config)] = {
            "run_id": run_id,
            "run_name": run.name,
            "expected_name": expected_name,
            "rows": normalized,
        }
    return histories


def write_csvs(histories):
    WANDB_DIR.mkdir(exist_ok=True)
    summary_path = WANDB_DIR / "loss_dynamics_summary.csv"
    history_path = WANDB_DIR / "loss_dynamics_history.csv"

    summary_fields = [
        "model",
        "config",
        "run_id",
        "run_name",
        "metric",
        "n",
        "mean",
        "std",
        "min",
        "max",
        "last",
    ]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for (model, config), payload in histories.items():
            rows = payload["rows"]
            for metric in LOSS_KEYS:
                key = "loss_kd_feature_bev" if metric == "loss_kd_bev" else metric
                values = [row.get(key) for row in rows]
                st = stats(values)
                if st["n"] == 0:
                    continue
                writer.writerow(
                    {
                        "model": model,
                        "config": config,
                        "run_id": payload["run_id"],
                        "run_name": payload["run_name"],
                        "metric": key,
                        **st,
                    }
                )

    history_fields = ["model", "config", "run_id", "run_name", "step"] + LOSS_KEYS
    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history_fields)
        writer.writeheader()
        for (model, config), payload in histories.items():
            for row in payload["rows"]:
                record = {
                    "model": model,
                    "config": config,
                    "run_id": payload["run_id"],
                    "run_name": payload["run_name"],
                    "step": row.get("step"),
                }
                for key in LOSS_KEYS:
                    norm_key = "loss_kd_feature_bev" if key == "loss_kd_bev" else key
                    record[key] = row.get(norm_key, "")
                writer.writerow(record)

    return summary_path, history_path


def summary_lookup(histories, model, config, metric):
    rows = histories[(model, config)]["rows"]
    return stats([row.get(metric) for row in rows])


def plot_component_bars(histories, model, path):
    configs = ["S1", "S2", "S3", "S4"]
    x = np.arange(len(configs))
    width = 0.15
    colors = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for i, (label, metric) in enumerate(KD_COMPONENTS):
        vals = []
        for config in configs:
            st = summary_lookup(histories, model, config, metric)
            vals.append(st["mean"] if st["mean"] is not None else 0.0)
        ax.bar(x + (i - 2) * width, vals, width, label=label, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("Rata-rata loss")
    ax.set_xlabel("Konfigurasi ablasi")
    ax.set_title(f"Rata-rata Loss KD per Komponen - {model}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_loss_grad(histories, model, path):
    configs = ["S1", "S2", "S3", "S4"]
    colors = {"S1": "#2563EB", "S2": "#059669", "S3": "#D97706", "S4": "#7C3AED"}
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.2), sharex=True)

    for config in configs:
        rows = histories[(model, config)]["rows"]
        steps = [row["step"] for row in rows if row.get("loss") is not None]
        loss = [row["loss"] for row in rows if row.get("loss") is not None]
        if steps and loss:
            axes[0].plot(steps, rolling_mean(loss), label=config, color=colors[config], linewidth=1.4)
        grad_rows = [row for row in rows if row.get("grad_norm") is not None]
        grad_steps = [row["step"] for row in grad_rows]
        grad = [row["grad_norm"] for row in grad_rows]
        if grad_steps and grad:
            axes[1].plot(grad_steps, rolling_mean(grad), label=config, color=colors[config], linewidth=1.4)

    axes[0].set_ylabel("Total loss")
    axes[0].set_title(f"Dinamika Total Loss - {model}")
    axes[1].set_ylabel("Norma gradien")
    axes[1].set_xlabel("Step pelatihan")
    axes[1].set_title(f"Dinamika Grad Norm - {model}")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(ncol=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_dwa_comparison(histories, path):
    models = ["Student S", "Student XS"]
    x = np.arange(len(WEIGHT_COMPONENTS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for idx, model in enumerate(models):
        means, stds = [], []
        for _, metric in WEIGHT_COMPONENTS:
            st = summary_lookup(histories, model, "S4", metric)
            means.append(st["mean"] if st["mean"] is not None else 0.0)
            stds.append(st["std"] if st["std"] is not None else 0.0)
        ax.bar(
            x + (idx - 0.5) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=model,
            color="#2563EB" if model == "Student S" else "#DC2626",
            alpha=0.85,
        )
    ax.axhline(1.0, color="#111827", linewidth=0.9, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in WEIGHT_COMPONENTS])
    ax.set_ylim(0.85, 1.20)
    ax.set_ylabel("Bobot DWA")
    ax.set_title("Perbandingan Bobot DWA Student S dan Student XS")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_report(histories, summary_path, history_path, figure_paths):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "loss_dynamics_analysis.md"
    lines = [
        "# Loss Dynamics Analysis",
        "",
        f"- Entity/project: `{ENTITY}/{PROJECT}`",
        f"- Summary CSV: `{summary_path.relative_to(ROOT)}`",
        f"- History CSV: `{history_path.relative_to(ROOT)}`",
        "",
        "## Runs",
        "",
        "| Model | Config | Run ID | Run name | Rows |",
        "|---|---|---|---|---:|",
    ]
    for (model, config), payload in histories.items():
        lines.append(
            f"| {model} | {config} | `{payload['run_id']}` | `{payload['run_name']}` | {len(payload['rows'])} |"
        )
    lines += ["", "## Generated Figures", ""]
    for path in figure_paths:
        lines.append(f"- `{path.relative_to(ROOT)}`")

    lines += [
        "",
        "## Key Summary",
        "",
        "| Model | Config | mAP | NDS | Mean loss | Mean grad norm | Mean camera KD | Mean LiDAR KD | Mean BEV KD | Mean relation KD | Mean logit KD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ["Student S", "Student XS"]:
        for config in ["S1", "S2", "S3", "S4"]:
            map_val, nds_val = METRICS[(model, config)]
            vals = []
            for metric in [
                "loss",
                "grad_norm",
                "loss_kd_feature_camera",
                "loss_kd_feature_lidar",
                "loss_kd_feature_bev",
                "loss_kd_rel",
                "loss_kd_logit",
            ]:
                st = summary_lookup(histories, model, config, metric)
                vals.append("-" if st["mean"] is None else f"{st['mean']:.3f}")
            lines.append(f"| {model} | {config} | {map_val:.2f} | {nds_val:.2f} | " + " | ".join(vals) + " |")

    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def main():
    FIG_DIR.mkdir(exist_ok=True)
    histories = fetch_histories()
    summary_path, history_path = write_csvs(histories)

    figure_paths = [
        FIG_DIR / "student_s_kd_loss_components.png",
        FIG_DIR / "student_xs_kd_loss_components.png",
        FIG_DIR / "student_s_loss_grad_dynamics.png",
        FIG_DIR / "student_xs_loss_grad_dynamics.png",
        FIG_DIR / "student_dwa_weight_comparison.png",
    ]
    plot_component_bars(histories, "Student S", figure_paths[0])
    plot_component_bars(histories, "Student XS", figure_paths[1])
    plot_loss_grad(histories, "Student S", figure_paths[2])
    plot_loss_grad(histories, "Student XS", figure_paths[3])
    plot_dwa_comparison(histories, figure_paths[4])

    report_path = write_report(histories, summary_path, history_path, figure_paths)
    print(json.dumps({
        "summary": str(summary_path),
        "history": str(history_path),
        "figures": [str(p) for p in figure_paths],
        "report": str(report_path),
    }, indent=2))


if __name__ == "__main__":
    main()
