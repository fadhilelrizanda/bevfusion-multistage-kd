import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "analysis"
FIGURES_DIR.mkdir(exist_ok=True)

# Read CSV
df = pd.read_csv(RESULTS_DIR / 'thesis_run_metrics.csv')

classes = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
class_labels = ['Car', 'Truck', 'Construction vehicle', 'Bus', 'Trailer', 'Barrier', 'Motorcycle', 'Bicycle', 'Pedestrian', 'Traffic cone']

# Target rows
# Teacher: thesis_label == 'Teacher (T)'
# Student S (S3): thesis_label == 'Student S — S3'
# Student XS uses S2 for "best" comparisons because NDS is the selection metric.

teacher = df[df['thesis_label'] == 'Teacher (T)'].iloc[0]
student_s_s3 = df[df['thesis_label'] == 'Student S — S3'].iloc[0]
student_xs_s2 = df[df['thesis_label'] == 'Student XS — S2'].iloc[0]
student_xs_s4_candidates = df[
    df['thesis_label'].isin(['Student XS — S4 DWA v2', 'Student XS — S4'])
]
student_xs_s4 = student_xs_s4_candidates.sort_values(
    by='thesis_label',
    key=lambda s: s.eq('Student XS — S4 DWA v2').astype(int),
    ascending=False,
).iloc[0]

def get_class_ap(row, cls):
    cols = [f"{cls}_AP_dist_0.5", f"{cls}_AP_dist_1.0", f"{cls}_AP_dist_2.0", f"{cls}_AP_dist_4.0"]
    return row[cols].mean() * 100

print("LATEX ROWS:")
teacher_aps = []
student_s_aps = []
student_xs_aps = []

for cls, label in zip(classes, class_labels):
    ap_t = get_class_ap(teacher, cls)
    ap_s = get_class_ap(student_s_s3, cls)
    ap_xs = get_class_ap(student_xs_s2, cls)
    
    teacher_aps.append(ap_t)
    student_s_aps.append(ap_s)
    student_xs_aps.append(ap_xs)
    
    print(f"\\textit{{{label}}} & {ap_t:.2f} & {ap_s:.2f} & {ap_xs:.2f} \\\\")

# Plotting AP per class
x = np.arange(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width, teacher_aps, width, label='Teacher', color='#4A90E2')
rects2 = ax.bar(x, student_s_aps, width, label='Student S (S3)', color='#50E3C2')
rects3 = ax.bar(x + width, student_xs_aps, width, label='Student XS (S2)', color='#F5A623')

ax.set_ylabel('Average Precision (AP) %', fontsize=12)
ax.set_xlabel('Object class', fontsize=12)
ax.set_title('Per-class AP comparison', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(class_labels, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'ap_per_class.png', dpi=300)
plt.close()

# Also let's plot mAP and NDS for the configurations
# Configurations for Student S: S0 to S4, and T
s_labels = ['Teacher', 'S0', 'S1', 'S2', 'S3', 'S4']
s_rows = [teacher] + [df[df['thesis_label'] == f'Student S — {lbl}'].iloc[0] for lbl in ['S0', 'S1', 'S2', 'S3', 'S4']]
s_map = [row['mAP']*100 for row in s_rows]
s_nds = [row['NDS']*100 for row in s_rows]

x_s = np.arange(len(s_labels))
width_2 = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x_s - width_2/2, s_map, width_2, label='mAP', color='#4A90E2')
rects2 = ax.bar(x_s + width_2/2, s_nds, width_2, label='NDS', color='#F5A623')
ax.set_ylabel('Skor (%)', fontsize=12)
ax.set_xlabel('Ablation configuration', fontsize=12)
ax.set_title('Knowledge-distillation evaluation for Student S', fontsize=14)
ax.set_xticks(x_s)
ax.set_xticklabels(s_labels)
ax.legend()
ax.set_ylim(55, 75)
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'student_s_ablation.png', dpi=300)
plt.close()

xs_labels = ['Teacher', 'S0', 'S1', 'S2', 'S3', 'S4']
xs_rows = [teacher] + [df[df['thesis_label'] == f'Student XS — {lbl}'].iloc[0] for lbl in ['S0', 'S1', 'S2', 'S3']]
xs_rows.append(student_xs_s4)
xs_map = [row['mAP']*100 for row in xs_rows]
xs_nds = [row['NDS']*100 for row in xs_rows]

x_xs = np.arange(len(xs_labels))
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x_xs - width_2/2, xs_map, width_2, label='mAP', color='#4A90E2')
rects2 = ax.bar(x_xs + width_2/2, xs_nds, width_2, label='NDS', color='#F5A623')
ax.set_ylabel('Skor (%)', fontsize=12)
ax.set_xlabel('Ablation configuration', fontsize=12)
ax.set_title('Knowledge-distillation evaluation for Student XS', fontsize=14)
ax.set_xticks(x_xs)
ax.set_xticklabels(xs_labels)
ax.legend()
ax.set_ylim(40, 75)
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'student_xs_ablation.png', dpi=300)
plt.close()

# Final DWA component weights at the best Student XS DWA v2 checkpoint.
dwa_labels = ['LiDAR', 'Camera', 'BEV', 'Relation', 'Logit']
dwa_values = [1.0127, 1.0181, 0.9987, 0.9424, 1.0281]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(dwa_labels, dwa_values, color=['#4A90E2', '#50E3C2', '#7ED321', '#F5A623', '#D0021B'])
ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='Bobot statis = 1.0')
ax.set_ylabel('Bobot DWA', fontsize=12)
ax.set_xlabel('Loss component', fontsize=12)
ax.set_title('Final DWA weights for Student XS S4')
ax.set_ylim(0.85, 1.08)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend()

for bar, val in zip(bars, dwa_values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f'{val:.3f}',
            ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'student_xs_dwa_weights.png', dpi=300)
plt.close()

print("Graphs generated in figures/")
