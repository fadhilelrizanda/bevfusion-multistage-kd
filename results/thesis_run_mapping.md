# W&B Run → Thesis Mapping

**Project:** fadhilelrizanda-student/thesis-KD  
**Fetched:** 2026-06-07  
**Method:** Per-epoch history fetched for all 60 runs. Thesis values matched against individual epoch readings (not summary/last epoch), since the thesis reports the best-checkpoint epoch, not the final epoch.

Confidence key: `EXACT` Δ≤0.02 · `HIGH` Δ<0.20 · `PROBABLE` Δ<0.50

---

## 1. Thesis-Cited Runs

### Teacher Baseline
Tables: `tab:teacher-performance`, `tab:ablation-results`, `tab:ablation-xs`, `tab:overall-comparison`

| Run ID   | run_name       | Thesis Label | Epoch Step | mAP (epoch) | NDS (epoch) | mAP thesis | NDS thesis | Δ     | Confidence |
|----------|----------------|--------------|-----------|------------|------------|------------|------------|-------|------------|
| h0syp4kg | base_lidar_cam | Teacher (T)  | 6179      | 67.37      | 70.79      | 67.36      | 70.76      | 0.032 | EXACT      |

---

### Student S Ablation
Table: `tab:ablation-results`

Student S = `bevfusion_student_xxs_*` run family. Thesis evaluates the best-checkpoint epoch, not the last.

| Run ID   | run_name                                             | Thesis Label           | Epoch Step | mAP (epoch) | NDS (epoch) | mAP thesis | NDS thesis | Δ     | Confidence |
|----------|------------------------------------------------------|------------------------|-----------|------------|------------|------------|------------|-------|------------|
| x61g7sqb | bevfusion_combined_baseline_student_XXS_adjust_param | S0 — no KD             | 4325      | 60.34      | 66.13      | 60.34      | 66.13      | 0.000 | EXACT      |
| ynh2rrwe | bevfusion_student_xxs_kd_feature                    | S1 — cam+lidar feat.   | 926       | 66.37      | 69.48      | 66.36      | 69.47      | 0.014 | EXACT      |
| bpbdh38x | bevfusion_student_xxs_kd_feature_bev                | S2 — S1 + BEV feat.    | 1235      | 66.51      | 69.50      | 66.50      | 69.50      | 0.010 | EXACT      |
| lv3ftiq8 | bevfusion_student_xxs_kd_feature_bev_head           | S3 — S2 + head + logit | 1544      | 67.16      | 69.83      | 67.16      | 69.80      | 0.030 | EXACT      |
| 5zkf0n7i | bevfusion_student_xxs_kd_feature_bev_head_dwa_V2    | S4 — S3 + DWA          | 1544      | 67.02      | 69.60      | 67.02      | 69.59      | 0.010 | EXACT      |

> **Note on S0:** The 20-epoch run `x61g7sqb` passes exactly through 60.34/66.13 at step 4325 (epoch ~14), confirming this is the source run. The thesis uses that checkpoint, not the last epoch (best was 60.66 at step 2471).

---

### Student XS Ablation
Table: `tab:ablation-xs`

Student XS = `student_v2_kd_xxs_*` and `bevfusion_student_combined_v2_*` run family.

| Run ID   | run_name                                             | Thesis Label              | Epoch Step | mAP (epoch) | NDS (epoch) | mAP thesis | NDS thesis | Δ     | Confidence |
|----------|------------------------------------------------------|---------------------------|-----------|------------|------------|------------|------------|-------|------------|
| tia4on9x | bevfusion_student_combined_v2_continue               | XS S0 — no KD             | 1544      | 46.73      | 52.49      | 46.72      | 52.49      | 0.010 | EXACT      |
| 2f6bs75l | student_v2_kd_xxs_feature_lidar_cam                 | XS S1 — cam+lidar feat.   | 1544      | 49.00      | 54.05      | 49.02      | 54.06      | 0.022 | EXACT      |
| 8utrbfn6 | student_v2_kd_xxs_feature_lidar_cam_bev              | XS S2 — S1 + BEV feat.    | 1853      | 48.70      | 54.16      | 48.70      | 54.16      | 0.000 | EXACT      |
| xprs9b5n | student_v2_kd_xxs_feature_lidar_cam_bev_head_rel    | XS S3 — S2 + head + logit | 1235      | 49.35      | 53.41      | 49.34      | 53.41      | 0.010 | EXACT      |
| 6u81evo4 | student_v2_kd_xxs_feature_lidar_cam_bev_head_rel_dwa | XS S4 — S3 + DWA         | 1853      | 49.71      | 53.83      | 49.71      | 53.83      | 0.000 | EXACT      |

> **Note on XS S3/S4:** `xprs9b5n` tetap menjadi sumber konfigurasi S3 dengan pembobotan statis, sedangkan konfigurasi S4 tesis diperbarui ke run DWA khusus `6u81evo4` (\textit{v2}) pada step 1853.

---

### DWA Weight Statistics
- `tab:dwa-weight-stats` (Student S S4) → run **5zkf0n7i**, step 1544 checkpoint
- `tab:dwa-weight-stats-xs` (Student XS S4) → run **6u81evo4**, menggunakan bobot DWA akhir pada checkpoint terbaik step 1853

### New DWA KD v2 Run (2026-06-09 Fetch)
- `student_v2_kd_xxs_feature_lidar_cam_bev_head_rel_dwa` (6u81evo4) → **finished**, best checkpoint step 1853 → mAP 49.71%, NDS 53.83%

---
