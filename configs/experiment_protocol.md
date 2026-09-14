# Recorded experiment protocol

The teacher was trained independently and frozen. Student S and Student XS were first trained without knowledge distillation for 20 epochs. The best validation checkpoint then initialized a six-epoch knowledge-distillation continuation. Consequently, changes from S0 to S4 are associations under a continued-training, single-run design, not isolated causal estimates.

| Item | Recorded setting |
| --- | --- |
| Dataset | nuScenes official split; validation results reported |
| Images | 256 x 704 pixels |
| LiDAR | Nine sweeps; horizontal range [-54, 54] m; vertical range [-5, 3] m |
| Voxel size | (0.075, 0.075, 0.2) m |
| Optimizer | AdamW with automatic mixed precision |
| Batch size | 4 per GPU; 8 total on two RTX 3090 GPUs |
| Gradient clipping | L2 norm 35 |
| S0 learning rate | 1e-4 |
| KD continuation learning rate | 2e-4 |
| Schedule | 500-iteration linear warm-up followed by cosine annealing |
| Static KD weights | 1 for every KD component in S1--S3 |
| DWA | Five KD components; first two epochs use weight 1; temperature 2 |

Inference measurements used batch size one after model/engine initialization. RTX 3090 TensorRT and NVIDIA Drive Orin measurements used FP16. Timings cover isolated forward inference with device synchronization; recorded warm-up and repetition counts are unavailable. Sensor I/O, preprocessing, postprocessing, planning, and resource contention are excluded.
