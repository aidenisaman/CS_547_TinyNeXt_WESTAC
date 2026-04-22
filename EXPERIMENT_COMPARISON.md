# Experiment Comparison: TinyNeXt-T (ImageNet-1K / CLS-LOC)

Author: Aiden West

School: SUNY POLY

Class: CS547
## Goal

Reproduce one official classification evaluation experiment using the included checkpoint and the ImageNet-style dataset described in the README, without retraining.

## Setup Used

- Model: `tinynext_t`
- Checkpoint: `classification/logs/tinynext_t/tinynext_t.pth`
- Dataset root used for eval: `Data/CLS-LOC_eval`
- Dataset size seen by run:
	- Train: 1,281,167 images
	- Val: 50,000 images
- Device: CPU
- Eval command:

```bash
python main.py --eval --model tinynext_t --data-set IMNET --data-path ../Data/CLS-LOC_eval --resume logs/tinynext_t/tinynext_t.pth --batch-size 128 --num_workers 4 --device cpu
```

## Results

From `classification/logs/tinynext_t/eval/rank0.log` (timestamp `2026-04-14 16:07:09`):

- `loss: 1.2420`
- `top1: 71.56`
- `top5: 90.23`

## Comparison Against README Expected

README expected for TinyNeXt-T:

- `loss: 1.2419`
- `top1: 71.54`
- `top5: 90.24`

| Metric | README Expected | This Run | Delta (This - Expected) |
|---|---:|---:|---:|
| Loss | 1.2419 | 1.2420 | +0.0001 |
| Top-1 | 71.54 | 71.56 | +0.02 |
| Top-5 | 90.24 | 90.23 | -0.01 |

## Conclusion

This reproduction matches the official TinyNeXt-T ImageNet evaluation very closely (all deltas are negligible), confirming the checkpoint and dataset pipeline are functioning correctly for eval-only testing. The difference isn't grand enough to have a reasoning for why there is a differnce

---

## Experiment 2: mini-imagenet 100-Class Dataset + Light Tuning

## Goal

Run TinyNeXt-T on the  dataset using two settings:

- Baseline-style (close to author training defaults)
- Light tuned setting (small regularization and optimization adjustments)

Then compare:

- Experiment 2 Baseline vs Experiment 2 Tuned
- Experiment 2 vs author ImageNet-1K result (reference only)

## Command (Windows)

```powershell
cd classification
.\run_experiment2_mini_imagenet_cpu.ps1 -RepoRoot "F:/CS_547_TinyNeXt_WESTAC"
```

Smoke + full:

```powershell
.\run_experiment2_mini_imagenet_cpu.ps1 -RepoRoot "F:/CS_547_TinyNeXt_WESTAC" -RunFull
```

Prepared data root (default):
- `Data/mini_imagenet_100_folder`

Run logs:
- `classification/logs_exp2_mini_imagenet_cpu/smoke/baseline/.../rank0.log`
- `classification/logs_exp2_mini_imagenet_cpu/smoke/tuned/.../rank0.log`
- `classification/logs_exp2_mini_imagenet_cpu/full/baseline/.../rank0.log`
- `classification/logs_exp2_mini_imagenet_cpu/full/tuned/.../rank0.log`
- active retry run: `classification/logs_exp2_mini_imagenet_cpu_retry/.../rank0.log`

## Experiment 2 Results

Fill from each run directory `rank0.log` (`* eval  loss: ... top1: ... top5: ...`).

| Run | Dataset | Epochs | Top-1 | Top-5 | Loss | Notes |
|---|---|---:|---:|---:|---:|---|
| Exp2 Baseline Smoke | mini-ImageNet-100 | 2 | 2.78 | 11.31 | 4.5790 | `reprob=0, mixup=0, cutmix=0` |
| Exp2 Tuned Smoke | mini-ImageNet-100 | 2 | 2.58 | 10.74 | 4.5810 | completed, `reprob=0.1, mixup=0.2, cutmix=0.2, wd=0.05, lr=0.004` |
| Exp2 Baseline Full | mini-ImageNet-100 | 20 | 35.35 | 66.22 | 3.2723 | completed |
| Exp2 Tuned Full | mini-ImageNet-100 | 20 | 23.75 | 53.10 | 3.8790 | completed |

## Comparison Table

Author TinyNeXt-T ImageNet number is included as a reference anchor.

| Model/Run | Dataset | Top-1 | Top-5 | Delta vs Author Top-1 |
|---|---|---:|---:|---:|
| Author TinyNeXt-T (README) | ImageNet-1K | 71.54 | 90.24 | 0.00 |
| Exp1 Reproduction (this repo) | ImageNet-1K / CLS-LOC | 71.56 | 90.23 | +0.02 |
| Exp2 Baseline Smoke | mini-ImageNet-100 | 2.78 | 11.31 | -68.76 |
| Exp2 Tuned Smoke | mini-ImageNet-100 | 2.58 | 10.74 | -68.96 |
| Exp2 Baseline Full | mini-ImageNet-100 | 35.35 | 66.22 | -36.19 |
| Exp2 Tuned Full | mini-ImageNet-100 | 23.75 | 53.10 | -47.79 |

## Delta Summary (Exp2 Tuned Full - Exp2 Baseline Full)

| Metric | Baseline Full | Tuned Full | Delta |
|---|---:|---:|---:|
| Top-1 | 35.35 | 23.75 | -11.60 |
| Top-5 | 66.22 | 53.10 | -13.12 |
| Loss | 3.2723 | 3.8790 | +0.6067 |

Interpretation:
- The tuned setting underperformed baseline on this mini-ImageNet-100 setup.
- Top-1 dropped by 11.60 points and Top-5 dropped by 13.12 points.
- Loss increased by 0.6067, indicating worse validation fit.

## Fairness Notes

The comparison to the author's TinyNeXt-T ImageNet-1K number should be treated as a reference anchor, not a strict apples-to-apples benchmark.

Why this is not a fully fair direct comparison:

- Different dataset/task regime:
	- Author number: ImageNet-1K (1000 classes).
	- Experiment 2: mini-ImageNet-100 (100 classes) with a custom merged-and-resplit train/val pipeline.
- Different training budget:
	- Author-style training config in this repo defaults to a long schedule (`epochs=300` in `classification/config.py`).
	- Experiment 2 full runs used only 20 epochs.
- Different compute scale:
	- Author training commands are multi-GPU distributed.
	- Experiment 2 here was CPU-only with batch size 32.
- Different objective context:
	- Exp1 was checkpoint evaluation (`--resume`), while Exp2 was fresh training on a different dataset.

Fair comparisons in this report are:

- Exp1 reproduction vs author README on ImageNet-1K eval (same checkpoint-style objective).
- Exp2 Baseline vs Exp2 Tuned on the same mini-ImageNet split.

## Why the Exp2 Gap Is Large

The large drop in tuned full performance versus baseline full is consistent with underfitting from the tuned recipe under a short 20-epoch budget.

Observed outcome:

- Baseline full: Top-1 35.35, Top-5 66.22, Loss 3.2723.
- Tuned full: Top-1 23.75, Top-5 53.10, Loss 3.8790.

Likely causes (from `classification/run_experiment2_mini_imagenet_cpu.ps1` and defaults in `classification/config.py`):

1. Too much warmup for a short run:
	 - Tuned used `--warmup-epochs 10` out of only 20 total epochs.
	 - Half of training is spent in warmup, which slows effective optimization progress.

2. Lower LR and stronger weight decay than baseline defaults:
	 - Baseline inherits defaults (`lr=0.006`, `weight-decay=0.025`).
	 - Tuned used `lr=0.004` and `weight-decay=0.05`.
	 - This combination is more conservative and can suppress fitting under limited epochs.

3. Additional regularization pressure:
	 - Tuned added `mixup=0.2`, `cutmix=0.2`, `reprob=0.1`.
	 - With small epoch budget and from-scratch training, this can reduce early convergence speed.

4. CPU-only, small-batch setting magnifies schedule sensitivity:
	 - With batch size 32 and slower throughput, hyperparameters tuned for larger-scale runs often need re-tuning.

Overall interpretation:

- The tuned setup was not inherently worse in principle; it was mismatched to this short, CPU-constrained training regime.
- In this context, the baseline configuration was better aligned and therefore achieved higher final accuracy.
