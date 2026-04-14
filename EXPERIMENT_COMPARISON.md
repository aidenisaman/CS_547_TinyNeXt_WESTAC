# Experiment Comparison: TinyNeXt-T (ImageNet-1K / CLS-LOC)

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
