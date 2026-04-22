# Image Classification Implementation

## Setup

```sh
conda create -n tinynext python=3.9
conda activate tinynext
pip install -r requirements.txt
```

## Image Classification for [ImageNet-1K](https://www.image-net.org):

| Model | Top-1 accuracy | #params | MACs | Latency | Logs |
|:--------------:|:----:|:---:|:--:|:--:|:--:|
| TinyNeXt-M |   75.3%   |    2.3M    |   475M   |     19.4ms     |  [M](logs/tinynext_m/rank0.log)    |
| TinyNeXt-S |   72.7%   |    1.3M    |   304M   |     14.3ms     |   [S](logs/tinynext_s/rank0.log)   |
| TinyNeXt-T |   71.5%   |   1.0M   |   259M   |     12.7ms     |   [T](logs/tinynext_t/rank0.log)   |

Latency is measured on Nvidia Jetson Nano.

## Training
### Dataset Preparation

Download the [ImageNet-1K](http://image-net.org/) dataset and structure the data as follows:
```
/path/to/imagenet-1k/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  validation/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```

Train TinyNeXt-M with 8 GPUs in one node: 

```sh
python -m torch.distributed.launch --nproc_per_node=8 --master_port 29501 --use_env main.py --model 'tinynext_m' --data-path '/data/imagenet' --reprob 0.0 --aa="" --mixup 0 --cutmix 0.0
```

Train TinyNeXt-S with 8 GPUs in one node: 

```sh
python -m torch.distributed.launch --nproc_per_node=8 --master_port 29501 --use_env main.py --model 'tinynext_s' --data-path '/data/imagenet' --reprob 0.0 --aa="" --mixup 0 --cutmix 0.0
```


Train TinyNeXt-T with 8 GPUs in one node: 

```sh
python -m torch.distributed.launch --nproc_per_node=8 --master_port 29501 --use_env main.py --model 'tinynext_t' --data-path '/data/imagenet' --reprob 0.0 --aa="" --mixup 0 --cutmix 0.0
```

## Experiment 2: mini-ImageNet Dataset 


### mini-ImageNet (CSV + flat image files) on CPU
Mini-ImageNet dataset used for this experiment:

- https://www.kaggle.com/datasets/zcyzhchyu/mini-imagenet/data

Expected dataset layout:

```
<DATASET_ROOT>/
  train.csv
  val.csv
  test.csv
  <image files referenced by filename in the CSVs>
```

If your mini-ImageNet dataset is provided as `train.csv` / `val.csv` with `filename,label` columns and images in a flat directory, use:

```powershell
cd classification
.\run_experiment2_mini_imagenet_cpu.ps1 -RepoRoot "<REPO_ROOT>" -ImageRoot "<DATASET_ROOT>" -TrainCsv "<DATASET_ROOT>/train.csv" -ValCsv "<DATASET_ROOT>/val.csv" -TestCsv "<DATASET_ROOT>/test.csv"
```

Default behavior is:
- Merge `train.csv`, `val.csv`, and `test.csv`, then create a stratified train/val split (default train ratio `0.8`) with shared 100 classes.
- Prepare `Data/mini_imagenet_100_folder` using hard links.
- Run smoke baseline + smoke tuned.
- Skip strict class-count validation unless `-ExpectedClasses` is provided.

For canonical merged mini-ImageNet setup, enforce class count with:

```powershell
.\run_experiment2_mini_imagenet_cpu.ps1 -RepoRoot "<REPO_ROOT>" -ImageRoot "<DATASET_ROOT>" -TrainCsv "<DATASET_ROOT>/train.csv" -ValCsv "<DATASET_ROOT>/val.csv" -TestCsv "<DATASET_ROOT>/test.csv" -ExpectedClasses 100
```

Run smoke + full runs:

```powershell
.\run_experiment2_mini_imagenet_cpu.ps1 -RepoRoot "<REPO_ROOT>" -ImageRoot "<DATASET_ROOT>" -TrainCsv "<DATASET_ROOT>/train.csv" -ValCsv "<DATASET_ROOT>/val.csv" -TestCsv "<DATASET_ROOT>/test.csv" -RunFull
```

Prepare only (no training):

```powershell
.\run_experiment2_mini_imagenet_cpu.ps1 -RepoRoot "<REPO_ROOT>" -ImageRoot "<DATASET_ROOT>" -TrainCsv "<DATASET_ROOT>/train.csv" -ValCsv "<DATASET_ROOT>/val.csv" -TestCsv "<DATASET_ROOT>/test.csv" -PrepareOnly
```

Force rebuilding prepared folders:

```powershell
.\run_experiment2_mini_imagenet_cpu.ps1 -RepoRoot "<REPO_ROOT>" -ImageRoot "<DATASET_ROOT>" -TrainCsv "<DATASET_ROOT>/train.csv" -ValCsv "<DATASET_ROOT>/val.csv" -TestCsv "<DATASET_ROOT>/test.csv" -ForceRebuild
```

If `python` is not on PATH, pass an explicit interpreter:

```powershell
.\run_experiment2_mini_imagenet_cpu.ps1 -RepoRoot "<REPO_ROOT>" -ImageRoot "<DATASET_ROOT>" -TrainCsv "<DATASET_ROOT>/train.csv" -ValCsv "<DATASET_ROOT>/val.csv" -TestCsv "<DATASET_ROOT>/test.csv" -PythonExe "<PYTHON_EXE_PATH>"
```

Log layout:
- `logs_exp2_mini_imagenet_cpu/smoke/<baseline|tuned>/<model>/<timestamp>/rank0.log`
- `logs_exp2_mini_imagenet_cpu/full/<baseline|tuned>/<model>/<timestamp>/rank0.log` (when `-RunFull` is used)

Run only one preset:

```powershell
.\run_experiment2_mini_imagenet_cpu.ps1 -RepoRoot "<REPO_ROOT>" -ImageRoot "<DATASET_ROOT>" -TrainCsv "<DATASET_ROOT>/train.csv" -ValCsv "<DATASET_ROOT>/val.csv" -TestCsv "<DATASET_ROOT>/test.csv" -RunBaseline -SmokeOnly
.\run_experiment2_mini_imagenet_cpu.ps1 -RepoRoot "<REPO_ROOT>" -ImageRoot "<DATASET_ROOT>" -TrainCsv "<DATASET_ROOT>/train.csv" -ValCsv "<DATASET_ROOT>/val.csv" -TestCsv "<DATASET_ROOT>/test.csv" -RunTuned -SmokeOnly
```

Equivalent baseline command (single run):

```sh
python main.py --model tinynext_t --data-set FOLDER --data-path /path/to/custom100 --train-split train --val-split val --num-classes 100 --input-size 224 --epochs 100 --batch-size 128 --reprob 0.0 --aa "" --mixup 0.0 --cutmix 0.0
```

Equivalent tuned command (single run):

```sh
python main.py --model tinynext_t --data-set FOLDER --data-path /path/to/custom100 --train-split train --val-split val --num-classes 100 --input-size 224 --epochs 100 --batch-size 128 --reprob 0.1 --aa rand-m9-mstd0.5-inc1 --mixup 0.2 --cutmix 0.2 --weight-decay 0.05 --lr 0.004 --warmup-epochs 10
```

Notes:
- `--num-classes 100` is validated against folder labels to avoid silent mismatch.
- You can set custom normalization with `--custom-mean r g b --custom-std r g b` if your dataset statistics differ significantly from ImageNet/CIFAR.
- Compare your resulting `top1/top5` against the ImageNet-1K baseline numbers in this README as a reference point, but these are cross-dataset comparisons and not directly apples-to-apples.

## Evaluation

<details>
<summary>
TinyNeXt-M
</summary>

Test  with 8 GPUs in one node:

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port 29501 --use_env \
main.py --model 'tinynext_m' --data-path '/data/imagenet' --eval --resume "logs/tinynext_m/tinynext_m.pth"
```

This should give `* eval  loss: 1.0679	top1: 75.28	top5: 92.24` 

<details>
<summary>
TinyNeXt-S
</summary>
Test  with 8 GPUs in one node:

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port 29501 --use_env \
main.py --model 'tinynext_s' --data-path '/data/imagenet' --eval --resume "logs/tinynext_s/tinynext_s.pth"
```

This should give `* eval  loss: 1.1817	top1: 72.70	top5: 90.93` 

<details>
<summary>
TinyNeXt-T
</summary>
Test  with 8 GPUs in one node:

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port 29501 --use_env \
main.py --model 'tinynext_t' --data-path '/data/imagenet' --eval --resume "logs/tinynext_t/tinynext_t.pth"
```

This should give `* eval  loss: 1.2419	top1: 71.54	top5: 90.24` 