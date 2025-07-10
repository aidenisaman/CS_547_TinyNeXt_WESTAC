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