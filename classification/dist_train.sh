#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port 29501 --use_env \
main.py --model 'tinynext_m' --data-path '/data/imagenet' --reprob 0.0 --aa="" --mixup 0 --cutmix 0.0
python -m torch.distributed.launch --nproc_per_node=8 --master_port 29501 --use_env \
main.py --model 'tinynext_s' --data-path '/data/imagenet' --reprob 0.0 --aa="" --mixup 0 --cutmix 0.0
python -m torch.distributed.launch --nproc_per_node=8 --master_port 29501 --use_env \
main.py --model 'tinynext_t' --data-path '/data/imagenet' --reprob 0.0 --aa="" --mixup 0 --cutmix 0.0