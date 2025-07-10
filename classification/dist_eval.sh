#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port 29501 --use_env \
main.py --model 'tinynext_m' --data-path '/data/imagenet' --eval --resume "logs/tinynext_m/tinynext_m.pth"
python -m torch.distributed.launch --nproc_per_node=8 --master_port 29501 --use_env \
main.py --model 'tinynext_s' --data-path '/data/imagenet' --eval --resume "logs/tinynext_s/tinynext_s.pth"
python -m torch.distributed.launch --nproc_per_node=8 --master_port 29501 --use_env \
main.py --model 'tinynext_t' --data-path '/data/imagenet' --eval --resume "logs/tinynext_t/tinynext_t.pth"