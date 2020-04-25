#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2  python main.py --batch_size 4 --test_batch_size 4 --method baseline --novel_num 1 --log_action d &&
CUDA_VISIBLE_DEVICES=2  python main.py --batch_size 4 --test_batch_size 4 --method va --novel_num 1 --log_action d &&
CUDA_VISIBLE_DEVICES=2  python main.py --batch_size 4 --test_batch_size 4 --method vasa --novel_num 1 --log_action d


python main.py --gpu 2 --batch_size 4 --test_batch_size 4 --method baseline --novel_num 1 --log_action d &&
python main.py --gpu 2 --batch_size 4 --test_batch_size 4 --method va --novel_num 1 --log_action d &&
python main.py --gpu 2 --batch_size 4 --test_batch_size 4 --method vasa --novel_num 1 --log_action d


python main.py --gpu 0 --method baseline --novel_num 5 --log_action d &&
python main.py --gpu 1 --method va --novel_num 5 --log_action d &&
python main.py --gpu 2  --method vasa --novel_num 5 --log_action d


CUDA_VISIBLE_DEVICES=0 python main.py  --meta_split 150_10_20_unseen20 --method baseline2d
CUDA_VISIBLE_DEVICES=1 python main.py  --meta_split 150_10_20_unseen20 --method moco_va
CUDA_VISIBLE_DEVICES=2 python main.py  --meta_split 150_10_20_unseen20 --method moco_vasa

CUDA_VISIBLE_DEVICES=3 python main.py  --meta_split 150_10_20_unseen20 --method moco_vasaaa


