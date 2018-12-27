#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1

# test on pic

python3 models/test.py -m sgdet -model stanford -order leftright -nl_obj 2 -nl_edge 4 -b 1 -clip 5 \
    -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-4 -ngpu 1 \
    -ckpt checkpoints/stanford/pic_rel-7.tar

python3 misc/eval.py output/stanford/
