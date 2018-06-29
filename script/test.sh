#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
DATASET=pic

# test on pic
python3 models/test_rels.py -m sgdet -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 2 -clip 5 \
    -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-4 -ngpu 1 \
    -ckpt ./checkpoints/picrel-9.tar \
    -dataset ${DATASET}

