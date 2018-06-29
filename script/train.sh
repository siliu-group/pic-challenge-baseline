#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
DATASET=pic
LOG=log/${DATASET}-train_rels-`date +%Y-%m-%d_%H-%M-%S`.log

# train on pic
# no -use_bias
python3 models/train_rels.py -m sgdet -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 2 -clip 5 \
    -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-4 -ngpu 1 \
    -save_dir checkpoints -nepoch 50 \
    -dataset ${DATASET} \
    2>&1 | tee $LOG


