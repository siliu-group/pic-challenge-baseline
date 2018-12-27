#!/usr/bin/env bash

# This is a script that will train all of the models for scene graph classification and then evaluate them.
export CUDA_VISIBLE_DEVICES=$1
DATASET=pic
LOG=log/${DATASET}-train_motif_p4-`date +%Y-%m-%d_%H-%M-%S`.log

# train on pic
python3 models/train.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 1 -clip 5 \
    -p 100 -hidden_dim 512 -lr 1e-3 -ngpu 1 -use_bias \
    -save_dir checkpoints/motif -nepoch 50 \
    2>&1 | tee $LOG



