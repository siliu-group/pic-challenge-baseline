#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1
DATASET=pic
LOG=log/${DATASET}-train_stanford_p4-`date +%Y-%m-%d_%H-%M-%S`.log

python models/train.py -m sgcls -model stanford -b 1 -p 100 -lr 1e-3 -ngpu 1 -clip 5 \
    -save_dir checkpoints/stanford -nepoch 50 \
    2>&1 | tee $LOG