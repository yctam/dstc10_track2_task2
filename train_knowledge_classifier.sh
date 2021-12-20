#!/bin/bash

# This script demonstrates how to train knowledge cluster classifier with this repo

# set path to dataset here
version="bert-base-uncased"
dataroot="data"
valset="dstc10val_train50"
testset="dstc10val_dev50"
field="text"
num_label=95
lm_weight=0.1
num_gpus=1

# knowledge cluster classification
python3 baseline.py \
        --params_file src/configs/clusterid_classification/params.${version}.json \
        --last_turn_only \
        --dataroot $dataroot \
        --val_dataset $valset \
        --eval_dataset $testset \
        --field $field \
        --num_labels $num_label \
        --lm_weight $lm_weight \
        --char_confusion pkl/noref/char_confusion.pkl \
        --exp_name kc-${version}-train-lmweight${lm_weight}
