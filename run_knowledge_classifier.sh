#!/bin/bash

version="bert-base-uncased"
dataroot="data"
eval_dataset="dstc10val_dev50"
field="text"
num_label=95
lm_weight=0.1
num_gpus=1
ckpt=runs/kc-${version}-train-lmweight${lm_weight}/checkpoint-18905

# Prepare directories for intermediate results of each subtask
outdir=pred/$eval_dataset
mkdir -p $outdir
python3 baseline.py \
        --eval_only \
        --no_labels \
        --last_turn_only \
        --checkpoint $ckpt \
        --dataroot $dataroot \
        --eval_dataset $eval_dataset \
        --field $field \
        --num_labels $num_label \
        --output_file $outdir/predict.${eval_dataset}.json

echo "$ckpt: Predicted file is in: $outdir/predict.${eval_dataset}.json"
python scripts/scores.py --dataset $eval_dataset --dataroot $dataroot --outfile $outdir/predict.${eval_dataset}.json --scorefile $outdir/score.${eval_dataset}.txt
cat $outdir/score.${eval_dataset}.txt
