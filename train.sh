#!/bin/bash
dataset=simple
model=UNet
loss=gaussian
combine=mixed
legswaps=0
armswaps=0
output=test

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate pytorch_env

python train.py -d data/$dataset/ --model $model --start 0 --checkpoints 100 --end 6000 --loss $loss --combine $combine --legswaps $legswaps --armswaps $armswaps --output $output