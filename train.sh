#!/bin/bash
dataset=stick
model=UNet
loss=log_dkl
samples=4
combine=mixed
legswaps=0.5
armswaps=0.1
output=test

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate pytorch_env

python train.py -d data/$dataset/ --model $model --start 0 --checkpoints 100 --end 6000 --loss $loss --samples $samples --combine $combine --legswaps $legswaps --armswaps $armswaps --output $output