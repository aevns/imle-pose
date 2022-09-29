#!/bin/bash
dataset=stick
model=UNet
loss=gaussian
legswaps=0.5
armswaps=0.1
output=s_g_p5p1

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate pytorch_env
python train.py -d data/$dataset/ --model $model --loss $loss --legswaps $legswaps --armswaps $armswaps --output $output