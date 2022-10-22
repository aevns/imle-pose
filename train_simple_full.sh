#!/bin/bash

#SBATCH --account=def-rhodin
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --array=0-1

### create virtual environment on every node
module load python/3.9.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

### installs all requirements
pip install --no-index --upgrade pip
pip install torch --no-index
pip install --no-index -r requirements.txt

wandb login 9a92298caf7b15ab1719f839763164b8932817a9

### extract dataset
dataset=simple
###cp ../scratch/data/$dataset/train.hdf5 $SLURM_TMPDIR/
###cp ../scratch/data/$dataset/person_keypoints_train.json $SLURM_TMPDIR/

models=(
   UNet
   UNetPretrained
)
the_models=${models[$SLURM_ARRAY_TASK_ID]}
combine=(
   mixed
   mixed
)
the_combine=${combine[$SLURM_ARRAY_TASK_ID]}

python train.py -d data_quick/$dataset/ --model $the_models --start 0 --checkpoints 200 --end 6000 --loss gaussian --combine $the_combine --legswaps 0 --armswaps 0 --output simple_mixture_$SLURM_ARRAY_TASK_ID