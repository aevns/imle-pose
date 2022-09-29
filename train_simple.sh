#!/bin/bash

#SBATCH --account=def-rhodin
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --array=0-11

### create virtual environment on every node
module load python/3.9.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

### installs all requirements
pip install --no-index --upgrade pip
pip install torch --no-index
pip install --no-index -r requirements.txt

### extract dataset
dataset=stick
###cp ../scratch/data/$dataset/train.hdf5 $SLURM_TMPDIR/
###cp ../scratch/data/$dataset/person_keypoints_train.json $SLURM_TMPDIR/

models=(
   UNet
   UNetPretrained
   UNet
   UNetPretrained
   UNet
   UNetPretrained
)
the_models=${models[$SLURM_ARRAY_TASK_ID]}

losses=(
   gaussian
   gaussian
   mse
   mse
   dkl
   dkl
)
the_losses=${losses[$SLURM_ARRAY_TASK_ID]}

python train.py -d ../scratch/data/$dataset/ --model $the_models --loss $the_losses --legswaps 0.5 --armswaps 0.1 --output simple_$SLURM_ARRAY_TASK_ID