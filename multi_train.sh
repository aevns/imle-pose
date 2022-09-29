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

losses=(
   gaussian
   gaussian
   gaussian
   gaussian
   mse
   mse
   mse
   mse
   dkl
   dkl
   dkl
   dkl
)
the_loss=${losses[$SLURM_ARRAY_TASK_ID]}

legswaps=(
   0.1
   0.1
   0.5
   0.5
   0.1
   0.1
   0.5
   0.5
   0.1
   0.1
   0.5
   0.5
)
the_legswaps=${legswaps[$SLURM_ARRAY_TASK_ID]}

armswaps=(
   0.1
   0.5
   0.1
   0.5
   0.1
   0.5
   0.1
   0.5
   0.1
   0.5
   0.1
   0.5
)
the_armswaps=${armswaps[$SLURM_ARRAY_TASK_ID]}

python train.py -d ../scratch/data/$dataset/ --model UNet --loss $the_loss --legswaps $the_legswaps --armswaps $the_armswaps --output stick_$SLURM_ARRAY_TASK_ID