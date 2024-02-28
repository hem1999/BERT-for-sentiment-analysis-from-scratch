#!/bin/bash 
#SBATCH --partition=gpuq 
#SBATCH --qos=gpu 
#SBATCH --job-name=gpu_basics 
#SBATCH --output=gpu_basics.%j.out 
#SBATCH --error=gpu_basics.%j.out 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100.80gb:1 
#SBATCH --mem=10G
#SBATCH --export=ALL
#SBATCH --time=0-00:20:00
set echo 
umask 0022 
# to see ID and state of GPUs assigned
nvidia-smi
## Load the necessary modules
module load gnu10
source /home/vduddu/miniconda/bin/activate bert_hw

## Execute your script

python3 /home/vduddu/678_hw1/code/prepare_submit.py /home/vduddu/678_hw1/code/ G01413649

