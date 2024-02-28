#!/bin/bash 
#SBATCH --partition=gpuq 
#SBATCH --qos=gpu 
#SBATCH --job-name=gpu_basics 
#SBATCH --output=gpu_basics.%j.out 
#SBATCH --error=gpu_basics.%j.out 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100.80gb:1 
#SBATCH --mem=20G
#SBATCH --export=ALL
#SBATCH --time=0-02:00:00
set echo 
umask 0022 
# to see ID and state of GPUs assigned
nvidia-smi
## Load the necessary modules
module load gnu10
source /home/vduddu/miniconda/bin/activate bert_hw

## Execute your script
python /home/vduddu/678_hw1/classifier.py --option finetune --epochs 20 --lr 5e-5 --train /home/vduddu/678_hw1/data/cfimdb-train.txt --dev /home/vduddu/678_hw1/data/cfimdb-dev.txt --test /home/vduddu/678_hw1/data/cfimdb-test.txt --dev_out cfimdb-dev-output.finetune.txt --test_out cfimdb-test-output.finetune.txt --batch_size 8 --hidden_dropout_prob 0.2 --use_gpu

python /home/vduddu/678_hw1/classifier.py --option pretrain --epochs 20 --lr 5e-5 --train /home/vduddu/678_hw1/data/cfimdb-train.txt --dev /home/vduddu/678_hw1/data/cfimdb-dev.txt --test /home/vduddu/678_hw1/data/cfimdb-test.txt --dev_out cfimdb-dev-output.pretrain.txt --test_out cfimdb-test-output.pretrain.txt --batch_size 8 --hidden_dropout_prob 0.2 --use_gpu

