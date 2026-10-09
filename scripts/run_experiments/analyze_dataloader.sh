#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=100G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1

#SBATCH --job-name="Jonathan_Exp"
#SBATCH --output=/iris/u/jyang27/cluster/outputs/exp-%j.out
source ~/.bashrc
conda deactivate 
#conda activate iris_env
conda activate navigation_env

cd /iris/u/jyang27/dev/icl_controller_transfer/scripts
python analyze_dataloader.py
#python analyze_model.py

#done


