#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=150G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:2

#SBATCH --job-name="Jonathan_Exp"
#SBATCH --output=/iris/u/jyang27/cluster/outputs/exp-%j.out
source ~/.bashrc
conda deactivate 
conda activate navigation_env

cd /iris/u/jyang27/dev/controller_transfer/scripts
python imitate_episodes.py --policy-class GNM --policy-config ../policy_config/gnmv9_policy_large.yaml --datasets all --val-datasets all --seed 70 --no-visualization  --gpu-ids 0 1

#done


