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
conda activate navigation_env

cd /iris/u/jyang27/dev/omnimimic/scripts
python imitate_episodes.py --policy-class GNM --policy-config ../policy_config/gnmv9_policy.yaml --dataset polybot_dataset gnm_dataset bridge_dataset --val-dataset polybot_dataset gnm_dataset bridge_dataset --seed 30 --no-visualization --wrist-only


#done


