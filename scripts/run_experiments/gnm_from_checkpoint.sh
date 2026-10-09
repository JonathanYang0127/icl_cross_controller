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
conda activate iris_env

cd /iris/u/jyang27/dev/icl_controller_transfer/scripts
python imitate_episodes.py --policy-class GNM --policy-config ../policy_config/gnmv9_policy.yaml --datasets gnm_dataset --val-datasets gnm_dataset --seed 50 --checkpoint-path /iris/u/jyang27/dev/icl_controller_transfer/scripts/checkpoints/23-10-27-GNM-icl_controller_transfer-seed-50_5-12-37/policy_epoch_100_seed_50.ckpt 


#done


