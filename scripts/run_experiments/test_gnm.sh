#!/bin/bash
#SBATCH --partition=iris
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

cd /iris/u/jyang27/dev/controller_transfer/scripts
python imitate_episodes.py --policy-class GNM --policy-config ../policy_config/gnmv9_policy.yaml --dataset gnm_dataset --val-dataset gnm_dataset --seed 200 --checkpoint-path checkpoints/23-10-10-GNM-controller_transfer-seed-50_6-22-33/policy_epoch_55_seed_50.ckpt --checkpoint-path ../model_weights/gnm_no_normalize/latest_state_dict.pth --test


#done


