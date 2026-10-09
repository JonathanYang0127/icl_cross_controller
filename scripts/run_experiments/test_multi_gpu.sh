#!/bin/bash
#SBATCH --partition=savio4_gpu
#SBATCH --account=co_rail
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=400G
#SBATCH --job-name=catjob
#SBATCH --qos=rail_gpu4_normal
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:A5000:8
#SBATCH --job-name="cglossop_omni"
#SBATCH --output=/global/scratch/users/catherineglossop/controller_transfer/outputs/exp-%j.out

source ~/.bashrc
cd /global/home/users/catherineglossop/controller_transfer/scripts
conda activate /global/scratch/users/catherineglossop/miniconda3/omni
# module load cuda/12.2
# export LD_PRELOAD=/global/software/sl-7.x86_64/modules/cuda/11.4/cudnn/8.7.0/lib64/libcudnn.so
# export NCCL_P2P_DISABLE=1
python imitate_episodes.py --policy-class GNM --policy-config ../policy_config/gnmv9_policy.yaml --datasets all --val-datasets all --seed 70 --no-visualization --gpu-ids 0 1

#done