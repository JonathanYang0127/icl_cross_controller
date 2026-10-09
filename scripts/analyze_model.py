import torch
import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial
from controller_transfer.data.rlds_data_transforms import RLDS_TRANSFORM_DICT
from controller_transfer.data.dataset import *
from controller_transfer.data.data_splits import DATASET_SPLITS
from controller_transfer.policies import *
import controller_transfer.torch.pytorch_util as ptu
import yaml
import matplotlib.pyplot as plt

POLICY_CONFIG_PATH = '/iris/u/jyang27/dev/controller_transfer/policy_config/gnmv9_policy_single.yaml'
POLICY_CHECKPOINT = '/iris/u/jyang27/dev/controller_transfer/scripts/checkpoints/23-11-03-GNM-controller_transfer-seed-70_53-23-29/policy_epoch_30_seed_70.ckpt' 
POLICY_CHECKPOINT = "/iris/u/jyang27/dev/controller_transfer/scripts/checkpoints/23-11-08-GNM-controller_transfer-seed-70_48-6-20/policy_epoch_5_seed_70.ckpt" 

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach().cpu()
    return new_d


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k].mean(dim=0)
        result[k] = value_sum / num_items
    return result


def analyze_policy(dataset_name, policy):
    #Load data
    TFDS_DATA_DIR = '/iris/u/jyang27/rlds_data'
    #datasets = list(DATASET_SPLITS.keys())
    #n_datasets = len(datasets)
    datasets = [dataset_name]
    train_dataloaders, val_dataloaders, val_metadata = [], [], []
    config = {'wrist_image_only': True,
              'seq_length': 5,
              'visualize': False,
              'no_normalization': False,
              'image_size': (64, 64)}
    print(datasets)
    batch_size_train = 20
    with tf.device('/cpu'):
        for dataset in datasets:
            #if dataset in val_datasets:
            #    train_split='train[:75%]'
            #else:
            train_split = 'train'
            train_dataloader, _ = make_dataloader(dataset, train_split, config,
                data_dir=TFDS_DATA_DIR)
            train_dataloaders.append(train_dataloader)

        #Interleave train dataloaders
        sample_weights = [DATASET_SPLITS[d] for d in datasets]
        print(datasets, sample_weights)
        sample_weights /= tf.reduce_sum(sample_weights)
        train_dataloader = tf.data.Dataset.sample_from_datasets(
            train_dataloaders, sample_weights)
        train_dataloader = shuffle_batch_and_prefetch_dataloader(train_dataloader,
             batch_size_train, shuffle_size=None)

    train_iterator = train_dataloader.as_numpy_iterator()

    train_epoch_dicts = []
    for batch_idx in tqdm(range(100)):
        batch = next(train_iterator)
        obs = batch['observation']['image']
        goal_image = batch['goal_image']
        action = batch['action']
        goal_distances = batch['goal_distance_indices']
        goal_state = batch.get('goal_state', None)

        loss_dict = policy.compute_loss_np(obs, goal_image, action, goal_distances, task_embedding=None)
        loss = loss_dict['loss']
        loss_dict = detach_dict(loss_dict)
        train_epoch_dicts.append(loss_dict)

    train_epoch_summary = compute_dict_mean(train_epoch_dicts)
    train_epoch_summary = {"train/"+key: value for key, value in train_epoch_summary.items()}
    print("Train Epoch Summary", train_epoch_summary)
    
if __name__ == '__main__':
    ptu.set_gpu_mode(True, gpu_ids=[0])

    #Load Policy Config and Policy
    print("Loading Config")
    with open(POLICY_CONFIG_PATH, 'rb') as f:
        policy_config = yaml.safe_load(f)

    print("Loading Policy")
    policy = make_policy('GNM', policy_config)
    print("Loading Checkpoint")
    if POLICY_CHECKPOINT is not None:
        with open(POLICY_CHECKPOINT, 'rb') as f:
            state_dict = torch.load(f)
            policy.load_state_dict(state_dict)
    policy.to(ptu.device)
    wrapper = get_numpy_wrapper('GNM')
    policy = wrapper(policy)
    print("Done loading policy!")

    for name in DATASET_SPLITS.keys():
        print(f"Statistics for dataset {name}")
        analyze_policy(name, policy)

