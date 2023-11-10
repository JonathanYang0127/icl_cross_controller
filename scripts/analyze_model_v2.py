import torch
import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial
from omnimimic.data.rlds_data_transforms import RLDS_TRANSFORM_DICT
from omnimimic.data.dataset import *
from omnimimic.data.data_splits import DATASET_SPLITS
from omnimimic.policies import *
import omnimimic.torch.pytorch_util as ptu
import yaml
import matplotlib.pyplot as plt
import glob 
import os
os.environ['WANDB_API_KEY'] = '6bc5ed6cff29e874c30c04f4a29faae7ae603964'
os.environ['WANDB_ENTITY'] = 'jyang27'
os.environ["WANDB__SERVICE_WAIT"] = "300"
import wandb

POLICY_CONFIG_PATH = '/iris/u/jyang27/dev/omnimimic/policy_config/gnmv9_policy.yaml'
POLICY_CHECKPOINT = '/iris/u/jyang27/dev/omnimimic/scripts/checkpoints/23-11-08-GNM-omnimimic-seed-70_48-6-20/*'
POLICY_CHECKPOINT = '/iris/u/jyang27/dev/omnimimic/scripts/checkpoints/23-11-08-GNM-omnimimic-seed-70_3-12-4/*'
POLICY_CHECKPOINT = '/iris/u/jyang27/dev/omnimimic/scripts/checkpoints/23-11-08-GNM-omnimimic-seed-70_26-12-39/*'

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach().cpu()
    return new_d

def compute_dict_mean(epoch_dicts, ignore=['action_pred']):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        skip = False
        for i_key in ignore:
            if i_key in k:
                skip = True
                break
        if skip:
            continue
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k].mean(dim=0)
        result[k] = value_sum / num_items
    return {k: v for k, v in result.items() if v is not None}

def analyze_policy(train_iterator, policy):
    train_epoch_dicts = []
    for batch_idx in tqdm(range(100)):
        batch = next(train_iterator)
        print(batch)
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

    logger = wandb.init(
        name='validation_recompute',
        settings=wandb.Settings(_service_wait=300),
        config={
            'path': POLICY_CONFIG_PATH
        }
    )
    #Load Policy Config and Policy
    print("Loading Config")
    with open(POLICY_CONFIG_PATH, 'rb') as f:
        policy_config = yaml.safe_load(f)

    print("Loading Policy")
    base_policy = make_policy('GNM', policy_config)

    TFDS_DATA_DIR = '/iris/u/jyang27/rlds_data'
    datasets = list(DATASET_SPLITS.keys())
    #n_datasets = len(datasets)
    train_dataloaders, val_dataloaders, val_metadata = [], [], []
    config = {'wrist_image_only': True,
              'seq_length': 1,
              'visualize': False,
              'no_normalization': False,
               'image_size': (64, 64),
               'discrete': True,
               'num_bins': 256}
    batch_size_train = 20
    with tf.device('/cpu'):
        for dataset in datasets:
            train_split = 'train'
            '''
            splits = ['train[:20%]', 'train[20%:40%]', 'train[40%:60%]', 'train[60%:80%]', 'train[80%:]']
            train_dataloader_splits = []
            for split in splits:
                train_split, train_meta = make_dataloader(dataset, split, config,
                    data_dir=TFDS_DATA_DIR)
                train_dataloader_splits.append(train_split)
            train_dataloader = tf.data.Dataset.sample_from_datasets(train_dataloader_splits)
            '''
            train_dataloader, _ = make_dataloader(dataset, train_split, config,
                data_dir=TFDS_DATA_DIR)
            train_dataloaders.append(train_dataloader)
        #Interleave train dataloaders
        sample_weights = [DATASET_SPLITS[d] for d in datasets]
        print(sample_weights)
        sample_weights /= tf.reduce_sum(sample_weights)
        train_dataloader = tf.data.Dataset.sample_from_datasets(
            train_dataloaders, sample_weights)
        train_dataloader = shuffle_batch_and_prefetch_dataloader(train_dataloader,
             batch_size_train, shuffle_size=None)
    
    train_iterator = train_dataloader.as_numpy_iterator()
    paths = glob.glob(POLICY_CHECKPOINT)


    paths = list(sorted(paths, key = lambda p: 
            int(os.path.basename(p).split('_')[2])))
    print(paths)
    for path in paths:
        print(f"Loading Checkpoint {path}")
        if path is not None:
            with open(path, 'rb') as f:
                state_dict = torch.load(f)
                base_policy.load_state_dict(state_dict)
        base_policy.to(ptu.device)
        wrapper = get_numpy_wrapper('GNM')
        policy = wrapper(base_policy)
        train_epoch_dicts = []
        for batch_idx in tqdm(range(100)):
            batch = next(train_iterator)
            print(batch_idx)
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
        checkpoint_name = os.path.basename(path)
        epoch = int(checkpoint_name.split('_')[2])
        logger.log(data=train_epoch_summary, step=epoch)

