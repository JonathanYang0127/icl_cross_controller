import torch
import numpy as np
import os
os.environ['WANDB_API_KEY'] = '6bc5ed6cff29e874c30c04f4a29faae7ae603964' 
os.environ['WANDB_ENTITY'] = 'jyang27'
import pickle
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm
import yaml
import tensorflow as tf
import tensorflow_datasets as tfds
import rlds
import time
from typing import Any, Dict
from functools import partial

from omnimimic.policies import GCConvPolicy, ACTPolicy
from omnimimic.data_utils.rlds_data_transforms import RLDS_TRANSFORM_DICT

import wandb

import IPython
e = IPython.embed


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(**policy_config)
    elif policy_class == 'ConvNet':
        policy = GCConvPolicy(**policy_config)
    elif policy_class == 'GNM':
        policy = GNMPolicy(**policy_config)
    return policy

def make_optimizer(policy_class, policy):
    return policy.configure_optimizers()

def decode_trajectory(builder, episode: Dict[str, Any]) -> Dict[str, Any]:
    steps = episode['steps']
    for key in steps:
        steps[key] = builder.info.features["steps"][
                        key
                    ].decode_batch_example(steps[key]) 
    return steps

def make_dataloader(dataset_name, split, batch_size, data_dir=None):
    builder = tfds.builder(dataset_name, data_dir=data_dir)
    dataset = builder.as_dataset(
            split=split,
            decoders={"steps": tfds.decode.SkipDecoding()},
            shuffle_files=True,
        )

    dataloader = dataset.map(partial(decode_trajectory, builder),
        num_parallel_calls=tf.data.AUTOTUNE)
    if dataset_name in RLDS_TRANSFORM_DICT.keys():
        dataloader = dataloader.map(RLDS_TRANSFORM_DICT[dataset_name])
    dataloader = dataloader.flat_map(lambda e: tf.data.Dataset.from_tensor_slices(e))

    dataloader = dataloader.repeat(1).shuffle(20000, reshuffle_each_iteration=True)
    dataloader = dataloader.batch(batch_size)
    dataloader = tfds.as_numpy(dataloader)
    return dataloader

def main(args):
    #Get argument values
    ckpt_dir = os.path.join(Path(__file__).parent.parent.absolute(), 'checkpoints')
    policy_class = args['policy_class']
    policy_config_path = args['policy_config']
    #task_name = args['task_name']
    num_epochs = args['num_epochs']
    dataset = args['dataset']
    seed = args['seed']   
 
    exp_name = '{}-{}-omnimimic-seed-{}'.format(time.strftime("%y-%m-%d"), policy_class, seed)

    #Initialize Logger
    logger = wandb.init()

    #Load Policy Config and Policy
    with open(policy_config_path, 'rb') as f:
        policy_config = yaml.safe_load(f)
    policy = make_policy(policy_class, policy_config)
    batch_size_train = policy_config['batch_size']
    batch_size_val = policy_config['batch_size']
    lr = policy_config['lr']

    #Define Dataloaders
    TFDS_DATA_DIR = '/iris/u/jyang27/rlds_data'
    train_dataloader = make_dataloader(dataset, 'train', 
        batch_size_train//2, data_dir=TFDS_DATA_DIR)

    val_dataloader = make_dataloader(dataset, 'train[75%:]',
        batch_size_train, data_dir=TFDS_DATA_DIR)
    
    gnm_dataloader = make_dataloader('gnm_dataset', 'train',
        batch_size_train//2, data_dir=TFDS_DATA_DIR)

    #Define Train Config
    train_config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        #'episode_len': episode_len,
        #'state_dim': state_dim,
        'lr': lr,
        'policy_class': policy_class,
        #'onscreen_render': onscreen_render,
        'policy': policy,
        #'task_name': task_name,
        'exp_name': exp_name,
        'seed': seed,
        #'temporal_agg': args['temporal_agg'],
        #'camera_names': camera_names,
        #'real_robot': not is_sim
    }
    train_bc(train_dataloader, val_dataloader, gnm_dataloader, train_config, logger)

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def train_bc(train_dataloader, val_dataloader, gnm_dataloader, train_config, logger):
    num_epochs = train_config['num_epochs']
    ckpt_dir = train_config['ckpt_dir']
    exp_name = train_config['exp_name']
    seed = train_config['seed']
    
    policy_class = train_config['policy_class']
    policy = train_config['policy']
    policy.cuda()
    policy.train() 
 
    optimizer = make_optimizer(policy_class, policy)

    for epoch in tqdm(range(num_epochs)):
        #gnm_itr = gnm_dataloader.as_numpy_iterator()
        optimizer.zero_grad()
        train_epoch_dicts = []

        for batch_idx, (data, batch) in enumerate(zip(train_dataloader, gnm_dataloader)):
            # batch = next(gnm_itr)
            obs = np.concatenate((batch['observation']['image'], data['observation']['image']), axis=0)
            goal_image = np.concatenate((batch['goal_image'], data['goal_image']), axis=0)
            action = np.concatenate((batch['action'], data['action']), axis=0)
            task_embedding = np.concatenate((np.zeros((len(batch['action']), data['task_embedding'].shape[1])), data['task_embedding']), axis=0)
            #loss_dict = policy.compute_loss_np(data['observation']['image'], 
            #        data['observation']['goal_image'], data['action'], data['task_embedding'])
            loss_dict = policy.compute_loss_np(obs, goal_image, action, task_embedding)
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_epoch_dicts.append(detach_dict(loss_dict))

        train_epoch_summary = compute_dict_mean(train_epoch_dicts)
        train_epoch_summary = {"train/"+key: value for key, value in train_epoch_summary.items()}
        logger.log(train_epoch_summary)
        #train_history.append(train_epoch_summary)
        #epoch_train_loss = train_epoch_summary['loss']

        if epoch % 10 == 0:
            val_epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                loss_dict = policy.compute_loss_np(data['observation']['image'],
                    data['observation']['goal_image'], data['action'], data['task_embedding'])
                loss = loss_dict['loss']
                val_epoch_dicts.append(detach_dict(loss_dict))
            val_epoch_summary = compute_dict_mean(val_epoch_dicts)
            val_epoch_summary = {"val/"+key: value for key, value in val_epoch_summary.items()}
            logger.log(val_epoch_summary)

            if not os.path.exists(os.path.join(ckpt_dir, exp_name)):
                os.makedirs(os.path.join(ckpt_dir, exp_name))
            ckpt_path = os.path.join(ckpt_dir, exp_name, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy-class', type=str, required=True)
    parser.add_argument('--policy-config', type=str, required=True)
    parser.add_argument('--num-epochs',  type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
   
    main(vars(args))

