import torch
import numpy as np
import os
os.environ['WANDB_API_KEY'] = '6bc5ed6cff29e874c30c04f4a29faae7ae603964' 
os.environ['WANDB_ENTITY'] = 'jyang27'
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
import yaml
import tensorflow_datasets as tfds
import rlds
import time

from omnimimic.policies import GCConvPolicy, ACTPolicy

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

def main(args):
    exp_name = '{}-{}-omnimimic'.format(time.strftime("%y-%m-%d"), args['policy_class'])

    #Get argument values
    ckpt_dir = 'checkpoints'
    policy_class = args['policy_class']
    policy_config_path = args['policy_config']
    #task_name = args['task_name']
    num_epochs = args['num_epochs']
    dataset = args['dataset']
    seed = args['seed']   
 
    #Initialize Logger
    logger = wandb.init()

    #Load Policy Config and Policy
    with open(policy_config_path, 'rb') as f:
        policy_config = yaml.safe_load(f)
    policy = make_policy(policy_class, policy_config)
    batch_size_train = policy_config['batch_size']
    batch_size_val = policy_config['batch_size']
    lr = policy_config['lr']

    #Define Dataloader
    dataset = tfds.load(dataset, data_dir=os.environ['TFDS_DATA_DIR'])
    train_dataloader = dataset['train'].flat_map(lambda x: x[rlds.STEPS])
    train_dataloader = train_dataloader.repeat(50).shuffle(50000, reshuffle_each_iteration=True)
    train_dataloader = train_dataloader.batch(batch_size_train)
    train_dataloader = tfds.as_numpy(train_dataloader)
    val_dataloader = None

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
    train_bc(train_dataloader, val_dataloader, train_config, logger)

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

def train_bc(train_dataloader, val_dataloader, train_config, logger):
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
        optimizer.zero_grad()
        train_epoch_dicts = []

        for batch_idx, data in enumerate(train_dataloader):
            loss_dict = policy.compute_loss_np(data['observation']['image'], data['observation']['goal_image'], data['action'])
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_epoch_dicts.append(detach_dict(loss_dict))

        train_epoch_summary = compute_dict_mean(train_epoch_dicts)
        logger.log(train_epoch_summary)
        #train_history.append(train_epoch_summary)
        #epoch_train_loss = train_epoch_summary['loss']

        if epoch % 10 == 0:
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

