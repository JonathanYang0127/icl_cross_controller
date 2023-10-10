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
from datetime import datetime
from typing import Any, Dict
from functools import partial

from omnimimic.policies import GCConvPolicy, ACTPolicy, GNMPolicy
from omnimimic.data.dataset import make_dataloader
from omnimimic.train_utils.visualize_utils import (visualize_obs_action,
    visualize_action_dist, visualize_goal_reaching_plots)
import omnimimic.torch.pytorch_util as ptu


from prettytable import PrettyTable
import wandb

import IPython
e = IPython.embed

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    #print(table)
    print(f"Total Trainable Params: {total_params/1e6:.2f}M")
    return total_params

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
    #Get argument values
    ckpt_dir = os.path.join(Path(__file__).parent.parent.absolute(), 'checkpoints')
    policy_class = args['policy_class']
    policy_config_path = args['policy_config']
    #task_name = args['task_name']
    num_epochs = args['num_epochs']
    datasets = args['datasets']
    val_datasets = args['val_datasets']
    seed = args['seed']  
    gpu_ids = args['gpu_ids'] 
    exp_name = '{}-{}-omnimimic-seed-{}'.format(time.strftime("%y-%m-%d"), policy_class, seed)

    #Load Policy Config and Policy
    with open(policy_config_path, 'rb') as f:
        policy_config = yaml.safe_load(f)
    policy = make_policy(policy_class, policy_config)

    if args['checkpoint_path'] is not None:
        with open(args['checkpoint_path'], 'rb') as f:
            policy.load_state_dict(torch.load(f))
    policy.to(ptu.device)
    if len(gpu_ids) > 1:
        policy = torch.nn.DataParallel(policy, device_ids=gpu_ids)

    batch_size_train = policy_config['batch_size']
    batch_size_val = policy_config['batch_size_val']
    print(count_parameters(policy))
    lr = policy_config['lr']

    #Define Dataloaders
    TFDS_DATA_DIR = '/iris/u/jyang27/rlds_data'
    n_datasets = len(datasets)
    batch_size = batch_size_train // n_datasets
    train_dataloaders, val_dataloaders = [], []
    config = {'wrist_image_only': args['wrist_only'], 
              'seq_length': policy_config['len_traj_pred']}
    with tf.device('/cpu'):
        for dataset in datasets:
            if dataset in val_datasets:
                train_split='train[:75%]'
            else:
                train_split = 'train'
            train_dataloader = make_dataloader(dataset, train_split, config, 
                batch_size, data_dir=TFDS_DATA_DIR)
            train_dataloaders.append(train_dataloader)

            if dataset in val_datasets:
                val_dataloader = make_dataloader(dataset, 'train[75%:]', config,
                    batch_size_val, data_dir=TFDS_DATA_DIR)
                val_dataloaders.append(val_dataloader)

    current_time = datetime.now()
    checkpoint_dir = f'{exp_name}_{current_time.minute}-{current_time.hour}-{current_time.second}' 
    checkpoint_dir = os.path.join(ckpt_dir, checkpoint_dir)

    #Define Train Config
    train_config = {
        'num_trains_per_train_loop': 1000,
        'num_vals_per_val_loop': 100,
        'train_datasets': datasets,
        'val_datasets': val_datasets,
        'num_epochs': num_epochs,
        #'episode_len': episode_len,
        #'state_dim': state_dim,
        'lr': lr,
        'policy_class': policy_class,
        #'onscreen_render': onscreen_render,
        'policy': policy,
        #'task_name': task_name,
        'exp_name': exp_name,
        'seed': seed,
        'checkpoint_dir': checkpoint_dir,
        #'temporal_agg': args['temporal_agg'],
        #'camera_names': camera_names,
        #'real_robot': not is_sim
    }

    #Set up logger
    run_name = f'omnimimic_train-{policy_class}-'
    run_name = run_name + '_'.join(datasets)
    logger = wandb.init(
        name=run_name,
        config={'train_config':{key: value for key, value 
                in train_config.items() if key != 'policy'}, 
                'policy_config': policy_config}
    )
    train_bc(train_dataloaders, val_dataloaders, train_config, logger)

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k].mean(dim=0)
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach().cpu()
    return new_d

def numpy_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.numpy()
    return new_d

def train_bc(train_dataloaders, val_dataloaders, train_config, logger):
    num_epochs = train_config['num_epochs']
    checkpoint_dir = train_config['checkpoint_dir']
    exp_name = train_config['exp_name']
    seed = train_config['seed']
    num_trains_per_train_loop = train_config['num_trains_per_train_loop']
    num_vals_per_val_loop = train_config['num_vals_per_val_loop']

    policy_class = train_config['policy_class']
    policy = train_config['policy']
    policy.train() 
 
    optimizer, scheduler = make_optimizer(policy_class, policy)
  
    train_iterators = [d.as_numpy_iterator() for d in train_dataloaders]
    val_iterators = [d.as_numpy_iterator() for d in val_dataloaders]

    for epoch in tqdm(range(num_epochs)):
        #gnm_itr = gnm_dataloader.as_numpy_iterator()
        optimizer.zero_grad()
        train_epoch_dicts = []
        
        for batch_idx in tqdm(range(num_trains_per_train_loop)):
            data = [next(i) for i in train_iterators]
            obs = np.concatenate([batch['observation']['image'] for batch in data], axis=0)
            goal_image = np.concatenate([batch['goal_image'] for batch in data], axis=0)
            action = np.concatenate([batch['action'] for batch in data], axis=0)
            goal_distances = np.concatenate([batch['goal_distance_indices'] 
                for batch in data], axis=0)
            goal_state = np.concatenate([batch['goal_state']
                for batch in data], axis=0)

            #task_embedding = np.concatenate((batch['task_embedding'] for batch in data), axis=0)
            #loss_dict = policy.compute_loss_np(data['observation']['image'], 
            #        data['observation']['goal_image'], data['action'], data['task_embedding'])
            loss_dict = policy.compute_loss_np(obs, goal_image, action, goal_distances, task_embedding=None)
            loss = loss_dict['loss']
            loss.backward()
            #print(torch.cuda.mem_get_info())
            #print(torch.cuda.memory_summary())
            optimizer.step()
            optimizer.zero_grad()

            loss_dict = detach_dict(loss_dict)
            train_epoch_dicts.append(loss_dict)
        
        visualize_dict = numpy_dict(loss_dict)
        visualize_obs_action(obs, goal_image, action, visualize_dict, epoch=epoch)
        visualize_goal_reaching_plots(obs, goal_image, goal_state, action, visualize_dict, epoch=epoch)
        # visualize_action_dist(action, visualize_dict, epoch=epoch)

        if scheduler is not None:
            scheduler.step()
        train_epoch_summary = compute_dict_mean(train_epoch_dicts)
        train_epoch_summary = {"train/"+key: value for key, value in train_epoch_summary.items()}
        logger.log(data=train_epoch_summary, step=epoch)
        #train_history.append(train_epoch_summary)
        #epoch_train_loss = train_epoch_summary['loss']

        if epoch % 5 == 0:
            if val_dataloaders != []:
                val_datasets = train_config['val_datasets']
                val_epoch_dicts = []
                for batch_idx in range(num_vals_per_val_loop):
                    data = [next(i) for i in val_iterators]
                    obs = [batch['observation']['image'] for batch in data]
                    goal_image = [batch['goal_image'] for batch in data]
                    action = [batch['action'] for batch in data]
                    goal_distances = [batch['goal_distance_indices'] for batch in data]
                    goal_state = [batch['goal_state'] for batch in data]

                    epoch_loss_dict = {}
                    for dataloader_idx in range(len(val_dataloaders)):
                        dataset_name = val_datasets[dataloader_idx]
                        loss_dict = policy.compute_loss_np(obs[dataloader_idx], goal_image[dataloader_idx], 
                                action[dataloader_idx], goal_distances[dataloader_idx], task_embedding=None)
                        loss_dict = {key + '/'+ dataset_name: value for key, value in loss_dict.items()}
                        loss_dict = detach_dict(loss_dict)
                        epoch_loss_dict.update(loss_dict)

                        if batch_idx == num_vals_per_val_loop - 1:
                            visualize_dict = numpy_dict(loss_dict)
                            visualize_obs_action(obs[dataloader_idx], goal_image[dataloader_idx], 
                                action[dataloader_idx], visualize_dict, val_key=dataset_name, epoch=epoch)
                            visualize_goal_reaching_plots(obs[dataloader_idx], goal_image[dataloader_idx], goal_state[dataloader_idx], 
                                action[dataloader_idx], visualize_dict, val_key=dataset_name, epoch=epoch)

                            #visualize_action_dist(action[dataloader_idx], visualize_dict, val_key=dataset_name,
                            #        epoch=epoch)
                    val_epoch_dicts.append(epoch_loss_dict)
                    
                val_epoch_summary = compute_dict_mean(val_epoch_dicts)
                val_epoch_summary = {"val/"+key: value for key, value in val_epoch_summary.items()}
                logger.log(data=val_epoch_summary, step=epoch)

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            ckpt_path = os.path.join(checkpoint_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            print(f"Saving checkpoints in {ckpt_path}")
            torch.save(policy.state_dict(), ckpt_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy-class', type=str, required=True)
    parser.add_argument('--policy-config', type=str, required=True)
    parser.add_argument('--num-epochs',  type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    parser.add_argument('--val-datasets', type=str, nargs='+', default=[])
    parser.add_argument('--gpu-ids', type=str, nargs='+', default=['0'])
    parser.add_argument('--checkpoint-path', type=str, default=None)
    parser.add_argument('--wrist-only', action='store_true', default=False)
    args = parser.parse_args()



    print("Num GPUs: {}", torch.cuda.device_count())
    ptu.set_gpu_mode(True, gpu_ids=args.gpu_ids)

    main(vars(args))
