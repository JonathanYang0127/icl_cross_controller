import numpy as np
import torch

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
from icl_controller_transfer.policies import GCConvPolicy, ACTPolicy, GNMPolicy
from icl_controller_transfer.policies.convnet_policy import GaussianCNNPolicy, CNN

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(**policy_config)
    elif policy_class == 'ConvNet':
        policy = GCConvPolicy(**policy_config)
    elif policy_class == 'GNM':
        policy = GNMPolicy(**policy_config)
    return policy

def make_dataloader(dataset_name, split, batch_size, data_dir=None, shuffle_size=10000):
    builder = tfds.builder(dataset_name, data_dir=data_dir)
    dataset = builder.as_dataset(
            split=split,
            shuffle_files=True,
        )

    #dataloader = dataset.map(partial(decode_trajectory, builder),
    #    num_parallel_calls=tf.data.AUTOTUNE)
    #if dataset_name == 'bridge_dataset':
    #    dataloader = dataloader.filter(lambda e:
    #        tf.math.reduce_mean(e['observation']['image_3'][0]) > 0)
    #if dataset_name in RLDS_TRANSFORM_DICT.keys():
    #    dataloader = dataloader.map(RLDS_TRANSFORM_DICT[dataset_name])
    #dataloader = dataloader.flat_map(lambda e: tf.data.Dataset.from_tensor_slices(e))
#
#    dataloader = dataloader.repeat(1).shuffle(shuffle_size, reshuffle_each_iteration=True)
 #   dataloader = dataloader.batch(batch_size)
 #   dataloader = tfds.as_numpy(dataloader)
    return dataset

#Load Policy Config and Policy
policy_config_path = '../policy_config/ConvPolicy.yaml'
with open(policy_config_path, 'rb') as f:
    policy_config = yaml.safe_load(f)
policy_class = 'ConvNet'
policy = make_policy(policy_class, policy_config)
checkpoint_path = '/iris/u/jyang27/dev/icl_controller_transfer/scripts/checkpoints/23-09-05-ConvNet-icl_controller_transfer-seed-2/policy_epoch_1900_seed_2.ckpt' 
#checkpoint_path = '/iris/u/jyang27/dev/icl_controller_transfer/scripts/checkpoints/23-09-06-ConvNet-icl_controller_transfer-seed-0/policy_epoch_1400_seed_0.ckpt'
with open(checkpoint_path, 'rb') as f:
    policy.load_state_dict(torch.load(f))
policy = policy.cuda()

#Make dataloader
TFDS_DATA_DIR = '/iris/u/jyang27/rlds_data'
train_dataloader = make_dataloader('polybot_dataset', 'train',
   2, data_dir=TFDS_DATA_DIR)
train_dataloaders = [train_dataloader]

def get_embedding(policy, image, goal_image, task_embedding=None):
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).cuda()
    if task_embedding is not None:
        task_embedding = torch.from_numpy(task_embedding)

    if goal_image is not None:
        goal_image = goal_image.astype(np.float32) / 255.0
        goal_image = torch.from_numpy(goal_image).cuda()
        image_data = torch.stack([image, goal_image], dim=1)      #b, 2, h, w, c
        image_data = torch.einsum('b k h w c -> b k c h w', image_data)
        image_data = image_data.flatten(1, 2) # b k c h w -> b (k c) h w
    else:
        image_data = torch.einsum('b h w c -> b c h w', image)

    return CNN.forward(policy, image_data, extra_fc_input=task_embedding, return_last_activations=True, intermediate_output_layer=0)

embeddings = []
distances = []
for i in train_dataloader:
    obs = [j['observation']['image'].numpy() for j in i['steps']]
    start = np.random.randint(len(obs)//2)
    goal_index = np.random.randint(start + 1, len(obs))
    embedding = get_embedding(policy, obs[start][None], obs[goal_index][None])
    embeddings.append(embedding[0][0].detach().cpu().numpy())
    distances.append(goal_index - start)

from scipy import stats
from sklearn.cross_decomposition import CCA

X = np.array(embeddings)
Y = np.array(distances).reshape(-1, 1)
cca = CCA(n_components=1)
cca.fit(X, Y)

X_c, Y_c = cca.transform(X, Y)
X_c = X_c.flatten()
Y_c = Y_c.flatten()
results = stats.pearsonr(X_c,Y_c)
print(results)
