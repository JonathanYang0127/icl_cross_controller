import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial
from omnimimic.data.rlds_data_transforms import RLDS_TRANSFORM_DICT
from omnimimic.data.dataset import *
from omnimimic.data.data_splits import DATASET_SPLITS

import matplotlib.pyplot as plt


if __name__ == '__main__':
    TFDS_DATA_DIR = '/iris/u/jyang27/rlds_data'
    datasets = list(DATASET_SPLITS.keys())[:3]
    #n_datasets = len(datasets)
    #datasets = ['taco_play']
    train_dataloaders, val_dataloaders, val_metadata = [], [], []
    config = {'wrist_image_only': True,
              'seq_length': 5,
              'visualize': False,
              'no_normalization': False}
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
             batch_size_train, shuffle_size=10000)

    train_dataloader = train_dataloader.as_numpy_iterator()
    traj = next(train_dataloader)

    for i in range(batch_size_train):
        plt.imsave(f'analyze_images/analyze_{i}.png', traj['observation']['image'][i])
        print(traj['action'][i])
