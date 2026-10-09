import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial
from icl_controller_transfer.data.rlds_data_transforms import RLDS_TRANSFORM_DICT
from icl_controller_transfer.data.dataset import *
from icl_controller_transfer.data.data_splits import DATASET_SPLITS

import matplotlib.pyplot as plt


if __name__ == '__main__':
    TFDS_DATA_DIR = '/iris/u/jyang27/rlds_data'
    datasets = list(DATASET_SPLITS.keys())[0:1]
    train_dataloaders, val_dataloaders, val_metadata = [], [], []
    config = {'wrist_image_only': True,
              'seq_length': 1,
              'context_size': 5,
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

    train_dataloader = train_dataloader.as_numpy_iterator()
    for i in range(10):
        traj = next(train_dataloader)
        plt.imsave(f'analyze_images/test.png', traj['observation']['image'][:, :, -3:])
        import pdb; pdb.set_trace()
        print(traj['dataset_idx'])


    for i in range(batch_size_train):
        plt.imsave(f'analyze_images/analyze_{i}.png', traj['observation']['image'][i])
        plt.imsave(f'analyze_images/analyze_goal_{i}.png', traj['goal_image'][i])
        print(traj['goal_idxs'])
        print(traj['goal_distance_indices'])
