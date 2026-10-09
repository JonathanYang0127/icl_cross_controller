import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial
from controller_transfer.data.rlds_data_transforms import RLDS_TRANSFORM_DICT

import matplotlib.pyplot as plt

def decode_trajectory(builder, episode):
    steps = episode['steps']
    for key in steps:
        steps[key] = builder.info.features["steps"][
                        key
                    ].decode_batch_example(steps[key])
    return steps

def make_dataloader(dataset_name, split, dataloader_config, data_dir=None):
    builder = tfds.builder(dataset_name, data_dir=data_dir)
    dataset = builder.as_dataset(
            split=split,
            decoders={"steps": tfds.decode.SkipDecoding()},
            shuffle_files=True,
        )

    dataloader = dataset.map(partial(decode_trajectory, builder),
        num_parallel_calls=tf.data.AUTOTUNE)
    if dataloader_config['wrist_image_only']:
        if dataset_name == 'bridge_dataset':
            dataloader = dataloader.filter(lambda e:
                tf.math.reduce_mean(e['observation']['image_3'][0]) > 0)
    if dataset_name in RLDS_TRANSFORM_DICT.keys():
        dataloader = dataloader.map(partial(
            RLDS_TRANSFORM_DICT[dataset_name],
            config=dataloader_config),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    return dataloader

config = {
    'wrist_image_only': True,
    'seq_length': 5
}
dataset = 'r2_d2'
TFDS_DATA_DIR = '/iris/u/jyang27/rlds_data'
dataloader = make_dataloader(dataset,
        'train', 
        config,
        data_dir=TFDS_DATA_DIR).shuffle(100)
dataloader = dataloader.as_numpy_iterator()


for i in range(10):
    traj = next(dataloader)

    image_key = 'image'
    image_key = 'exterior_image_1_left'
    image_keys = ['wrist_image_left', 'exterior_image_1_left', 'exterior_image_1_right']
    state_key = 'cartesian_position'

    #state_key = 'state'
    first_index = 0
    second_index = 30

    print(f'Traj {i}')
    print(traj['observation'][state_key][first_index])
    print(traj['observation'][state_key][second_index])
    print(traj['observation'][state_key][second_index] - traj['observation'][state_key][first_index])
    print('')
    for key in image_keys:
        plt.imsave(f'coord_images/out_coord_initial_{key}_{i}.png', traj['observation'][image_key][first_index])
        plt.imsave(f'coord_images/out_coord_final_{key}_{i}.png', traj['observation'][image_key][second_index])

