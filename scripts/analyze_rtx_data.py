import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial
from omnimimic.data.rlds_data_transforms import RLDS_TRANSFORM_DICT

import matplotlib.pyplot as plt

def decode_trajectory(builder, episode):
    steps = episode['steps']
    print(episode.keys())
    for key in steps:
        steps[key] = builder.info.features["steps"][
                        key
                    ].decode_batch_example(steps[key])
    if 'episode_metadata' in episode.keys() and 'file_path' in episode['episode_metadata'].keys():
        steps['file_path'] = builder.info.features[
            'episode_metadata']['file_path'].decode_batch_example(
            episode['episode_metadata']['file_path'])
    return steps

def dataset2path(dataset_name):
  if dataset_name == 'robo_net':
    version = '1.0.0'
  elif dataset_name == 'language_table':
    version = '0.0.1'
  else:
    version = '0.1.0'
  return f'gs://gresearch/robotics/{dataset_name}/{version}'

def make_dataloader(dataset_name, split, dataloader_config, data_dir=None):
    builder = tfds.builder(dataset_name, data_dir=data_dir)
    #builder = tfds.builder_from_directory(builder_dir=dataset2path(dataset_name))
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
dataset = 'kuka'
TFDS_DATA_DIR = '/iris/u/jyang27/rlds_data'
OXE_DATA_DIR = 'gs://gresearch/robotics/'
dataloader = make_dataloader(dataset,
        'train', 
        config,
data_dir = TFDS_DATA_DIR).shuffle(100)
dataloader = dataloader.as_numpy_iterator()

traj = next(dataloader)
print(traj.keys())
print(traj['observation'].keys())
import pdb; pdb.set_trace()


for i in range(20):
    traj = next(dataloader)

    print(traj.keys())
    print(traj['observation'].keys())
    #import pdb; pdb.set_trace()
    image_key = 'image'
    #image_key = 'exterior_image_1_left'
    #image_keys = ['wrist_image_left', 'exterior_image_1_left', 'exterior_image_1_right']
    image_keys=['rgb_static', 'rgb_gripper']
    #state_key = 'cartesian_position'
    state_key = None#'state'
    
    first_index = 0

    print(f'Traj {i}')
    if state_key is not None:
        print(traj['observation'][state_key][first_index])
    for key in image_keys:
        plt.imsave(f'analyze_images/analyze_{key}_{i}.png', traj['observation'][key][first_index])

