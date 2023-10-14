import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial
from .rlds_data_transforms import RLDS_TRANSFORM_DICT
from .common_transformations import *


def decode_trajectory(builder, episode):
    steps = episode['steps']
    for key in steps:
        steps[key] = builder.info.features["steps"][
                        key
                    ].decode_batch_example(steps[key])
    return steps

def apply_common_transforms(dataset, config):
    dataset = dataset.map(
        partial(
            random_dataset_sequence_transform_v2,
            frame_stack=1,
            seq_length=config['seq_length'],
            pad_frame_stack=True,
            pad_seq_length=True
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.map(
        process_batch_transform
    )
    return dataset


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
    dataloader = apply_common_transforms(dataloader, dataloader_config)
    dataloader = dataloader.repeat()
    
    return dataloader

def shuffle_batch_and_prefetch_dataloader(dataloader, batch_size, shuffle_size=None):
    if shuffle_size is not None:
        dataloader = dataloader.shuffle(shuffle_size)
    dataloader = dataloader.batch(batch_size)
    dataloader = dataloader.prefetch(1)
    return dataloader

