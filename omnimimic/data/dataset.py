import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial
from .rlds_data_transforms import RLDS_TRANSFORM_DICT
from .common_transformations import *
from .data_utils import *
from tqdm import tqdm
import numpy as np
import pickle

def decode_trajectory(builder, episode):
    steps = episode['steps']
    for key in steps:
        steps[key] = builder.info.features["steps"][
                        key
                    ].decode_batch_example(steps[key])
    return steps

def apply_common_transforms(dataset, config, 
    obs_action_metadata=None):
    if obs_action_metadata is not None:
        dataset = dataset.map(
            partial(
                normalize_obs_and_actions,
                action_keys=['action'],
                metadata=obs_action_metadata,
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
     
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
        process_batch_transform_v2
    )
    return dataset


def get_obs_action_metadata(
    name, builder, dataset, keys,
    load_if_exists=True
):
    # get statistics file path --> embed unique hash that catches if dataset info changed
    data_info_hash = name
    path = tf.io.gfile.join(
        builder.info.data_dir, f"obs_action_stats_{data_info_hash}.pkl"
    )

    # check if stats already exist and load, otherwise compute
    if tf.io.gfile.exists(path) and load_if_exists:
        print(f"Loading existing statistics for normalization from {path}.")
        with tf.io.gfile.GFile(path, "rb") as f:
            metadata = pickle.load(f)
    else:
        print("Computing obs/action statistics for normalization...")
        eps_by_key = {key: [] for key in keys}

        i, n_samples = 0, 1000
        dataset_iter = dataset.repeat().as_numpy_iterator()
        for _ in tqdm(range(n_samples)):
            episode = next(dataset_iter)
            i = i + 1
            for key in keys:
                eps_by_key[key].append(index_nested_dict(episode, key))
        eps_by_key = {key: np.concatenate(values) for key, values in eps_by_key.items()}

        metadata = {}
        for key in keys:
            metadata[key] = {
                "mean": eps_by_key[key].mean(0),
                "std": eps_by_key[key].std(0),
                "max": eps_by_key[key].max(0),
                "min": eps_by_key[key].min(0),
            }
        with tf.io.gfile.GFile(path, "wb") as f:
            pickle.dump(metadata, f)
        print("Done!")

    return metadata

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
    action_keys = ['action']
    metadata = get_obs_action_metadata(
        dataset_name,
        builder,
        dataloader,
        action_keys,
        load_if_exists=True,
    )
    dataloader = apply_common_transforms(dataloader, dataloader_config, 
            obs_action_metadata=metadata)
    dataloader = dataloader.flat_map(lambda e: tf.data.Dataset.from_tensor_slices(e))
    dataloader = dataloader.repeat()
    
    return dataloader

def shuffle_batch_and_prefetch_dataloader(dataloader, batch_size, shuffle_size=None):
    if shuffle_size is not None:
        dataloader = dataloader.shuffle(shuffle_size)
    dataloader = dataloader.batch(batch_size)
    dataloader = dataloader.prefetch(1)
    return dataloader

