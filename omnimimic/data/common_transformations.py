import tensorflow as tf
from .data_utils import selective_tree_map

def random_dataset_sequence_transform_v2(traj, frame_stack, seq_length,
    pad_frame_stack, pad_seq_length):
    '''
    Extract a random subsequence of the data given sequence_length given keys
    '''

    if 'action_sequence' in traj.keys():
        traj_len = tf.shape(traj["action"])[0]
        index_in_demo = tf.random.uniform(shape=[],
            maxval=traj_len, dtype=tf.int32)
        traj['action'] = traj['action_sequence'][index_in_demo]
        traj['observation']['image'] = traj['observation']['image'][index_in_demo][None]
        for k in traj.keys():
            if 'goal' in k:    
                traj[k] = traj[k][index_in_demo][None]
        return traj

    traj_len = tf.shape(traj["action"])[0]
    seq_begin_pad, seq_end_pad = 0, 0
    if pad_frame_stack:
        seq_begin_pad = frame_stack - 1
    if pad_seq_length:
        seq_end_pad = seq_length - 1
    index_in_demo = tf.random.uniform(shape=[],
            maxval=traj_len + seq_end_pad - (seq_length - 1), 
          dtype=tf.int32)
    pad_mask = tf.concat((tf.repeat(0, repeats=seq_begin_pad),
                        tf.repeat(1, repeats=traj_len), 
                        tf.repeat(0, repeats=seq_end_pad)), axis=0)[:, None]
    traj['pad_mask'] = tf.cast(pad_mask, dtype=tf.bool)
    keys = ["observation", "action",  "goal"]

    def random_sequence_func(x):
        begin_padding = tf.repeat([x[0]], repeats=[seq_begin_pad], axis=0)
        end_padding = tf.repeat([x[-1]], repeats=[seq_end_pad], axis=0)
        sequence = tf.concat((begin_padding, x, end_padding), axis=0)
        return sequence[index_in_demo: index_in_demo + seq_length + frame_stack - 1]

    traj = selective_tree_map(
        traj,
        match=lambda x: any([key in str(x) for key in keys]),
        map_fn=random_sequence_func
    )
    return traj

def process_batch_transform(traj):
    new_traj = dict()
    new_traj['observation'] = dict()
    new_traj['observation']['image'] = traj['observation']['image'][0]
    new_traj['goal_image'] = traj['goal_image'][0]
    new_traj['goal_state'] = traj['goal_state'][0]
    new_traj['goal_distance_indices'] = traj['goal_distance_indices'][0]
    new_traj['action'] = traj['action']


    return new_traj

