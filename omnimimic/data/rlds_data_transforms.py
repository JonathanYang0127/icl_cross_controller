from typing import Any, Dict
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import rlds

def gnm_dataset_transform(
    trajectory: Dict[str, Any],
    config
) -> Dict[str, Any]:

    def edit_action(step):
        step["action"] = tf.concat((
            tf.zeros(1, dtype=tf.dtypes.float64),
            -step["action"][1:2],
            -step["action"][0:1],
            tf.zeros(4, dtype=tf.dtypes.float64)
        ), axis=0)
        return step
    #goal_image = [i for i in trajectory[rlds.STEPS]][-1]['observation']['image']
    #goal_image = None
     #ds= trajectory[rlds.STEPS].as_numpy_iterator()
    step = rlds.transformations.final_step(trajectory[rlds.STEPS])
    goal_image = step['observation']['image']    
    def edit_goal_image(step):
        step['goal_image'] = goal_image
        return step
     
    trajectory[rlds.STEPS] = trajectory[rlds.STEPS].map(edit_action, num_parallel_calls=tf.data.AUTOTUNE)
    trajectory[rlds.STEPS] = trajectory[rlds.STEPS].map(edit_goal_image, num_parallel_calls=tf.data.AUTOTUNE)
    return trajectory[rlds.STEPS]

def gnm_dataset_transform_nodecode(
    trajectory: Dict[str, Any],
    config
) -> Dict[str, Any]:

    #Relabel actions
    traj_len = tf.shape(trajectory["action"])[0]
    len_seq_pred = config['seq_length']

    #Pad trajectory states
    padding = tf.tile(trajectory['observation']['state'][-1:, :], 
            [len_seq_pred, 1])
    trajectory['observation']['state'] = tf.concat(
            (trajectory['observation']['state'], padding),
            axis=0)

    #Get next len_seq_pred indices
    indices = tf.reshape(tf.range(traj_len), [-1, 1]) + tf.range(1, len_seq_pred + 1)
    global_waypoints = tf.gather(trajectory['observation']['state'], indices)[:, :, :2]

    #Get current position indices
    curr_pos = tf.expand_dims(trajectory['observation']['state'][:traj_len, :2], 1)
    global_waypoints -= curr_pos
    global_waypoints = tf.expand_dims(global_waypoints, 2)
    trajectory["action_sequence"] = tf.squeeze(tf.linalg.matmul(global_waypoints, tf.expand_dims(
        trajectory['observation']['yaw_rotmat'][:, :2, :2], 1)), 2)
    
    #Transform into manipulation coords
    trajectory["action_sequence"] = tf.concat((
            tf.zeros((traj_len, len_seq_pred, 1), dtype=tf.float64),
            -trajectory["action_sequence"][:,:, 1:2],
            -trajectory["action_sequence"][:,:, 0:1],
            tf.zeros((traj_len, len_seq_pred, 4), dtype=tf.float64)
            ), axis=-1)
    #trajectory["action_sequence"] = tf.zeros((traj_len, len_seq_pred, 7), dtype=tf.float64)
    trajectory["action"] = trajectory["action_sequence"][:, 0, :]

    #Relabel random goals
    rand = tf.random.uniform(shape=[traj_len])
    low = tf.cast(tf.range(traj_len) + len_seq_pred, dtype=tf.float32)
    high = tf.ones_like(low) * tf.cast(traj_len, dtype=tf.float32)
    goal_idxs = tf.cast((high-low) * rand + low, dtype=tf.int32)
    goal_idxs = tf.minimum(goal_idxs, traj_len - 1)
    
    trajectory["goal_image"] = tf.nest.map_structure(
        lambda x: tf.gather(x, goal_idxs),
        trajectory["observation"]["image"]
    )
    trajectory["goal_state"] = tf.nest.map_structure(
        lambda x: tf.gather(x, goal_idxs),
        trajectory["observation"]["state"]
        )[:, :2]
    trajectory["goal_state"] -= trajectory['observation']['state'][:traj_len, :2]
    trajectory["goal_state"] = tf.expand_dims(trajectory["goal_state"], 1)
    trajectory["goal_state"] = tf.squeeze(tf.linalg.matmul(trajectory["goal_state"],
        trajectory['observation']['yaw_rotmat'][:, :2, :2]), 1)
    trajectory["goal_distance_indices"] = tf.cast(goal_idxs, tf.float32) - low
    return trajectory

def polybot_dataset_transform_nodecode(
    trajectory: Dict[str, Any],
    config
) -> Dict[str, Any]:
    #Relabel random goals
    traj_len = len(trajectory["action"])
    rand = tf.random.uniform(shape=[traj_len])
    low = tf.cast(tf.range(traj_len), dtype=tf.float32)
    high = tf.ones_like(low) * tf.cast(traj_len, dtype=tf.float32)
    goal_idxs = tf.cast((high-low) * rand + low, dtype=tf.int32)
    goal_idxs = tf.minimum(goal_idxs, traj_len - 1)

    trajectory["goal_image"] = tf.nest.map_structure(
        lambda x: tf.gather(x, goal_idxs),
        trajectory["observation"]["image"]
    )
    #trajectory["goal_pose"] = tf.nest.map_structure(
    #    lambda x: tf.gather(x, goal_idxs),
    #    trajectory["observation"]["cartesian_pose"]
    #)
    #trajectory["goal_distance"] = torch_linalg.norm(
    #    trajectory["goal_pose"][:, :3] - trajectory["observation"]["cartesian_pose"][:, :3],
    #    dim=-1
    #)
    trajectory["goal_distance_indices"] = tf.cast(goal_idxs, tf.float32) - low
    return trajectory


def bridge_dataset_transform_nodecode(
    trajectory: Dict[str, Any],
    config
) -> Dict[str, Any]:
    print("keys", trajectory.keys())
    print(trajectory["observation"].keys())
    #set image as wrist image
    image_keys = []
    for key in trajectory["observation"].keys():
        if "image" in key and tf.math.reduce_mean(trajectory['observation'][key][0]) > 0:
            image_keys.append(key)
    if config['wrist_image_only']:
        image_keys = ['image_3']
    key = np.random.choice(image_keys)
    trajectory["observation"]["image"] = trajectory["observation"][key][:,::4,::4,:]
    
    #Set image as wrist camera image
    #trajectory["observation"]["image"] = trajectory["observation"]["image_3"][:,::4,::4,:]

    #Relabel random goals
    traj_len = len(trajectory["action"])
    rand = tf.random.uniform(shape=[traj_len])
    low = tf.cast(tf.range(traj_len), dtype=tf.float32)
    high = tf.ones_like(low) * tf.cast(traj_len, dtype=tf.float32)
    goal_idxs = tf.cast((high-low) * rand + low, dtype=tf.int32)
    goal_idxs = tf.minimum(goal_idxs, traj_len - 1)

    trajectory["goal_image"] = tf.nest.map_structure(
        lambda x: tf.gather(x, goal_idxs),
        trajectory["observation"]["image"]
    )
    trajectory["goal_distance_indices"] = tf.cast(goal_idxs, tf.float32) - low
    return trajectory

def r2d2_dataset_transform_nodecode(
    trajectory: Dict[str, Any],
    config
) -> Dict[str, Any]:
    print("keys", trajectory.keys())
    print(trajectory["observation"].keys())
    #set image as wrist image
    image_keys = []
    for key in trajectory["observation"].keys():
        if "image" in key and tf.math.reduce_mean(trajectory['observation'][key][0]) > 0:
            image_keys.append(key)
    if config['wrist_image_only']:
        image_keys = ['wrist_image_left', 'wrist_image_right']
    key = np.random.choice(image_keys)
    trajectory["observation"]["image"] = trajectory["observation"][key][:,::4,::4,:]
    trajectory["observation"]["image"] = tf.image.resize(trajectory["observation"]["image"],
            size=(64, 64), method='bicubic')

    #Set image as wrist camera image
    #trajectory["observation"]["image"] = trajectory["observation"]["image_3"][:,::4,::4,:]

    #Relabel random goals
    traj_len = len(trajectory["action"])
    rand = tf.random.uniform(shape=[traj_len])
    low = tf.cast(tf.range(traj_len), dtype=tf.float32)
    high = tf.ones_like(low) * tf.cast(traj_len, dtype=tf.float32)
    goal_idxs = tf.cast((high-low) * rand + low, dtype=tf.int32)
    goal_idxs = tf.minimum(goal_idxs, traj_len - 1)

    trajectory["goal_image"] = tf.nest.map_structure(
        lambda x: tf.gather(x, goal_idxs),
        trajectory["observation"]["image"]
    )
    trajectory["goal_distance_indices"] = tf.cast(goal_idxs, tf.float32) - low
    return trajectory


RLDS_TRANSFORM_DICT = {
    'gnm_dataset': gnm_dataset_transform_nodecode,
    'polybot_dataset': polybot_dataset_transform_nodecode,
    'bridge_dataset': bridge_dataset_transform_nodecode,
    'r2_d2': r2d2_dataset_transform_nodecode
}

