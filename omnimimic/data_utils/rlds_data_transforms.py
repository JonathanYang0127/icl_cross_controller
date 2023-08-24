from typing import Any, Dict
import tensorflow as tf
import tensorflow_datasets as tfds
import rlds

def gnm_dataset_transform(
    trajectory: Dict[str, Any]
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
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    #Relabel actions
    batch_size = len(trajectory["action"])
    trajectory["action"] = tf.concat((
            tf.zeros((batch_size, 1), dtype=tf.dtypes.float64),
            -trajectory["action"][:, 1:2],
            -trajectory["action"][:, 0:1],
            tf.zeros_like(trajectory["action"][:, 3:])
        ), axis=1)

    #Relabel random goals
    rand = tf.random.uniform(shape=[batch_size])
    low = tf.cast(tf.range(batch_size), dtype=tf.float32)
    high = tf.ones_like(low) * tf.cast(batch_size, dtype=tf.float32)
    goal_idxs = tf.cast((high-low) * rand + low, dtype=tf.int32)
    goal_idxs = tf.minimum(goal_idxs, batch_size - 1)
    
    trajectory["goal_image"] = tf.nest.map_structure(
        lambda x: tf.gather(x, goal_idxs),
        trajectory["observation"]["image"]
    )
    return trajectory

def polybot_dataset_transform_nodecode(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    #Relabel random goals
    batch_size = len(trajectory["action"])
    rand = tf.random.uniform(shape=[batch_size])
    low = tf.cast(tf.range(batch_size), dtype=tf.float32)
    high = tf.ones_like(low) * tf.cast(batch_size, dtype=tf.float32)
    goal_idxs = tf.cast((high-low) * rand + low, dtype=tf.int32)
    goal_idxs = tf.minimum(goal_idxs, batch_size - 1)

    trajectory["goal_image"] = tf.nest.map_structure(
        lambda x: tf.gather(x, goal_idxs),
        trajectory["observation"]["image"]
    )
    return trajectory



RLDS_TRANSFORM_DICT = {
    'gnm_dataset': gnm_dataset_transform_nodecode,
    'polybot_dataset': polybot_dataset_transform_nodecode
}

