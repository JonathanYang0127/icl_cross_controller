from iris_robots.robot_env import RobotEnv
import numpy as np
import pickle

env = RobotEnv(robot_model='wx250s', control_hz=5, use_local_cameras=True, camera_types='cv2',
    blocking=True)
env.reset()
env.step(np.array([0, 0, -0.3, 0, 0, 0, 0]))
import pdb; pdb.set_trace()

with open('goal_image.pkl', 'wb') as f:
    pickle.dump(env.get_observation()['images'][0]['array'], f)
