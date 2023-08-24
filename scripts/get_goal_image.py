from iris_robots.robot_env import RobotEnv
import numpy as np
import pickle
from PIL import Image

env = RobotEnv(robot_model='wx250s', control_hz=5, use_local_cameras=True, camera_types='cv2',
    blocking=True)
env.reset()
env.step(np.array([0.1, 0.1, -0.5, 0, 0, 0, 0]))
import pdb; pdb.set_trace()
env.step(np.array([0, 0, 0, 0, 0, 0, 1])) 

#with open('goal_image.png', 'wb') as f:
obs = env.get_observation()
image = Image.fromarray(obs['images'][0]['array'])
image = image.save('goal_image.png')


