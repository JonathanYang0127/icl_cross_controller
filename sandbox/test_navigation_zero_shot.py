# ROS
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray

# UTILS
from utils import msg_to_pil, to_numpy, transform_images, load_model, pil_to_msg
from iris_robots.robot_env import RobotEnv

import torch
from PIL import Image as PILImage
import numpy as np
import os
import argparse
import yaml
import clip
import tqdm
import time
import pdb
import pickle
import pdb
#from topic_names import BACKTRACKING_IMAGE_TOPIC

MODEL_WEIGHTS_PATH = "../model_weights/latest.pth"
MODEL_CONFIG_PATH = "../checkpoints/base_vint_4x4_0c.yaml"

# DEFAULT MODEL PARAMETERS (can be overwritten by model.yaml)
model_params = {
    "path": "large_gnm.pth", # path of the model in ../
    "model_type": "gnm", # gnm (conditioned), stacked, or siamese
    "context_size": 5, # number of images to use as context
    "len_traj_pred": 5, # number of waypoints to predict
    "normalize": True, # bool to determine whether or not normalize images
    "image_size": [64, 85], # (height, width)
    "normalize": True, # bool to determine whether or not normalize the waypoints
    "learn_angle": True, # bool to determine whether or not to learn/predict heading of the robot
    "obs_encoding_size": 1024,
    "goal_encoding_size": 1024,
    "obsgoal_encoding_size": 2048,
    "goal_type": "image", # image 
}

# GLOBALS
context_queue = []    
# v9_context_queue = []
constant_context = False


# Load the model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def add_recent_image(obs_img):
    if len(context_queue) < model_params["context_size"] + 1:
        context_queue.append(obs_img)
    else:
        if constant_context:
            print('only most recent image updated')
        else:
            context_queue.pop(0)
            context_queue.append(obs_img)
            print('context queue updated')

    # if len(v9_context_queue) < 5 + 1:
    #     v9_context_queue.append(obs_img)
    # else:
    #     v9_context_queue.pop(0)
    #     v9_context_queue.append(obs_img)
    #     print('context queue updated')


def main(args: argparse.Namespace):
    global context_queue, constant_context, recent_obs

    #Load goal image
    goal_image = None
    with open('goal_image.pkl', 'rb') as f:
        goal_image = pickle.load(f)
        goal_image = PILImage.fromarray(goal_image)
    env = RobotEnv(robot_model='wx250s', control_hz=5, use_local_cameras=True, camera_types='cv2',
            blocking=True)
    obs = env.reset()


    # load model parameters
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_config = yaml.safe_load(f)
    # import pdb; pdb.set_trace()
    print(model_config.keys())
    v9_params = model_params.copy()
    for param in model_config:
        model_params[param] = model_config[param]

    # load model weights
    model_path = MODEL_WEIGHTS_PATH
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    model_params['model_type'] = 'gnmv9'
    model = load_model(
        model_path,
        model_params,
        device,
    )
    model.eval()

    constant_context = False
    if "constant_context" in model_params:
        constant_context = model_params["constant_context"]

    if constant_context:
        print("using constant context")
    else:
        print("using variable context")

    # navigation loop
    for i in range(1):
        recent_obs = env.get_observation()
        add_recent_image(PILImage.fromarray(recent_obs['images'][0]['array']))
        if len(context_queue) > model_params["context_size"]: # and len(v9_context_queue) > v9_params["context_size"]:
            #start = max(closest_node - args.radius, 0)
            #end = min(closest_node + args.radius + 1, goal_node)
            distances = []
            waypoints = []
            batch_obs_imgs = []
            batch_goal_data = []
            batch_recent_obs = []

            '''
            for i, sg_img in enumerate(topomap[start: end + 1]):
                transf_obs_img = transform_images(context_queue, model_params["image_size"])
                tranf_recent_obs_img = transform_images(recent_obs, model_params["image_size"])
                if "goal_type" in model_params and model_params["goal_type"] == "clip":
                    goal_data = clip_goal_embeddings[start + i]
                else:
                    goal_data = transform_images(sg_img, model_params["image_size"])
                batch_obs_imgs.append(transf_obs_img)
                batch_goal_data.append(goal_data)
                batch_recent_obs.append(tranf_recent_obs_img)
            '''

            batch_obs_imgs = [transform_images(context_queue, model_params["image_size"])]
            batch_goal_data = [transform_images([goal_image], model_params["image_size"])]
            
            # predict distances and waypoints
            batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(device)
            batch_goal_data = torch.cat(batch_goal_data, dim=0).to(device)
       
            # v9_batch_obs_imgs = torch.cat(v9_batch_obs_imgs, dim=0).to(device)
            # v9_batch_goal_data = torch.cat(v9_batch_goal_data, dim=0).to(device)
            if constant_context:
                batch_recent_obs = torch.cat(batch_recent_obs, dim=0).to(device)
                assert model_params["model_type"] == "gnmv9"
                distances, waypoints = model.constant_forward(batch_obs_imgs, batch_recent_obs, batch_goal_data) 
    
            distances, waypoints = model(batch_obs_imgs, batch_goal_data)
            # distances, _ = v9_no_resample(v9_batch_obs_imgs, v9_batch_goal_data)
            distances = to_numpy(distances)
            waypoints = to_numpy(waypoints)
            # look for closest node
            closest_node = np.argmin(distances)
            # chose subgoal and output waypoints
            
            
            #if distances[closest_node] > args.close_threshold:
            chosen_waypoint = waypoints[closest_node][args.waypoint]
                #sg_img = topomap[start + closest_node]
            #else:
                #chosen_waypoint = waypoints[min(
                #    closest_node + 1, len(waypoints) - 1)][args.waypoint]
                #sg_img = topomap[start + min(closest_node + 1, len(waypoints) - 1)]
            # publish subgoal image to backtrack
            #backtrack_pub.publish(pil_to_msg(sg_img, "rgb8"))
            #waypoint_msg = Float32MultiArray()
            #if model_params["normalize"]:
            #    chosen_waypoint[:2] *= (MAX_V / RATE)
            #waypoint_msg.data = chosen_waypoint
            #waypoint_pub.publish(waypoint_msg)
            #closest_node += start
            
            print("waypoint:", chosen_waypoint)
            env.step(np.array([0, -chosen_waypoint[1]/10, -chosen_waypoint[0]/10, 0, 0, 0, 0]))
            #rate.sleep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to run GNMs on the locobot")
    parser.add_argument(
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help="path to topomap images",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="gnmv9",
        type=str,
        help="model name (hint: check ../config/models.yaml) (default: large_gnm)",
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=3,
        type=int,
        help="""temporal distance within the next node in the topomap before 
        localizing to it (default: 3)""",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=4,
        type=int,
        help="""temporal number of locobal nodes to look at in the topopmap for
        localization (default: 2)""",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2, # close waypoints exihibit straight line motion (the middle waypoint is a good default)
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or 
        how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="""goal node index in the topomap (if -1, then the goal node is 
        the last node in the topomap) (default: -1)""",
    )
    args = parser.parse_args()
    print(f"Using {device}")
    main(args)


