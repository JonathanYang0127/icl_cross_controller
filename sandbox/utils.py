# modify sys.path to import from train/
import os
import sys
import io
import matplotlib.pyplot as plt
# FILE_DIR = os.path.dirname(__file__)
# sys.path.append(os.path.join(FILE_DIR, "../../train/"))

# ROS
from sensor_msgs.msg import Image

# pytorch
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image as PILImage
from typing import List, Tuple, Dict, Optional

# models
from models.gnm import GNM
from models.stacked import StackedModel
from models.siamese import SiameseModel
# from gnm_train.models.gnm_distilled import GNM_distilled
# from gnm_train.models.gnm_distilled_double import GNM_distilled_double
# from gnm_train.models.gnm_distilled_double_bins import GNM_distilled_double_bins
from gnm_train.models.gnm_v4 import GNM_V4_SA
from gnm_train.models.gnm_v6 import GNM_V6_SA
from gnm_train.models.gnm_v7 import GNM_V7_SA
from gnm_train.models.gnm_v9 import GNM_V9_SA
from gnm_train.models.gnm_v10 import GNM_V10_SA
from gnm_train.models.gnm_v1_bc import GNM_V1_BC
from gnm_train.models.gnm_v9_bc import GNM_V9_BC
from gnm_train.models.gnm_v9_fastrlap import GNM_V9_FASTRLAP
from gnm_train.models.gnm_vib_deploy import GNM_VIB
from gnm_train.models.gnm_vit import GNM_VIT
from gnm_train.models.vint_distillation import ViNT_distilled

from gnm_train.data.data_utils import IMAGE_ASPECT_RATIO, VISUALIZATION_IMAGE_SIZE

def load_model(
    model_path: str,
    config: dict,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Load a model from a checkpoint file (works with models trained on multiple GPUs)"""
    checkpoint = torch.load(model_path, map_location=device)
    loaded_model = checkpoint["model"]
    if config["model_type"] == "gnm":
        model = GNM(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoding_size"],
            config["goal_encoding_size"],
        )
    # elif model_params["model_type"] == "gnm_distilled":
    #     model = GNM_distilled(
    #         model_params["context_size"],
    #         None,
    #         model_params["len_traj_pred"],
    #         model_params["learn_angle"],
    #         model_params["obs_encoding_size"],
    #         model_params["goal_encoding_size"],
    #         False, # pretrained
    #         1, # 1 or 3 bins
    #     )
    # elif model_params["model_type"] == "gnm_cil":
    #     model = GNM_distilled(
    #         model_params["context_size"],
    #         None,
    #         model_params["len_traj_pred"],
    #         model_params["learn_angle"],
    #         model_params["obs_encoding_size"],
    #         model_params["goal_encoding_size"],
    #         False, # pretrained
    #         3, # 1 or 3 bins
    #     )
    # elif model_params["model_type"] == "gnm_distilled_double":
    #     print("Loading distilled double model")
    #     model = GNM_distilled_double(
    #         model_params["context_size"],
    #         None,
    #         model_params["len_traj_pred"],
    #         model_params["learn_angle"],
    #         model_params["obs_encoding_size"],
    #         model_params["goal_encoding_size"],
    #         False, # pretrained
    #         1, # 1 or 3 bins
    #     )
    # elif model_params["model_type"] == "gnm_distilled_double_bins":
    #     print("Loading distilled double model")
    #     model = GNM_distilled_double_bins(
    #         model_params["context_size"],
    #         None,
    #         model_params["len_traj_pred"],
    #         model_params["learn_angle"],
    #         model_params["obs_encoding_size"],
    #         model_params["goal_encoding_size"],
    #         False, # pretrained
    #         3, # 1 or 3 bins
    #     )
    elif config["model_type"] == "siamese":
        model = SiameseModel(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoding_size"],
            config["goal_encoding_size"],
            config["obsgoal_encoding_size"],
        )
    elif config["model_type"] == "stacked":
        model = StackedModel(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoding_size"],
            config["obsgoal_encoding_size"],
        )
    elif config["model_type"] == "gnmv4":
        print("Loading GNM V4 model")
        print("Model params: ", config)
        model = GNM_V4_SA(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoder"],
            config["goal_encoder"],
            config["goal_embedding_size"],
            config["obs_encoding_size"],
            config["freeze_obs_encoder"],
            config["freeze_goal_encoder"],
            config["mha_num_attention_heads"],
            config["mha_num_attention_layers"],
            config["mha_ff_dim_factor"],
        )
    elif config["model_type"] == "gnmv6":
        print("Loading GNM V6 model")
        print("Model params: ", config)
        model = GNM_V6_SA(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoder"],
            config["obs_encoding_size"],
            config["mha_num_attention_heads"],
            config["mha_num_attention_layers"],
            config["mha_ff_dim_factor"],
        ) 
    elif config["model_type"] == "gnmv7":
        model = GNM_V7_SA(
            context_size = config["context_size"],
            len_traj_pred = config["len_traj_pred"],
            learn_angle = config["learn_angle"],
            obs_encoder = config["obs_encoder"],
            goal_encoder = config["goal_encoder"],
            goal_embedding_size = config["goal_embedding_size"],
            obs_encoding_size = config["obs_encoding_size"],
            freeze_obs_encoder = config["freeze_obs_encoder"],
            freeze_goal_encoder = config["freeze_goal_encoder"],
            mha_num_attention_heads = config["mha_num_attention_heads"],
            mha_num_attention_layers = config["mha_num_attention_layers"],
            mha_ff_dim_factor = config["mha_ff_dim_factor"],
        )
    elif config["model_type"] == "gnmv9":
        model = GNM_V9_SA(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    elif config["model_type"] == "gnmv10":
        model = GNM_V10_SA(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    elif config["model_type"] == "gnm_v1_bc":
        model = GNM_V1_BC(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoding_size=config["obs_encoding_size"],
        )
    elif config["model_type"] == "gnm_v9_bc":
        model = GNM_V9_BC(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    elif config["model_type"] == "gnm_vib":
        model = GNM_VIB(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoding_size=config["obs_encoding_size"],
            goal_encoding_size=config["goal_encoding_size"],
            goal_distro_size=config["goal_distro_size"],
        )
    elif config["model_type"] == "gnmv9_fastrlap":
        model = GNM_V9_FASTRLAP(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoding_size=config["obs_encoding_size"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    elif config["model_type"] == "gnm_vit":
        model = GNM_VIT(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            image_size=config["image_size"],
            patch_size=config["patch_size"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
        )
    elif config["model_type"] == "vint_distill":
        model = ViNT_distilled(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoding_size=config["obs_encoding_size"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
            pretrained=False,
            checkpoint_path=None,
            mode_dim=2, # position
            freeze=False,
            deploy_mode=True,
        )
    else:
        model_type = config["model_type"]
        raise ValueError(f"Invalid model type: {model_type}")
    try:
        state_dict = loaded_model.module.state_dict()
        model.load_state_dict(state_dict, strict=False)
    except AttributeError as e:
        state_dict = loaded_model.state_dict()
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model


def msg_to_pil(msg: Image) -> PILImage.Image:
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, -1)
    pil_image = PILImage.fromarray(img)
    return pil_image


def pil_to_msg(pil_img: PILImage.Image, encoding="mono8") -> Image:
    img = np.asarray(pil_img)  
    ros_image = Image(encoding=encoding)
    try:
        ros_image.height, ros_image.width, _ = img.shape
    except ValueError as e:
        print(e)
        print("shape: ", img.shape)
    ros_image.data = img.ravel().tobytes() 
    ros_image.step = ros_image.width
    return ros_image


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def transform_images(pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = True) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225]),
        ]
    )
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))  # crop to the right ratio
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_img = pil_img.resize(image_size[::-1]) #image_size[::-1])
        # pil_img = TF.resize(pil_img, )
        transf_img = transform_type(pil_img)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)


def graph_actions(actions: np.ndarray, ax, alphas=None, colors=None):
    if alphas is None:
        alphas = [0.5] * len(actions)
    if colors is None:
        colors = ["blue"] * len(actions)
    for action, alpha, color in zip(actions, alphas, colors):
        # make dots large
        # print("action.shape", action.shape)
        ax.plot(-action[:, 1], action[:, 0], "o-", color=color, alpha=alpha, markersize=2)
        # plot quiver
        # ax.quiver(action_tensor[:, 0], action_tensor[:, 1], action_tensor[:, 2], action_tensor[:, 3],  color="red", scale=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("actions")

    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal", "box")


def graph_samples_img(waypoints: np.ndarray, distances: Optional[np.ndarray], im_obs: PILImage.Image, save_path: str, colors=None):
    fig, axs = plt.subplots(1,2) 
    indices = np.argsort(distances)
    # graph the image on ax[0]
    axs[0].imshow(im_obs)

    # graph all the actions on ax[1]
    if distances is None:
        alphas_sorted = None
    else:
        alphas = [(len(distances) - i)/len(distances) for i in range(len(distances))]
        alphas_sorted = [alphas[i] for i in indices]
    graph_actions(waypoints, axs[1], alphas=alphas_sorted, colors=colors)
    plt.savefig(save_path)
    plt.close()
    

# clip angle between -pi and pi
def clip_angle(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi
