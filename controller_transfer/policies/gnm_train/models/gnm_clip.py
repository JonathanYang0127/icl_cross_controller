import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from .modified_mobilenetv2 import MobileNetEncoder
from .base_model import BaseModel


class GNM_CLIP(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoding_size: Optional[int] = 1024,
        clip_encoding_size: Optional[int] = 1024,
        obsgoal_encoding_size: Optional[int] = 512,
    ) -> None:
        """
        GNM main class
        Args:
            context_size (int): how many previous observations to used for context
            len_traj_pred (int): how many waypoints to predict in the future
            observation_encoding_size (int): size of the encoding of the observation images
            goal_encoding_size (int): size of the encoding of the goal images
            learn_angle (bool): whether to predict the yaw of the robot
        """
        super(GNM_CLIP, self).__init__(context_size, len_traj_pred, learn_angle)
        obs_mobilenet = MobileNetEncoder(num_images=1 + self.context_size)
        self.obs_cnn_enc = obs_mobilenet.features
        self.obs_encoding_size = obs_encoding_size
        self.compress_obs = nn.Sequential(
            nn.Linear(obs_mobilenet.last_channel, self.obs_encoding_size),
            nn.ReLU(),
        )
        goal_mobilenet = MobileNetEncoder(
            num_images=1
        )  # just the goal img (unlike v1 which has obs+goal)
        self.goal_cnn_enc = goal_mobilenet.features
        self.clip_encoding_size = clip_encoding_size
        # self.compress_goal = nn.Sequential(
        #     nn.Linear(goal_mobilenet.last_channel, self.clip_encoding_size),
        #     nn.ReLU(),
        # )
        self.obsgoal_encoding_size = obsgoal_encoding_size
        self.compress_obsgoal = nn.Sequential(
            nn.Linear(
                self.obs_encoding_size + self.clip_encoding_size,
                1024,
            ),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(self.obs_encoding_size + self.obsgoal_encoding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.dist_predictor = nn.Sequential(
            nn.Linear(32, 1),
        )
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
        )

    def forward(
        self, obs_img: torch.tensor, clip_goal_encoding: torch.tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_encoding = self.obs_cnn_enc(obs_img)
        obs_encoding = self.flatten(obs_encoding)
        obs_encoding = self.compress_obs(obs_encoding)

        # combine the obs and goal encoding
        clip_goal_encoding = torch.squeeze(clip_goal_encoding, dim=1)
        z_1 = torch.cat([obs_encoding, clip_goal_encoding], dim=1)
        obsgoal_encoding = self.compress_obsgoal(z_1)

        z_2 = torch.cat([obs_encoding, obsgoal_encoding], dim=1)
        z_2 = self.linear_layers(z_2)
        dist_pred = self.dist_predictor(z_2)
        action_pred = self.action_predictor(z_2)

        # augment outputs to match labels size-wise
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(
            action_pred[:, :, :2], dim=1
        )  # convert position deltas into waypoints
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )  # normalize the angle prediction
        return dist_pred, action_pred
