import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from efficientnet_pytorch import EfficientNet
from .base_model import BaseModel
import pdb

class GNM_V3_FC(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoder: Optional[str] = "efficientnet-b0",
        goal_encoder: Optional[str] = None,
        goal_embedding_size: Optional[int] = None,
        obs_encoding_size: Optional[int] = 512,
        goal_encoding_size: Optional[int] = 512,
        freeze_obs_encoder: Optional[bool] = True,
        freeze_goal_encoder: Optional[bool] = True,
    ) -> None:
        """
        GNMv3 FC class: This encodes each observation independently, towards an attention-based scheme.
        Uses FC layers and standard goal-conditioning.
        """
        super(GNM_V3_FC, self).__init__(context_size, len_traj_pred, learn_angle)
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = goal_encoding_size
        self.obs_encoder = obs_encoder
        self.goal_encoder = goal_encoder

        if self.obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_pretrained(self.obs_encoder)
        else:
            raise NotImplementedError

        if goal_encoder is None:
            assert goal_embedding_size is not None
            assert freeze_goal_encoder is True
            self.num_goal_features = goal_embedding_size

        if freeze_obs_encoder:
            for param in self.obs_encoder.parameters():
                param.requires_grad = False
        self.num_obs_features = self.obs_encoder._fc.in_features
        
        if freeze_goal_encoder and goal_encoder is not None:
            raise NotImplementedError
            for param in self.goal_encoder.parameters():
                param.requires_grad = False
            self.num_goal_features = self.goal_encoder._fc.in_features

        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
        
        if self.num_goal_features != self.goal_encoding_size:
            self.compress_goal_enc = nn.Linear(self.num_goal_features, self.goal_encoding_size)
        else:
            self.compress_goal_enc = nn.Identity()

        self.decoder = nn.Sequential(
            nn.Linear(
                (self.context_size + 2) * self.obs_encoding_size, self.obs_encoding_size
            ),
            nn.ReLU(),
            nn.Linear(self.obs_encoding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
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
        self, obs_img: torch.tensor, goal_img: torch.tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.goal_encoder is None:
            # goal image is goal embedding, precomputed
            goal_embedding = goal_img.float()  # cast to float
        else:
            goal_embedding = self.goal_encoder(goal_img)
        
        # split the observation into context based on the context size
        # image size is [batch_size, 3*self.context_size, H, W]
        obs_img = torch.split(obs_img, 3, dim=1)

        # image size is [batch_size*self.context_size, 3, H, W]
        obs_img = torch.concat(obs_img, dim=0)
        # get the observation encoding
        obs_encoding = self.obs_encoder.extract_features(obs_img)
        # currently the size is [batch_size*self.context_size, 1280, H/32, W/32]

        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        # currently the size is [batch_size*self.context_size, 1280, 1, 1]

        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        obs_encoding = self.compress_obs_enc(obs_encoding)
        # currently, the size is [batch_size*self.context_size, self.obs_encoding_size]
        
        # reshape the obs_encoding to [context, batch, encoding_size], note that the order is flipped
        obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)
        # currently, the size is [batch_size, self.context_size+1, self.obs_encoding_size]

        goal_embedding = self.compress_goal_enc(goal_embedding)
        # currently, the size of goal_embedding is [batch_size, 1, self.goal_encoding_size]
        assert goal_embedding.shape[2] == self.goal_encoding_size

        # concatenate the goal embedding to the observation encoding
        obs_encoding = torch.cat((obs_encoding, goal_embedding), dim=1)
        # currently, the size is [batch_size, self.context_size+2, self.obs_encoding_size]
        

        obs_encoding = obs_encoding.reshape((obs_encoding.shape[0], -1))
        # currently, the size is [batch_size, (self.context_size+2)*self.obs_encoding_size]

        final_repr = self.decoder(obs_encoding)
        dist_pred = self.dist_predictor(final_repr)
        action_pred = self.action_predictor(final_repr)

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