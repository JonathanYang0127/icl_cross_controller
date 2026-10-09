import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from .base_model import BaseModel
from vitc.early_convolutions import EarlyConvViT
import pdb

IMAGE_SIZE = 128

class GNM_VITC(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoding_size: Optional[int] = 512,
        bottleneck_dim: Optional[int] = 32,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
    ) -> None:
        """
        GNM_VIT class: attention goes brrr
        """
        super(GNM_VITC, self).__init__(context_size, len_traj_pred, learn_angle)
        self.ViTC = EarlyConvViT(
            dim=obs_encoding_size,
            num_classes=bottleneck_dim,
            depth = mha_num_attention_layers,
            heads = mha_num_attention_heads,
            mlp_dim = 2048,
            channels = 3,
        )
        self.dist_predictor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(bottleneck_dim, 1),
        )
        self.action_predictor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(bottleneck_dim, self.len_trajectory_pred * self.num_action_params),
        )

    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # image size is [batch_size, 3*self.context_size, H, W]
        obs_img_list = list(torch.split(obs_img, 3, dim=1))
        obsgoal_img_list = obs_img_list + [goal_img]
        x = torch.cat(obsgoal_img_list, dim=-1)
        assert len(x.shape) == 4, "input image shape is not 4D"
        assert x.shape[1] == 3, "input image channel is not 3"
        assert x.shape[2] == IMAGE_SIZE, "input image height is not 128"
        assert x.shape[3] == IMAGE_SIZE*(self.context_size + 2), "input image width is not 128*(context_size + 2)"
       
        final_repr = self.ViTC(x)
        # currently, the size is [batch_size, 32]

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

    
