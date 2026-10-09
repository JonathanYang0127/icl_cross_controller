import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from .base_model import BaseModel
from vit_pytorch import SimpleViT
import pdb

IMAGE_SIZE = 128

class ViT_DP_Encoder(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        image_size: int = 128, # if tuple, then (width, height)
        patch_size: int = 32,
        bottleneck_dim: Optional[int] = 32,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
    ) -> None:
        """
        GNM_VIT class: attention goes brrr
        """
        super(ViT_DP_Encoder, self).__init__(context_size, len_traj_pred, learn_angle)
        self.patch_size = patch_size
        if type(image_size) == int:
            self.image_height = image_size
            self.image_width = image_size
        else:
            self.image_width = image_size[0]
            self.image_height = image_size[1]
        self.ViT = SimpleViT(
            image_size=(self.image_height, self.image_width*(self.context_size + 2)),
            patch_size=self.patch_size,
            dim=512,
            num_classes=bottleneck_dim,
            depth = mha_num_attention_layers,
            heads = mha_num_attention_heads,
            mlp_dim = 2048
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
        assert x.shape[2] == self.image_height, "input image height is not 128"
        assert x.shape[3] == self.image_width*(self.context_size + 2), "input image width is not 128*(context_size + 2)"
       
        final_repr = self.ViT(x)
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
    

    def final_embedding(self, obs_img: torch.tensor, goal_img: torch.tensor,):
        # image size is [batch_size, 3*self.context_size, H, W]
        obs_img_list = list(torch.split(obs_img, 3, dim=1))
        obsgoal_img_list = obs_img_list + [goal_img]
        x = torch.cat(obsgoal_img_list, dim=-1)
        assert len(x.shape) == 4, "input image shape is not 4D"
        assert x.shape[1] == 3, "input image channel is not 3"
        assert x.shape[2] == self.image_height, "input image height is not 128"
        assert x.shape[3] == self.image_width*(self.context_size + 2), "input image width is not 128*(context_size + 2)"
       
        latent = self.ViT.get_latent(x)
        return latent

    
