import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from .modified_mobilenetv2 import MobileNetEncoder
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence


class VariationalModel(nn.Module):
    def __init__(
        self,
        context_size: int = 0,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoding_size: Optional[int] = 1024,
        goal_encoding_size: Optional[int] = 1024,
        goal_distro_size: Optional[int] = 64,
    ) -> None:
        """
        GNM VIB model main class
        Args:
            context_size: how many previous observations to used for context
            len_traj_pred: how many waypoints to predict in the future
            learn_angle: whether to predict the yaw of the robot
            observation_encoding_size: size of the encoding of the observation images
            goal_encoding_size: size of the encoding of the goal images
        """
        super(GNM_VIB, self).__init__()
        self.context_size = context_size
        self.learn_angle = learn_angle
        self.len_trajectory_pred = len_traj_pred
        if self.learn_angle:
            self.num_action_params = 4
        else:
            self.num_action_params = 2

    def _flatten(self, z: torch.Tensor) -> torch.Tensor:
        z = nn.functional.adaptive_avg_pool2d(z, (1, 1))
        z = torch.flatten(z, 1)
        return z

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[1]
        return x[:, : size // 2], x[:, size // 2 :]

    def _distribution(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> MultivariateNormal:
        covar = torch.diag_embed(torch.exp(log_var))
        return MultivariateNormal(mu, covar)

    def _obs_encoder(self, obs_img: torch.Tensor) -> torch.Tensor:
        obs_encoding = self.obs_mobilenet(obs_img)
        obs_encoding = self._flatten(obs_encoding)
        obs_encoding = self.compress_observation(obs_encoding)
        return obs_encoding

    def _goal_encoder(
        self, obs_img: torch.Tensor, goal_img: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        obs_goal_input = torch.cat([obs_img, goal_img], dim=1)
        obs_goal_encoding = self.goal_mobilenet(obs_goal_input)
        obs_goal_encoding = self._flatten(obs_goal_encoding)
        obs_goal_encoding = self.goal_to_z(obs_goal_encoding)
        mu, log_var = self._split(obs_goal_encoding)
        z_distro = self._distribution(mu, log_var)
        return z_distro
    
    def encode(self, obs_img: torch.Tensor, goal_img: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _decode(
        self, obs_encoding: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        raise NotImplementedError

    def forward(self,
             obs_encoding: torch.Tensor,
             z_distro: torch.Tensor,
             use_mean: bool = False,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
    
    def loss(self,
            dist_pred: torch.Tensor,
            action_pred: torch.Tensor,
            dist_label: torch.Tensor,
            action_label: torch.Tensor,
            dist_z_distro: torch.Tensor,
            action_z_distro: torch.Tensor,
            ) -> torch.Tensor:
        """
        Compute the loss for the given observation encoding and latent space distribution.
        """
        dist_loss = F.mse_loss(dist_pred, dist_label)
        action_loss = F.mse_loss(action_pred, action_pred)
        kld = 0.5 * torch.mean(kl_divergence(action_z_distro, self.r) + kl_divergence(dist_z_distro, self.r))
        return dist_loss, action_loss, kld