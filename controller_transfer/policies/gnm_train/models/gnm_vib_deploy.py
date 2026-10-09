
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple
from .modified_mobilenetv2 import MobileNetEncoder
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence


class GNM_VIB(nn.Module):
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
        mobilenet = MobileNetEncoder(num_images=1 + self.context_size)
        self.obs_mobilenet = mobilenet.features
        self.obs_encoding_size = obs_encoding_size
        self.compress_observation = nn.Sequential(
            nn.Linear(mobilenet.last_channel, self.obs_encoding_size),
            nn.ReLU(),
            nn.Linear(self.obs_encoding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
        )
        stacked_mobilenet = MobileNetEncoder(
            num_images=2 + self.context_size
        )  # stack the goal and the current observation
        self.goal_mobilenet = stacked_mobilenet.features
        self.goal_encoding_size = goal_encoding_size
        self.distro_size = goal_distro_size

        self.goal_to_z = nn.Sequential(
            nn.Linear(stacked_mobilenet.last_channel, self.goal_encoding_size),
            nn.ReLU(),
            nn.Linear(self.goal_encoding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * self.distro_size),
        )
        self.linear_layers = nn.Sequential(
            # TODO: change back output dim to 64
            nn.Linear(self.distro_size + 32, 32),
            # nn.ReLU(),
            # nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        self.dist_predictor = nn.Sequential(
            nn.Linear(32, 1),
        )
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
        )

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
    
    def vib_encode(self, obs_img: torch.Tensor, goal_img: torch.Tensor) -> torch.Tensor:
        obs_encoding = self._obs_encoder(obs_img)
        z_distro = self._goal_encoder(obs_img, goal_img)
        return obs_encoding, z_distro

    def _vib_decode(
        self, obs_encoding: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        x = torch.cat([obs_encoding, z], dim=-1)
        x = self.linear_layers(x)
        dist_pred = self.dist_predictor(x)
        action_pred = self.action_predictor(x)

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

    def n_sample(self, obs_encoding: torch.Tensor, z_distro: torch.Tensor, n: int) -> Any:
        """
        Sample n times from the given observation encoding and latent space distribution.
        TODO: Output dimensions are (batch_size, n, ...) 
        """
        with torch.no_grad():
            z = z_distro.sample((n,))
            # repeat the observation encoding n times
            obs_encoding = obs_encoding.unsqueeze(0).repeat(n, 1, 1)
            # combine the first two dimensions
            z = z.reshape(-1, z.shape[-1])
            obs_encoding = obs_encoding.reshape(-1, obs_encoding.shape[-1])
            dist_pred, action_pred = self._vib_decode(obs_encoding, z)
            # reshape the output
            dist_pred = dist_pred.reshape(n, -1, dist_pred.shape[-1])
            action_pred = action_pred.reshape(n, -1, action_pred.shape[1], action_pred.shape[2])
            return dist_pred, action_pred

    def forward(self,
             obs_img: torch.Tensor,
             goal_img: torch.Tensor,
             use_mean: bool = True,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_encoding, z_distro = self.vib_encode(obs_img, goal_img)
        if use_mean:
            z = z_distro.mean
        else:
            z = z_distro.rsample()
        dist_pred, action_pred = self._vib_decode(obs_encoding, z)
        return dist_pred, action_pred

    def loss(self,
            dist_pred: torch.Tensor,
            action_pred: torch.Tensor,
            dist_label: torch.Tensor,
            action_label: torch.Tensor,
            dist_z_distro: torch.Tensor,
            action_z_distro: torch.Tensor,
            r: torch.Tensor,
            ) -> torch.Tensor:
        """
        Compute the loss for the given observation encoding and latent space distribution.
        """
        dist_loss = F.mse_loss(dist_pred, dist_label)
        action_loss = F.mse_loss(action_pred, action_label)
        kld = 0.5 * torch.mean(kl_divergence(action_z_distro, r) + kl_divergence(dist_z_distro, r))
        return dist_loss, action_loss, kld