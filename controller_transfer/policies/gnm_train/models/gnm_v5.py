import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
from efficientnet_pytorch import EfficientNet
from .base_model import BaseModel
from .self_attention import MultiLayerDecoder
import pdb
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence


class GNM_V5(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoder: Optional[str] = "efficientnet-b0",
        goal_encoder: Optional[str] = None,
        goal_embedding_size: Optional[int] = None,
        obs_encoding_size: Optional[int] = 512,
        freeze_obs_encoder: Optional[bool] = True,
        freeze_goal_encoder: Optional[bool] = True,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        """
        GNMv5 class: Uses the architecture from v4, but adds sampling (like RECON).
        Goal-conditioning is standard: goal is concatenated to the context.
        """
        super(GNM_V5, self).__init__(
            context_size, len_traj_pred, learn_angle)
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.obs_encoder = obs_encoder
        self.encode = goal_encoder

        if self.obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_pretrained(self.obs_encoder)
        else:
            raise NotImplementedError

        if goal_encoder is None:
            assert goal_embedding_size is not None
            assert freeze_goal_encoder is True
            self.num_goal_features = goal_embedding_size
        elif self.encode.split("-")[0] == "efficientnet":
            self.encode = EfficientNet.from_pretrained(self.encode)
            self.num_goal_features = self.encode._fc.in_features
        else:
            raise NotImplementedError

        if freeze_obs_encoder:
            for param in self.obs_encoder.parameters():
                param.requires_grad = False
        self.num_obs_features = self.obs_encoder._fc.in_features

        if freeze_goal_encoder and goal_encoder is not None:
            for param in self.encode.parameters():
                param.requires_grad = False
            self.num_goal_features = self.encode._fc.in_features

        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(
                self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()

        if self.num_goal_features != self.goal_encoding_size:
            self.compress_goal_enc = nn.Linear(
                self.num_goal_features, self.goal_encoding_size)
        else:
            self.compress_goal_enc = nn.Identity()

        self.z_decoder = MultiLayerDecoder(
            embed_dim=self.obs_encoding_size,
            seq_len=self.context_size+2,
            output_layers=[obs_encoding_size, obs_encoding_size * 2],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )

        self.decoder = MultiLayerDecoder(
            embed_dim=self.obs_encoding_size,
            seq_len=self.context_size+2,
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )
        self.dist_predictor = nn.Sequential(
            nn.Linear(32, 1),
        )
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
        )
        self.distro_size = self.obs_encoding_size
        self.r = MultivariateNormal(torch.zeros(
            self.distro_size), torch.eye(self.obs_encoding_size))

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[1]
        return x[:, : size // 2], x[:, size // 2:]

    def _distribution(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> MultivariateNormal:
        covar = torch.diag_embed(torch.exp(log_var))
        return MultivariateNormal(mu, covar)

    def vib_encode(self, obs_img: torch.tensor, goal_img: torch.tensor) -> Tuple[torch.Tensor, MultivariateNormal]:
        """
        Encode the observation and goal images into a latent space.
        """

        if self.encode is None:
            # goal image is goal embedding, precomputed
            goal_embedding = goal_img.float()  # cast to float
        else:
            goal_embedding = self.encode.extract_features(goal_img)
            goal_embedding = self.encode._avg_pooling(goal_embedding)
            if self.encode._global_params.include_top:
                goal_embedding = goal_embedding.flatten(start_dim=1)
                goal_embedding = self.encode._dropout(goal_embedding)
            # currently, the size is [batch_size, num_goal_features]

        # split the observation into context based on the context size
        # image size is [batch_size, 3*self.context_size, H, W]
        obs_img = torch.split(obs_img, 3, dim=1)

        # image size is [batch_size*self.context_size, 3, H, W]
        obs_img = torch.concat(obs_img, dim=0)
        # get the observation encoding
        obs_encoding = self.obs_encoder.extract_features(obs_img)
        # currently the size is [batch_size*(self.context_size + 1), 1280, H/32, W/32]

        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        # currently the size is [batch_size*(self.context_size + 1), 1280, 1, 1]

        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        obs_encoding = self.compress_obs_enc(obs_encoding)
        # currently, the size is [batch_size*(self.context_size + 1), self.obs_encoding_size]

        # reshape the obs_encoding to [context + 1, batch, encoding_size], note that the order is flipped
        obs_encoding = obs_encoding.reshape(
            (self.context_size+1, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)
        # currently, the size is [batch_size, self.context_size+1, self.obs_encoding_size]

        goal_embedding = self.compress_goal_enc(goal_embedding)
        if len(goal_embedding.shape) == 2:
            goal_embedding = goal_embedding.unsqueeze(1)
        # currently, the size of goal_embedding is [batch_size, 1, self.goal_encoding_size]
        assert goal_embedding.shape[2] == self.goal_encoding_size

        # concatenate the goal embedding to the observation encoding
        obsgoal_encoding = torch.cat((obs_encoding, goal_embedding), dim=1)
        # currently, the size is [batch_size, self.context_size+2, self.obs_encoding_size]

        z_distro_params = self.z_decoder(obsgoal_encoding)
        # currently, the size is [batch_size, self.obs_encoding_size]

        mu, log_var = self._split(z_distro_params)
        z_distro = self._distribution(mu, log_var)
        return obs_encoding, z_distro

    def _vib_decode(self, obs_encoding: torch.Tensor, z: torch.Tensor) -> Any:
        """
        Decode the observation encoding and latent space into a distribution over the next action.
        """
        # concatenate the z to the observation encoding
        z = z.unsqueeze(1)
        obsgoal_encoding = torch.cat((obs_encoding, z), dim=1)
        # currently, the size is [batch_size, self.context_size+1, self.obs_encoding_size]

        # decode the obsgoal_encoding
        final_repr = self.decoder(obsgoal_encoding)
        
        # currently, the size is [batch_size, 32]
        dist_pred = self.dist_predictor(final_repr)
        action_pred = self.action_predictor(final_repr)

        # augment outputs to match labels size-wise
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred,
             self.num_action_params)
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
        """
        with torch.no_grad():
            z = z_distro.sample((n,))
            # repeat the observation encoding n times
            obs_encoding = obs_encoding.unsqueeze(0).repeat(n, 1, 1, 1)
            # combine the first two dimensions
            z = z.reshape(-1, z.shape[2])
            obs_encoding = obs_encoding.reshape(-1, obs_encoding.shape[2], obs_encoding.shape[3])
            dist_pred, action_pred = self._vib_decode(obs_encoding, z)
            # reshape the output
            dist_pred = dist_pred.reshape(n, -1, dist_pred.shape[-1])
            action_pred = action_pred.reshape(n, -1, action_pred.shape[1], action_pred.shape[2])
            return dist_pred, action_pred
        
    def forward(self,
             obs_encoding: torch.Tensor,
             z_distro: torch.Tensor,
             use_mean: bool = False,
             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the loss for the given observation encoding and latent space distribution.
        """
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
