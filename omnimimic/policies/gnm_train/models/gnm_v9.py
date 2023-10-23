import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from efficientnet_pytorch import EfficientNet
import vc_models
from .base_model import BaseModel
from .self_attention import MultiLayerDecoder
import pdb
import yaml

class GNM_V9_SA(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoder: Optional[str] = "efficientnet-b0",
        # goal_encoder: Optional[str] = None,
        goal_embedding_size: Optional[int] = None,
        obs_encoding_size: Optional[int] = 512,
        # freeze_obs_encoder: Optional[bool] = True,
        # freeze_goal_encoder: Optional[bool] = True,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        """
        GNMv9 class: Uses the architecture from v4, but uses a multi-headed self-attention block for the context.
        Goal-conditioning is standard: goal is concatenated to the context.
        """
        super(GNM_V9_SA, self).__init__(context_size, len_traj_pred, learn_angle)
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        # self.goal_encoder = goal_encoder

        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3) # context
            # self.model_transforms = lambda x: x
            self.num_obs_features = self.obs_encoder._fc.in_features
            self.obs_encoder_type = "efficientnet"
        else:
            raise NotImplementedError
        
        self.goal_encoder = EfficientNet.from_name(obs_encoder, in_channels=6) # obs+goal
        self.num_goal_features = self.goal_encoder._fc.in_features

        #self.goal_encoder_params = nn.ParameterList(list(self.goal_encoder.parameters()))

        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
        
        if self.num_goal_features != self.goal_encoding_size:
            self.compress_goal_enc = nn.Linear(self.num_goal_features, self.goal_encoding_size)
        else:
            self.compress_goal_enc = nn.Identity()

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
        self.test_predictor = nn.Linear(512, 32)

    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor, return_final_repr: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        ### OBS_GOAL ENCODING STUFF ##
        obsgoal_img = torch.cat([obs_img[:, 3*self.context_size:, :, :], goal_img], dim=1)
        obsgoal_embedding = self.goal_encoder.extract_features(obsgoal_img)
        obsgoal_embedding = self.goal_encoder._avg_pooling(obsgoal_embedding)
        if self.goal_encoder._global_params.include_top:
            obsgoal_embedding = obsgoal_embedding.flatten(start_dim=1)
            obsgoal_embedding = self.goal_encoder._dropout(obsgoal_embedding)
        # currently, the size is [batch_size, num_goal_features]
        obsgoal_embedding = self.compress_goal_enc(obsgoal_embedding)
        if len(obsgoal_embedding.shape) == 2:
            obsgoal_embedding = obsgoal_embedding.unsqueeze(1)
        # currently, the size of obsgoal_embedding is [batch_size, 1, self.goal_encoding_size]
        assert obsgoal_embedding.shape[2] == self.goal_encoding_size
        ###############################
        
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
        # # currently, the size is [batch_size, self.context_size+2, self.obs_encoding_size]

        obs_encoding = self.compress_obs_enc(obs_encoding)
        # currently, the size is [batch_size*(self.context_size + 1), self.obs_encoding_size]
        # reshape the obs_encoding to [context + 1, batch, encoding_size], note that the order is flipped
        obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)
        # currently, the size is [batch_size, self.context_size+1, self.obs_encoding_size]

        # # concatenate the goal embedding to the observation encoding
        obs_encoding = torch.cat((obs_encoding, obsgoal_embedding), dim=1)
        final_repr = self.decoder(obs_encoding)
        # currently, the size is [batch_size, 32]

        if return_final_repr:
            return final_repr

        dist_pred = self.dist_predictor(final_repr)
        action_pred = self.action_predictor(final_repr)

        # augment outputs to match labels size-wise
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        '''
        action_pred[:, :, :2] = torch.cumsum(
            action_pred[:, :, :2], dim=1
        )  # convert position deltas into waypoints
        '''
        '''
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )  # normalize the angle prediction
        '''
        return dist_pred, action_pred

    
    def constant_forward(self, old_obs: torch.tensor, new_obs: torch.tensor, goal_img: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]:
         # if self.goal_encoder is None:
        #     # goal image is goal embedding, precomputed
        #     goal_embedding = goal_img.float()  # cast to float
        # else:
        # concat obs and goal image

        ### OBS_GOAL ENCODING STUFF ###
        obsgoal_img = torch.cat([new_obs, goal_img], dim=1)
        obsgoal_embedding = self.goal_encoder.extract_features(obsgoal_img)
        obsgoal_embedding = self.goal_encoder._avg_pooling(obsgoal_embedding)
        if self.goal_encoder._global_params.include_top:
            obsgoal_embedding = obsgoal_embedding.flatten(start_dim=1)
            obsgoal_embedding = self.goal_encoder._dropout(obsgoal_embedding)
        # currently, the size is [batch_size, num_goal_features]
        obsgoal_embedding = self.compress_goal_enc(obsgoal_embedding)
        if len(obsgoal_embedding.shape) == 2:
            obsgoal_embedding = obsgoal_embedding.unsqueeze(1)
        # currently, the size of obsgoal_embedding is [batch_size, 1, self.goal_encoding_size]
        assert obsgoal_embedding.shape[2] == self.goal_encoding_size
        ###############################
        
        # split the observation into context based on the context size
        # image size is [batch_size, 3*self.context_size, H, W]
        obs_img = torch.split(old_obs, 3, dim=1)

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
        obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)
        # currently, the size is [batch_size, self.context_size+1, self.obs_encoding_size]

        # # concatenate the goal embedding to the observation encoding
        obs_encoding = torch.cat((obs_encoding, obsgoal_embedding), dim=1)
        # # currently, the size is [batch_size, self.context_size+2, self.obs_encoding_size]
        
        final_repr = self.decoder(obs_encoding)
        # currently, the size is [batch_size, 32]

        dist_pred = self.dist_predictor(final_repr)
        action_pred = self.action_predictor(final_repr)

        # augment outputs to match labels size-wise
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        action_pred[:, :, :3] = torch.cumsum(
            action_pred[:, :, :3], dim=1
        )  # convert position deltas into waypoints
        '''
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )  # normalize the angle prediction
        '''
        return dist_pred, action_pred
