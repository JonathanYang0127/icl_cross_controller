import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from efficientnet_pytorch import EfficientNet
from .base_model import BaseModel
from .self_attention import MultiLayerDecoder
import pdb

class ViNT_distilled(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
        pretrained: Optional[bool] = True,
        checkpoint_path: str = None,
        mode_dim: Optional[int] = 2,
        fuse_mode: Optional[bool] = False,
        freeze: Optional[bool] = False,
        deploy_mode: Optional[bool] = False,
    ) -> None:
        """
        GNMv9 class: Uses the architecture from v9, but uses a multi-headed self-attention block for the context.
        Goal-conditioning is standard: goal is concatenated to the context.
        """
        super(ViNT_distilled, self).__init__(context_size, len_traj_pred, learn_angle)
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.pretrained = pretrained
        self.checkpoint_path = checkpoint_path
        self.mode_dim = mode_dim
        self.fuse_mode = fuse_mode
        self.freeze = freeze

        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3) # context
        else:
            raise NotImplementedError
        
        if deploy_mode:
            self.goal_encoder = EfficientNet.from_name(obs_encoder, in_channels=6) # obs+goal
            self.num_goal_features = self.goal_encoder._fc.in_features

        self.num_obs_features = self.obs_encoder._fc.in_features

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

        if pretrained:
            self.load_pretrained(self.checkpoint_path, self.freeze)

        self.mode_encoder = nn.Sequential(
            nn.Linear(self.mode_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, self.obs_encoding_size),
        )
       

    def forward(
        self, obs_img: torch.tensor, mode_data: torch.tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        mode_embedding = self.mode_encoder(mode_data)
        mode_embedding = mode_embedding.unsqueeze(1)
        
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
        obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)
        # currently, the size is [batch_size, self.context_size+1, self.obs_encoding_size]

        # # concatenate the goal embedding to the observation encoding
        # pdb.set_trace()
        obs_encoding = torch.cat((obs_encoding, mode_embedding), dim=1)
        # # currently, the size is [batch_size, self.context_size+2, self.obs_encoding_size]
        
        final_repr = self.decoder(obs_encoding)
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
    
    def load_pretrained(self, checkpoint_path: str, freeze: bool) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        loaded_model = checkpoint["model"]
        try:  # for DataParallel
            state_dict = loaded_model.module.state_dict()
            self.load_state_dict(state_dict)
        except (RuntimeError, AttributeError) as e:
            state_dict = loaded_model.state_dict()
            self.load_state_dict(state_dict)

        # Remove goal embedding tower
        del self.goal_encoder

        if freeze:
            # Freeze the pretrained system
            for param in self.parameters():
                param.requires_grad = False
