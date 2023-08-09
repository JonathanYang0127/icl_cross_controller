import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from efficientnet_pytorch import EfficientNet
from .base_model import BaseModel
from .self_attention import MultiLayerDecoderResidualGoal
import pdb

class GNM_V8_SA(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoder: Optional[str] = "efficientnet-b0",
        goal_encoder: Optional[str] = None,
        obs_encoding_size: Optional[int] = 512,
        goal_embedding_size: Optional[int] = None,
        freeze_obs_encoder: Optional[bool] = True,
        freeze_goal_encoder: Optional[bool] = True,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        """
        GNM_V8_SA class: Uses the architecture from v3, but uses a multi-headed self-attention block for the context.
        Goal-conditioning is standard: goal is concatenated to the context.
        """
        super(GNM_V8_SA, self).__init__(context_size, len_traj_pred, learn_angle)
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.obs_encoder = obs_encoder
        self.goal_encoder = goal_encoder

        if self.obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(self.obs_encoder)
        else:
            raise NotImplementedError

        if goal_encoder is None:
            assert goal_embedding_size is not None
            assert freeze_goal_encoder is True
            self.num_goal_features = goal_embedding_size
        elif self.goal_encoder.split("-")[0] == "efficientnet":
            self.goal_encoder = EfficientNet.from_name(self.goal_encoder)
            self.num_goal_features = self.goal_encoder._fc.in_features
        else:
            raise NotImplementedError

        if freeze_obs_encoder:
            for param in self.obs_encoder.parameters():
                param.requires_grad = False
        self.num_obs_features = self.obs_encoder._fc.in_features
        
        if freeze_goal_encoder and goal_encoder is not None:
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

        self.decoder = MultiLayerDecoderResidualGoal(
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

    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.goal_encoder is None:
            # goal image is goal embedding, precomputed
            goal_embedding = goal_img.float()  # cast to float
        else:
            goal_embedding = self.goal_encoder.extract_features(goal_img)
            goal_embedding = self.goal_encoder._avg_pooling(goal_embedding)
            if self.goal_encoder._global_params.include_top:
                goal_embedding = goal_embedding.flatten(start_dim=1)
                goal_embedding = self.goal_encoder._dropout(goal_embedding)
            # currently, the size is [batch_size, num_goal_features]
        
        # split the observation into context based on the context size
        # image size is [batch_size, 3*self.context_size, H, W]
        obs_img = torch.split(obs_img, 3, dim=1)
        # image size is [batch_size*self.context_size, 3, H, W]
        obs_img = torch.concat(obs_img, dim=0)

        # get the observation encoding
        obs_encoding = self.obs_encoder.extract_features(obs_img)

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

        goal_embedding = self.compress_goal_enc(goal_embedding)
        goal_embedding = goal_embedding.unsqueeze(1)
        # currently, the size of goal_embedding is [batch_size, 1, self.goal_encoding_size]
        assert goal_embedding.shape[2] == self.goal_encoding_size

        obs_encoding = torch.cat((obs_encoding, goal_embedding), dim=1)
        
        # currently, the size is [batch_size, self.context_size+1, self.obs_encoding_size]
        
        final_repr = self.decoder(obs_encoding, goal_embedding)
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


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import List, Dict, Optional, Tuple
# from efficientnet_pytorch import EfficientNet
# from .base_model import BaseModel
# from .self_attention import MultiLayerDecoder, FiLM
# import pdb
# import numpy as np

# class GNM_V8_SA(BaseModel):
#     def __init__(
#         self,
#         context_size: int = 5,
#         len_traj_pred: Optional[int] = 5,
#         learn_angle: Optional[bool] = True,
#         image_size: Tuple[int] = (85, 64), # (W, H)
#         obs_encoder: Optional[str] = "efficientnet-b0",
#         goal_encoder: Optional[str] = None,
#         goal_embedding_size: Optional[int] = None,
#         obs_encoding_size: Optional[int] = 512,
#         freeze_obs_encoder: Optional[bool] = True,
#         freeze_goal_encoder: Optional[bool] = True,
#         mha_num_attention_heads: Optional[int] = 2,
#         mha_num_attention_layers: Optional[int] = 2,
#         mha_ff_dim_factor: Optional[int] = 4,
#     ) -> None:
#         """
#         GNM_V8_SA class: Uses the architecture from v3, but uses a multi-headed self-attention block for the context.
#         Goal-conditioning is standard: goal is concatenated to the context.
#         """
#         super(GNM_V8_SA, self).__init__(context_size, len_traj_pred, learn_angle)
#         self.obs_encoding_size = obs_encoding_size
#         self.goal_encoding_size = obs_encoding_size
#         self.obs_encoder = obs_encoder
#         self.goal_encoder = goal_encoder

#         if self.obs_encoder.split("-")[0] == "efficientnet":    
#             self.obs_encoder = EfficientNet.from_name(self.obs_encoder)
#         else:
#             raise NotImplementedError

#         if goal_encoder is None:
#             assert goal_embedding_size is not None
#             assert freeze_goal_encoder is True
#             self.num_goal_features = goal_embedding_size
#         elif self.goal_encoder.split("-")[0] == "efficientnet":
#             self.goal_encoder = EfficientNet.from_name(self.goal_encoder)
#             self.num_goal_features = self.goal_encoder._fc.in_features
#         else:
#             raise NotImplementedError

#         if freeze_obs_encoder:
#             for param in self.obs_encoder.parameters():
#                 param.requires_grad = False
#         self.num_obs_features = self.obs_encoder._fc.in_features
        
#         if freeze_goal_encoder and goal_encoder is not None:
#             for param in self.goal_encoder.parameters():
#                 param.requires_grad = False
#             self.num_goal_features = self.goal_encoder._fc.in_features

#         if self.num_obs_features != self.obs_encoding_size:
#             self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
#         else:
#             self.compress_obs_enc = nn.Identity()
        
#         if self.num_goal_features != self.goal_encoding_size:
#             self.compress_goal_enc = nn.Linear(self.num_goal_features, self.goal_encoding_size)
#         else:
#             self.compress_goal_enc = nn.Identity()

#         # get film dimensions
#         # dummy input image
#         dummy_img = torch.zeros((1, 3, *image_size[::-1])) # reverse the image size size torch expects (H, W)
#         dims = get_film_dimensions(self.obs_encoder, dummy_img)
#         self.film_layers = nn.ModuleList([FiLM(dim, self.num_goal_features) for dim in dims])

#         self.decoder = MultiLayerDecoder(
#             embed_dim=self.obs_encoding_size,
#             seq_len=self.context_size+1,
#             output_layers=[256, 128, 64, 32],
#             nhead=mha_num_attention_heads,
#             num_layers=mha_num_attention_layers,
#             ff_dim_factor=mha_ff_dim_factor,
#         )
#         self.dist_predictor = nn.Sequential(
#             nn.Linear(32, 1),
#         )
#         self.action_predictor = nn.Sequential(
#             nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
#         )

#     def forward(
#         self, obs_img: torch.tensor, goal_img: torch.tensor,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:

#         ######## GET GOAL EMBEDDING ########
#         if self.goal_encoder is None:
#             # goal image is goal embedding, precomputed
#             goal_embedding = goal_img.float()  # cast to float
#         else:
#             goal_embedding = self.goal_encoder.extract_features(goal_img)
#             goal_embedding = self.goal_encoder._avg_pooling(goal_embedding)
#             if self.goal_encoder._global_params.include_top:
#                 goal_embedding = goal_embedding.flatten(start_dim=1)
#                 goal_embedding = self.goal_encoder._dropout(goal_embedding)
#             # currently, the size is [batch_size, num_goal_features]

#         # currently, the size of goal_embedding is [batch_size, 1, self.goal_encoding_size]
#         # assert goal_embedding.shape[2] == self.goal_encoding_size

#         # goal_embedding size is [batch_size, self.context_size + 1, self.goal_encoding_size]
#         goal_embedding_repeat = torch.repeat_interleave(goal_embedding, self.context_size + 1, dim=0)
#         #####################################
        
#         # split the observation into context based on the context size
#         # image size is [batch_size, 3*(self.context_size + 1), H, W]
#         obs_img = torch.split(obs_img, 3, dim=1)

#         # image size is [batch_size, (self.context_size + 1), 3, H, W]
#         obs_img = torch.concat(obs_img, dim=0)

#         # # flatten first two dimensions
#         # # image size is [batch_size*(self.context_size + 1), 3, H, W]
#         # obs_img = obs_img.flatten(start_dim=0, end_dim=1)

#         # get the observation encoding
#         obs_encoding = extract_features_film(self.obs_encoder, obs_img, goal_embedding_repeat, self.film_layers)
#         # currently the size is [batch_size*(self.context_size + 1), 1280, H/32, W/32]

#         obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
#         # currently the size is [batch_size*(self.context_size + 1), 1280, 1, 1]

#         if self.obs_encoder._global_params.include_top:
#             obs_encoding = obs_encoding.flatten(start_dim=1)
#             obs_encoding = self.obs_encoder._dropout(obs_encoding)
#         obs_encoding = self.compress_obs_enc(obs_encoding)
#         # currently, the size is [batch_size*(self.context_size + 1), self.obs_encoding_size]
        
#         # reshape the obs_encoding to [context + 1, batch, encoding_size], note that the order is flipped
#         obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
#         obs_encoding = torch.transpose(obs_encoding, 0, 1)
#         # currently, the size is [batch_size, self.context_size+1, self.obs_encoding_size]

#         # goal_embedding = self.compress_goal_enc(goal_embedding)
#         # # goal_embedding = self.compress_goal_enc(goal_embedding)
#         # if len(goal_embedding.shape) == 2:
#         #     goal_embedding = goal_embedding.unsqueeze(1)

#         # # # torch cat the goal embedding to the obs_encoding
#         # obs_encoding = torch.cat((obs_encoding, goal_embedding), dim=1)
        
#         final_repr = self.decoder(obs_encoding)
#         # currently, the size is [batch_size, 32]

#         dist_pred = self.dist_predictor(final_repr)
#         action_pred = self.action_predictor(final_repr)

#         # augment outputs to match labels size-wise
#         action_pred = action_pred.reshape(
#             (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
#         )
#         action_pred[:, :, :2] = torch.cumsum(
#             action_pred[:, :, :2], dim=1
#         )  # convert position deltas into waypoints
#         if self.learn_angle:
#             action_pred[:, :, 2:] = F.normalize(
#                 action_pred[:, :, 2:].clone(), dim=-1
#             )  # normalize the angle prediction
#         return dist_pred, action_pred
    

# def get_film_dimensions(encoder, inputs):
#         dims = []
#         # Stem
#         x = encoder._swish(encoder._bn0(encoder._conv_stem(inputs)))

#         # Blocks
#         for idx, block in enumerate(encoder._blocks):
#             x = block(x)
#             # multiply all dims except batch size
#             if idx % 3 == 1:
#                 dims.append(np.prod(x.shape[1:]))
            
#         return dims
    
# def extract_features_film(encoder, inputs, goal, film_layers):
#     """use convolution layer to extract feature .

#     Args:
#         inputs (tensor): Input tensor.

#     Returns:
#         Output of the final convolution
#         layer in the efficientnet model.
#     """
#     # Stem
#     x = encoder._swish(encoder._bn0(encoder._conv_stem(inputs)))
#     film_idx = 0

#     # Blocks
#     for idx, block in enumerate(encoder._blocks):
#         drop_connect_rate = encoder._global_params.drop_connect_rate
#         if drop_connect_rate:
#             drop_connect_rate *= float(idx) / len(encoder._blocks)  # scale drop connect_rate  
#         x = block(x, drop_connect_rate=drop_connect_rate)
#         if idx % 3 == 1:
#             # FiLM layer
#             shape = x.shape
#             # use torch view to get flattened tensor
#             x = x.view(shape[0], -1)
#             x = film_layers[film_idx](x, goal)
#             # revert back to original shape
#             x = x.view(shape)
#             film_idx += 1

#     # Head
#     x = encoder._swish(encoder._bn1(encoder._conv_head(x)))

#     return x