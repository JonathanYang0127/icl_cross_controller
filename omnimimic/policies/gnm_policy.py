from omnimimic.policies.gnm_train.models.gnm_v9 import GNM_V9_SA
from omnimimic.policies.utils import get_augmentation_transforms

import numpy as np
import torch.nn.functional as F
from torch.optim import AdamW
import torch


class GNMPolicy(GNM_V9_SA):
    def __init__(self, **kwargs):
        super().__init__(
            context_size=kwargs['context_size'],
            len_traj_pred=kwargs['len_traj_pred'],
            learn_angle=kwargs['learn_angle'],
            obs_encoder=kwargs['obs_encoder'],
            obs_encoding_size=kwargs['obs_encoding_size'],
            mha_num_attention_heads=kwargs['mha_num_attention_heads'],
            mha_num_attention_layers=kwargs['mha_num_attention_layers'],
            mha_ff_dim_factor=kwargs['mha_ff_dim_factor']
        )
        self.input_width, self.input_height = kwargs['input_width'], kwargs['input_height']
        
        self.lr = kwargs['lr']
        self.model_type = kwargs["model_type"]
        self.no_goal = False
        
        self.optimizer_type = kwargs['optimizer']
        self.optimizer = AdamW(self.parameters(), lr=float(self.lr))

        self.augmentation_types = kwargs['image_augmentation']
        self.augmentation_config = dict()
        self.augmentation_config['input_dim'] = self.input_height 
        self.transforms = get_augmentation_transforms(
            self.augmentation_types,
            self.augmentation_config
        )
    
        #self.scheduler_type = kwargs['scheduler']
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #        optimizer, T_max=kwargs["epochs"]
        #    ) 

    def forward(self, image, goal_image):
        for transform in self.transforms:
            image = transform(image)
            goal_image = transform(goal_image)
        traj_dist_pred, traj_action_pred = super().forward(image, goal_image)
        return traj_dist_pred.squeeze(), traj_action_pred.squeeze()

    def forward_np(self, image, goal_image, task_embedding=None):
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).cuda()
        image = torch.einsum('b h w c -> b c h w', image)
        if task_embedding is not None:
            task_embedding = torch.from_numpy(task_embedding)

        if not self.no_goal:
            goal_image = goal_image.astype(np.float32) / 255.0
            goal_image = torch.from_numpy(goal_image).cuda()
            goal_image = image = torch.einsum('b h w c -> b c h w', goal_image)

        return super().forward(image, goal_image) 
        
    def compute_loss_np(self, image, goal_image, action, task_embedding=None):
        dist_pred, action_pred = self.forward_np(image, goal_image, task_embedding)
        action_label = torch.from_numpy(action).cuda()

        action_loss = F.mse_loss(action_pred.float(), action_label.float(), reduction="none").mean()
        print(action_loss.dtype, image.dtype, action.dtype)
        return {'loss': action_loss} 

    def get_action(self, image, goal_image, task_embedding=None):
        action_pred = self.forward_np(image, goal_image, task_embedding)

    def configure_optimizers(self):
        return self.optimizer, None #self.scheduler

