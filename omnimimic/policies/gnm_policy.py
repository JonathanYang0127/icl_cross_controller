from omnimimic.policies.gnm_train.models.gnm_v9 import GNM_V9_SA
from omnimimic.policies.utils import get_augmentation_transforms

import numpy as np
import torch.nn.functional as F
from torch.optim import AdamW, Adam, SGD
from torch.distributions import constraints, Distribution, Independent, Normal
import torch
import omnimimic.torch.pytorch_util as ptu
from warmup_scheduler import GradualWarmupScheduler

class GaussianGNMPolicy(GNM_V9_SA):
    def __init__(
            self,
            action_dim=7,
            std=None,
            init_w=1e-3,
            min_log_std=-6,
            max_log_std=0,
            std_architecture="values",
            **kwargs
    ):
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
       
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.log_std = None
        self.std = std
        self.std_architecture = std_architecture
        last_hidden_size = 64
        self.last_fc = torch.nn.Linear(last_hidden_size, action_dim)
        if std is None:
            if self.std_architecture == "shared":
                self.last_fc_log_std = torch.nn.Linear(last_hidden_size, action_dim)
                self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
                self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
            elif self.std_architecture == "values":
                self.log_std_logits = torch.nn.Parameter(
                    torch.zeros(action_dim, requires_grad=True).to(ptu.device))
            else:
                raise ValueError(self.std_architecture)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def forward(self, image, goal_image):
        h = super().forward(image, goal_image, return_final_repr=True)
        mean = self.last_fc(h)
        distance = self.dist_predictor(h)
        #mean = torch.tanh(mean)        
        if self.std is None:
            if self.std_architecture == "shared":
                log_std = torch.sigmoid(self.last_fc_log_std(h))
            elif self.std_architecture == "values":
                log_std = torch.sigmoid(self.log_std_logits)
            else:
                raise ValueError(self.std_architecture)
            log_std = self.min_log_std + log_std * (
                        self.max_log_std - self.min_log_std)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(np.array([self.std, ])).float().to(
                ptu.device)

        return distance, Independent(Normal(mean, std), 1)


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
        if self.optimizer_type == 'sgd':
            self.optimizer = SGD(self.parameters(), lr=float(self.lr))
        elif self.optimizer_type == 'adam':
            self.optimizer = Adam(self.parameters(), lr=float(self.lr))
        elif self.optimizer_type == 'adamw':
            self.optimizer = AdamW(self.parameters(), lr=float(self.lr))


        self.augmentation_types = kwargs.get('image_augmentation', [])
        self.augmentation_config = dict()
        self.augmentation_config['input_dim'] = self.input_height 
        self.transforms = get_augmentation_transforms(
            self.augmentation_types,
            self.augmentation_config
        )
    
        self.scheduler_type = kwargs['scheduler']
        if kwargs['scheduler'] == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=kwargs["epochs"]
            )
            
            if kwargs['warmup']:
                self.scheduler = GradualWarmupScheduler(
                    self.optimizer,
                    multiplier=1,
                    total_epoch=kwargs["warmup_epochs"],
                    after_scheduler=self.scheduler,
                )
        elif kwargs['scheduler'] == 'cosinewarm':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=1, T_mult=2
            )
        else:
            self.scheduler = None

    def forward(self, image, goal_image):
        for transform in self.transforms:
            image = transform(image)
            goal_image = transform(goal_image)
        distance_pred, action_pred = super().forward(image, goal_image)
        return distance_pred, action_pred

    def forward_np(self, image, goal_image, task_embedding=None):
        '''
        Image is (batch, height, width, channels)
        Data type is uint8 and normalized
        '''
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).to(ptu.device)
        if task_embedding is not None:
            task_embedding = torch.from_numpy(task_embedding)

        if not self.no_goal:
            goal_image = goal_image.astype(np.float32) / 255.0
            goal_image = torch.from_numpy(goal_image).to(ptu.device)

        image_data = torch.einsum('b h w c -> b c h w', image)
        goal_image_data = torch.einsum('b h w c -> b c h w', goal_image)
        return self.forward(image_data, goal_image_data)

    def compute_loss_np(self, image, goal_image, actions, goal_distances, task_embedding=None):
        distance_pred, action_pred = self.forward_np(image, goal_image, task_embedding)
        actions = torch.from_numpy(actions).to(ptu.device)
        goal_distances = torch.from_numpy(goal_distances).to(ptu.device)

        mse_loss = F.mse_loss(action_pred.float(), actions.float(), reduction="none").mean()

        distance_mse_loss = F.mse_loss(distance_pred[0].float(), goal_distances.float(), reduction="none").mean()
        #loss = 0.01 * distance_mse_loss + mse_loss
        loss = mse_loss

        similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        similarity_score = similarity(action_pred, actions).mean()

        return {'loss': loss, 'mse_loss': mse_loss, 
                'action_pred': action_pred, 
                'action_cosine_similarity': similarity_score}

    def get_action(self, image, goal_image, task_embedding=None):
        distance_pred, action_pred_dist = self.forward_np(image, goal_image, task_embedding)
        return action_pred_dist.mean.detach().cpu().numpy()

    def get_distance_pred(self, image, goal_image, task_embedding=None):
        distance_pred, action_pred_dist = self.forward_np(image, goal_image, task_embedding)
        return distance_pred.detach().cpu().numpy()


    def configure_optimizers(self):
        return self.optimizer, self.scheduler

