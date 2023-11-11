from omnimimic.policies.gnm_train.models.gnm_v9 import GNM_V9_SA
from omnimimic.policies.gnm_train.models.vint import ViNT
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
            mha_ff_dim_factor=kwargs['mha_ff_dim_factor'],
            discrete=kwargs['discrete'],
            num_bins=kwargs['num_bins']
        )
        self.input_width, self.input_height = kwargs['input_width'], kwargs['input_height']
        
        self.lr = kwargs['lr']
        self.model_type = kwargs["model_type"]
        self.discrete = kwargs['discrete']
        
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
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=50
            )
            self.scheduler = GradualWarmupScheduler(
                    self.optimizer, 
                    multiplier=1,
                    total_epoch=5,
                    after_scheduler=self.scheduler,
                )
        else:
            self.scheduler = None

    def forward(self, image, goal_image):
        for transform in self.transforms:
            image = transform(image)
            goal_image = transform(goal_image)
        distance_pred, action_pred = super().forward(image, goal_image)
        return distance_pred, action_pred

    def configure_optimizers(self):
        return self.optimizer, self.scheduler


class GNMNumpyWrapper:
    def __init__(self, policy):
        self.policy = policy

    def forward_np(self, image, goal_image, task_embedding=None):
        '''
        Image is (batch, height, width, channels)
        Data type is uint8 and normalized
        '''
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).to(ptu.device)
        if task_embedding is not None:
            task_embedding = torch.from_numpy(task_embedding)

        goal_image = goal_image.astype(np.float32) / 255.0
        goal_image = torch.from_numpy(goal_image).to(ptu.device)

        image_data = torch.einsum('b h w c -> b c h w', image)
        goal_image_data = torch.einsum('b h w c -> b c h w', goal_image)
        return self.policy(image_data, goal_image_data)

    def compute_loss_np(self, image, goal_image, actions, goal_distances, task_embedding=None):
        distance_pred, action_pred = self.forward_np(image, goal_image, task_embedding)
        distance_pred, action_pred = distance_pred.float(), action_pred.float()
        
        #Use action_pred device for data parallelism
        actions = torch.from_numpy(np.array(actions)).to(action_pred.device)
        goal_distances = torch.from_numpy(np.array(goal_distances)).to(action_pred.device)
        goal_distances = goal_distances.reshape(-1, 1)
        '''        
        b_size = image.shape[0]
        action_pred = torch.cat((
            torch.zeros((b_size, 5, 1), dtype=torch.float64).to(ptu.device),
            -action_pred[:,:, 1:2],
            -action_pred[:,:, 0:1],
            torch.zeros((b_size, 5, 4), dtype=torch.float64).to(ptu.device)
            ), axis=-1) 
        actions = actions * 4
        '''

        if self.policy.discrete:   
            #Compute CE Loss
            ce_loss = F.cross_entropy(action_pred, actions, reduction='mean')

            #Compute Distance Loss
            goal_distance_loss = F.mse_loss(distance_pred, goal_distances, reduction="none").mean()

            loss = ce_loss + goal_distance_loss
            action_pred = self.sample_action(distance_pred, action_pred)
            return {'loss': loss, 
                    'ce_loss': ce_loss,
                    'goal_distance_loss': goal_distance_loss,
                    'action_pred': action_pred} 
        else:
            goal_distances, actions = goal_distances.float(), actions.float()

            #Compute MSE Loss
            mse_loss = F.mse_loss(action_pred.float(), actions.float(), reduction="none").mean()

            similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
            similarity_score = similarity(action_pred, actions).mean()

            l1_loss = F.l1_loss(action_pred, actions, reduction="none").mean()

            #Compute Distance Loss
            goal_distance_loss = F.mse_loss(distance_pred, goal_distances, reduction="none").mean() 

            loss = mse_loss + goal_distance_loss
            action_pred = self.sample_action(distance_pred, action_pred)
            return {'loss': loss, 'mse_loss': mse_loss, 
                'l1_loss': l1_loss,
                'goal_distance_loss': goal_distance_loss,
                'action_pred': original_action_pred, 
                'action_cosine_similarity': similarity_score}

    def get_action(self, image, goal_image, task_embedding=None):
        distance_pred, action_pred = self.forward_np(image, goal_image, task_embedding)
        return self.sample_action(distance_pred, action_pred)

    def sample_action(self, distance_pred, action_pred):
        if self.policy.discrete:
            probs = F.softmax(action_pred, dim=1)
            probs = torch.permute(probs, (0, 2, 3, 1))
            return torch.distributions.Categorical(probs).sample()
        else:
            return action_pred_dist.detach().cpu().numpy()

    def get_distance_pred(self, image, goal_image, task_embedding=None):
        distance_pred, action_pred_dist = self.forward_np(image, goal_image, task_embedding)
        return distance_pred.detach().cpu().numpy()



