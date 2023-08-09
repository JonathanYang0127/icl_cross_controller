import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from .act_builder import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
from .base_policy import BaseTorchPolicy
import IPython
e = IPython.embed
import numpy as np


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()

        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def compute_loss(self, qpos, image, actions, is_pad):
        env_state = None
        #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
        #                                 std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225])
        #image = normalize(image)
        actions = actions[:, :self.model.num_queries]
        is_pad = is_pad[:, :self.model.num_queries]

        a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
        loss_dict = dict()
        all_l1 = F.l1_loss(actions, a_hat, reduction='none')
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        loss_dict['l1'] = l1
        loss_dict['kl'] = total_kld[0]
        loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
        return loss_dict

    def forward(self, qpos, image):
        env_state = None
        a_hat, _, (_, _) = self.model(qpos, image, env_state)
        return a_hat

    def configure_optimizers(self):
        return self.optimizer


def GCACTPolicy(ACTPolicy, BaseTorchPolicy):
    def __init__(self, args_override):
        super().__init__()

    def forward_np(self, image, goal_image):
        '''
        Image is (batch, height, width, channels)
        Data type is uint8
        '''

        image = image.astype(np.float32) / 255.0
        goal_image = goal_image.astype(np.float32) / 255.0
        curr_image = torch.from_numpy(image).cuda()
        goal_image = torch.from_numpy(goal_image).cuda()
        
        image_data = torch.stack([curr_image, goal_image], dim=1)
        image_data = torch.einsum('b k h w c -> b k c h w', image_data)
        image_data = image_data.flattten(1, 2)

        #Create dummy 0 qpos data (temporary)
        bs = qpos_data.shape[0]
        qpos_data = np.zeros([bc, 8], dtype='float32')

        return super().forward(qpos, image_data)

    def compute_loss_np(self, image, goal_image, actions):
 
        assert image.dtype == np.float32 and image.mean() <= 1
        
        curr_image = torch.from_numpy(curr_image.copy()).cuda()
        goal_image = torch.from_numpy(goal_image.copy()).cuda()
        
        image_data = torch.stack([curr_image, goal_image], dim=1) # bs, 2, 128, 128, 3
        image_data = torch.einsum('b k h w c -> b k c h w', image_data)
        image_data = image_data.flatten(1, 2)

        #Create dummy 0 qpos data and 0 is_pad data(temporary)
        bs = qpos_data.shape[0]
        qpos_data = torch.zeros([bs, 8])
        is_pad_data = torch.zeros([bs, 1])

        action_data = torch.from_numpy(action_data.copy()).cuda()
        
        return super().compute_loss(qpos_data, image_data, action_data, is_pad_data)

    def get_action(self, image, goal_image):
        action = self.forward_np(image, goal_image)
        return action.detach().numpy()


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

